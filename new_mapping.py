import os
import pdb
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from osgeo import gdal, osr
import os
import glob
import random
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import Canadian_mapping_model_func, MonteCarloConsistency, UnsupervisedPixelContrastLoss, UncertaintyEstimator
from trainer import Trainer_container, visualization_func
import json
from generate_geotiff_province_map import map_generation

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def parse_args():
    parser = argparse.ArgumentParser(description='Ice Classification Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Path to log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Path to checkpoint directory')
    parser.add_argument('--results_dir', type=str, default='./results', help='Path to results directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    return parser.parse_args()

class IceClassificationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, patch_size=256,
                 class_mapping={2: 0, 3: 1, 4: 2}, include_background=False,
                 add_hh_hv_ratio=True, ratio_epsilon=1e-6):
        """
        Dataset loader for ice classification with full 256x256 geospatial arrays.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.patch_size = patch_size
        self.class_mapping = class_mapping
        self.include_background = include_background
        self.add_hh_hv_ratio = add_hh_hv_ratio
        self.ratio_epsilon = ratio_epsilon

        self.class_names = {
            0: 'first_year_ice',
            1: 'young_ice',
            2: 'water',
            3: 'background' if include_background else None
        }

        # Load all patch file paths
        self.hh_files, self.hv_files, self.gt_files = self._load_file_paths()

        # Precompute geospatial information for each patch
        self.geo_info = self._precompute_geospatial_info()

        # Filter files based on class requirements
        self.valid_indices = self._filter_valid_patches()

        print(f"Loaded {len(self.valid_indices)} patches for {split} split")

    def _load_file_paths(self):
        """Load all file paths for HH, HV, and GT patches."""
        split_dir = os.path.join(self.root_dir, f"{self.split}_patches")

        hh_dir = os.path.join(split_dir, 'hh')
        hv_dir = os.path.join(split_dir, 'hv')
        gt_dir = os.path.join(split_dir, 'gt')

        # Get all GeoTIFF files
        hh_files = sorted(glob.glob(os.path.join(hh_dir, '*.tif')))
        hv_files = sorted(glob.glob(os.path.join(hv_dir, '*.tif')))
        gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.tif')))

        assert len(hh_files) == len(hv_files) == len(gt_files), \
            f"Mismatch in number of files: HH={len(hh_files)}, HV={len(hv_files)}, GT={len(gt_files)}"

        return hh_files, hv_files, gt_files

    def _precompute_geospatial_info(self):
        """Precompute geospatial information for each patch."""
        geo_info = []

        for gt_file in self.gt_files:
            ds = gdal.Open(gt_file)
            if ds is None:
                geo_info.append({
                    'geotransform': None,
                    'projection': None,
                    'filename': os.path.basename(gt_file)
                })
                continue

            geotransform = ds.GetGeoTransform()
            projection = ds.GetProjection()

            geo_info.append({
                'geotransform': geotransform,
                'projection': projection,
                'filename': os.path.basename(gt_file)
            })

            ds = None

        return geo_info

    def _create_lon_lat_arrays(self, geotransform, projection, patch_size):
        """Create 256x256 arrays of longitude and latitude coordinates."""
        if geotransform is None:
            # Return arrays of zeros if no geotransform available
            return (np.zeros((patch_size, patch_size), dtype=np.float32),
                    np.zeros((patch_size, patch_size), dtype=np.float32))

        # Create coordinate arrays
        x_coords = np.zeros((patch_size, patch_size), dtype=np.float64)
        y_coords = np.zeros((patch_size, patch_size), dtype=np.float64)

        # Calculate coordinates for each pixel
        for row in range(patch_size):
            for col in range(patch_size):
                x = geotransform[0] + col * geotransform[1] + row * geotransform[2]
                y = geotransform[3] + col * geotransform[4] + row * geotransform[5]
                x_coords[row, col] = x
                y_coords[row, col] = y

        # Convert to lat/lon if projected coordinates
        if projection and 'PROJCS' in projection:
            try:
                source_srs = osr.SpatialReference()
                source_srs.ImportFromWkt(projection)
                target_srs = osr.SpatialReference()
                target_srs.ImportFromEPSG(4326)  # WGS84

                transform = osr.CoordinateTransformation(source_srs, target_srs)

                # Transform coordinates - flatten arrays for batch processing
                x_flat = x_coords.flatten()
                y_flat = y_coords.flatten()
                coords = np.column_stack((x_flat, y_flat, np.zeros_like(x_flat)))

                # Transform all points at once
                transformed_coords = np.array(transform.TransformPoints(coords))

                # Reshape back to 2D arrays
                lon_array = transformed_coords[:, 0].reshape(patch_size, patch_size).astype(np.float32)
                lat_array = transformed_coords[:, 1].reshape(patch_size, patch_size).astype(np.float32)

                return lon_array, lat_array

            except Exception as e:
                print(f"Coordinate transformation failed: {e}")
                # Fall back to original coordinates
                return x_coords.astype(np.float32), y_coords.astype(np.float32)
        else:
            # Assume already in geographic coordinates
            return x_coords.astype(np.float32), y_coords.astype(np.float32)

    def _filter_valid_patches(self):
        """Filter patches to include only those with valid classes."""
        valid_indices = []

        for idx in range(len(self.gt_files)):
            gt_patch = self._read_geotiff(self.gt_files[idx])

            # Check if patch contains any of the target classes
            has_target_class = np.any(np.isin(gt_patch, list(self.class_mapping.keys())))
            has_background = self.include_background and np.any(gt_patch == 1)
            not_all_land = not np.all(gt_patch == 0)

            if (has_target_class or has_background) and not_all_land:
                valid_indices.append(idx)

        return valid_indices

    def _read_geotiff(self, file_path):
        """Read a GeoTIFF file and return as numpy array."""
        ds = gdal.Open(file_path)
        if ds is None:
            raise ValueError(f"Could not open file: {file_path}")
        array = ds.ReadAsArray()
        ds = None
        return array

    def _get_class_label(self, gt_patch):
        """Create a 256x256 label patch with class mapping applied."""
        label_patch = np.full_like(gt_patch, -1, dtype=np.int64)

        # Map valid classes (2,3,4)
        for original_class, mapped_class in self.class_mapping.items():
            class_mask = (gt_patch == original_class)
            label_patch[class_mask] = mapped_class

        # Optionally include background (class 1)
        if self.include_background:
            background_mask = (gt_patch == 1)
            label_patch[background_mask] = len(self.class_mapping)

        return label_patch

    def _create_multi_channel_input(self, hh_patch, hv_patch):
        """Create multi-channel input from HH, HV, and HH/HV ratio."""
        channels = []

        # Normalize and add HH channel
        hh_normalized = hh_patch.copy()
        # if np.any(hh_normalized > 0):
        #     hh_normalized = (hh_normalized - np.mean(hh_normalized)) / np.std(hh_normalized)
        channels.append(hh_normalized)

        # Normalize and add HV channel
        hv_normalized = hv_patch.copy()
        # if np.any(hv_normalized > 0):
        #     hv_normalized = (hv_normalized - np.mean(hv_normalized)) / np.std(hv_normalized)
        channels.append(hv_normalized)

        # Add HH/HV ratio channel if requested
        if self.add_hh_hv_ratio:
            # Avoid division by zero by adding epsilon
            hh_hv_ratio = self._calculate_hh_hv_ratio(hh_patch, hv_patch)

            # # Normalize ratio channel
            # if np.any(hh_hv_ratio > 0):
            #     hh_hv_ratio = (hh_hv_ratio - np.mean(hh_hv_ratio)) / np.std(hh_hv_ratio)
            channels.append(hh_hv_ratio*255)

        # Stack all channels
        multi_channel = np.stack(channels, axis=0)

        return multi_channel.astype(np.float32)

    def _calculate_hh_hv_ratio(self, hh_patch, hv_patch):
        """Calculate HH/HV ratio with safety against division by zero and invalid values."""
        # Convert to float32 for better numerical stability
        hh_safe = hh_patch.astype(np.float32)
        hv_safe = hv_patch.astype(np.float32)

        # Replace zeros and very small values in HV to avoid division by zero
        hv_safe[hv_safe < self.ratio_epsilon] = self.ratio_epsilon

        # Calculate ratio
        ratio = hh_safe / hv_safe

        # Handle infinite and NaN values
        ratio[~np.isfinite(ratio)] = 0.0  # Replace inf and NaN with 0

        # Apply log transform if desired (often better for ratio data)
        # ratio = np.log1p(ratio)  # log(1 + ratio) to handle zeros

        return ratio.astype(np.float32)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]

        try:
            # Read data
            hh_patch = self._read_geotiff(self.hh_files[actual_idx])
            hv_patch = self._read_geotiff(self.hv_files[actual_idx])
            gt_patch = self._read_geotiff(self.gt_files[actual_idx])

            # Get the full 256x256 label patch
            label_patch = self._get_class_label(gt_patch)

            # Check if patch contains any valid labels
            if np.all(label_patch == -1):
                return self.__getitem__((idx + 1) % len(self))

            # Create multi-channel input
            input_data = self._create_multi_channel_input(hh_patch, hv_patch)

            # Calculate HH/HV ratio separately
            hh_hv_ratio = self._calculate_hh_hv_ratio(hh_patch, hv_patch)

            # Get geospatial information
            geo_info = self.geo_info[actual_idx]

            # Create 256x256 longitude and latitude arrays
            lon_array, lat_array = self._create_lon_lat_arrays(
                geo_info['geotransform'],
                geo_info['projection'],
                self.patch_size
            )

            # Convert to tensors
            input_tensor = torch.from_numpy(input_data)
            label_tensor = torch.from_numpy(label_patch).long()
            lon_tensor = torch.from_numpy(lon_array.astype(np.float32))
            lat_tensor = torch.from_numpy(lat_array.astype(np.float32))

            # Apply transformations if specified
            if self.transform:
                input_tensor = self.transform(input_tensor)

            # Return comprehensive data with 256x256 coordinate arrays
            return {
                'input': input_tensor,  # Combined channels (2 or 3 channels)
                'label': label_tensor,  # 256×256 label map
                'hh': torch.from_numpy(hh_patch.astype(np.float32)),  # Raw HH channel
                'hv': torch.from_numpy(hv_patch.astype(np.float32)),  # Raw HV channel
                'hh_hv_ratio': torch.from_numpy(hh_hv_ratio),  # HH/HV ratio channel
                'longitude': lon_tensor,  # 256×256 longitude array
                'latitude': lat_tensor,  # 256×256 latitude array
                'filename': geo_info['filename'],
                'geotransform': geo_info['geotransform'],
                'projection': geo_info['projection'],
                'num_channels': input_data.shape[0]  # Number of input channels
            }

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return self.__getitem__((idx + 1) % len(self))

    def get_class_weights(self, weighting_strategy='balanced'):
        """Calculate class weights using pixel-level counts."""
        class_counts = {class_id: 0 for class_id in
                        range(len(self.class_mapping) + (1 if self.include_background else 0))}
        total_valid_pixels = 0

        for idx in tqdm(self.valid_indices, desc="Calculating class weights"):
            try:
                gt_patch = self._read_geotiff(self.gt_files[idx])
                label_patch = self._get_class_label(gt_patch)

                for class_id in class_counts.keys():
                    class_mask = (label_patch == class_id)
                    class_count = np.sum(class_mask)
                    class_counts[class_id] += class_count
                    total_valid_pixels += class_count

            except Exception as e:
                print(f"Error processing patch {idx}: {e}")
                continue

        # Calculate weights
        weights = {}
        for class_id, count in class_counts.items():
            if count > 0:
                weights[class_id] = total_valid_pixels / (len(class_counts) * count)
            else:
                weights[class_id] = 1.0

        return weights


# Updated visualization function to show coordinate arrays
def visualize_batch_with_coordinates(batch, class_names, num_samples=4):
    """Visualize a batch with HH, HV, HH/HV ratio, label map, and coordinate arrays."""

    fig, axes = plt.subplots(num_samples, 6, figsize=(24, 4 * num_samples))

    color_map = {
        -1: [0, 0, 0],  # Black - ignore/no data
        0: [1, 0, 0],  # Red - first year ice
        1: [0, 0, 1],  # Blue - young ice
        2: [0, 1, 0],  # Green - water
        3: [1, 1, 0]  # Yellow - background
    }

    for i in range(min(num_samples, len(batch['input']))):
        # HH channel
        im1 = axes[i, 0].imshow(batch['hh'][i].cpu().numpy(), cmap='gray')
        axes[i, 0].set_title(f'HH Channel', fontsize=10)
        axes[i, 0].axis('off')

        # HV channel
        im2 = axes[i, 1].imshow(batch['hv'][i].cpu().numpy(), cmap='gray')
        axes[i, 1].set_title(f'HV Channel', fontsize=10)
        axes[i, 1].axis('off')

        # HH/HV Ratio channel
        ratio_data = batch['hh_hv_ratio'][i].cpu().numpy()
        im3 = axes[i, 2].imshow(ratio_data, cmap='RdBu_r', vmin=-3, vmax=3)
        axes[i, 2].set_title(f'HH/HV Ratio', fontsize=10)
        axes[i, 2].axis('off')

        # Label map
        label_map = batch['label'][i].cpu().numpy()
        label_rgb = np.zeros((label_map.shape[0], label_map.shape[1], 3))

        for class_id, color in color_map.items():
            mask = (label_map == class_id)
            for c in range(3):
                label_rgb[..., c][mask] = color[c]

        im4 = axes[i, 3].imshow(label_rgb)
        axes[i, 3].set_title(f'Label Map', fontsize=10)
        axes[i, 3].axis('off')

        # Longitude array
        lon_data = batch['longitude'][i].cpu().numpy()
        im5 = axes[i, 4].imshow(lon_data, cmap='viridis')
        axes[i, 4].set_title(f'Longitude\nRange: [{lon_data.min():.4f}, {lon_data.max():.4f}]', fontsize=10)
        axes[i, 4].axis('off')
        plt.colorbar(im5, ax=axes[i, 4], fraction=0.046, pad=0.04)

        # Latitude array
        lat_data = batch['latitude'][i].cpu().numpy()
        im6 = axes[i, 5].imshow(lat_data, cmap='viridis')
        axes[i, 5].set_title(f'Latitude\nRange: [{lat_data.min():.4f}, {lat_data.max():.4f}]', fontsize=10)
        axes[i, 5].axis('off')
        plt.colorbar(im6, ax=axes[i, 5], fraction=0.046, pad=0.04)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = []
    for cls in sorted(color_map.keys()):
        if cls in class_names or cls == -1:
            color = color_map[cls]
            label_name = "Ignore/NoData" if cls == -1 else class_names.get(cls, f"Class {cls}")
            legend_elements.append(Patch(facecolor=color, edgecolor='black', label=label_name))

    fig.legend(handles=legend_elements, loc='upper right', fontsize=12, frameon=True)

    plt.tight_layout()
    plt.show()

# Custom collate function to handle dictionary returns
def custom_collate_fn(batch):
    """Custom collate function to handle dictionary returns with mixed data types."""
    if isinstance(batch[0], dict):
        # Handle dictionary returns
        result = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], torch.Tensor):
                result[key] = torch.stack([item[key] for item in batch])
            elif isinstance(batch[0][key], (int, float, str)):
                result[key] = [item[key] for item in batch]
            else:
                result[key] = [item[key] for item in batch]
        return result
    else:
        # Default collate for tuple returns
        return torch.utils.data.dataloader.default_collate(batch)


# Updated create_data_loaders function
def create_data_loaders(root_dir, batch_size=16, num_workers=4, include_background=False):
    """Create data loaders with geospatial information."""

    train_transform = transforms.Compose([
        # Add your transformations here
    ])

    # Create datasets
    train_dataset = IceClassificationDataset(root_dir, 'train', transform=train_transform, include_background=include_background)
    val_dataset = IceClassificationDataset(root_dir, 'val', include_background=include_background)
    test_dataset = IceClassificationDataset(root_dir, 'test', include_background=include_background)

    # Create data loaders with custom collate
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn)

    # Get class weights
    class_weights = train_dataset.get_class_weights()
    weight_tensor = torch.tensor([class_weights[i] for i in sorted(class_weights.keys())])

    return train_loader, val_loader, test_loader, weight_tensor, test_dataset

def main():
    args = parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
        data_info_params = config.get("Datainfo")
        model_params = config.get("Model_params")
        Train_setting_params = config.get("Train_setting")

    set_seed(model_params["seed"])

    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Your training code here using the config parameters
    # For example:
    train_loader, val_loader, test_loader, class_weights, test_dataset = create_data_loaders(
        root_dir=args.data_dir,
        batch_size=config['Train_setting']['batch_size'],
        num_workers=args.num_workers,
        include_background=False
    )

    batch = next(iter(train_loader))

    print("Batch keys:", batch.keys())
    print("Input shape:", batch['input'].shape)
    print("HH shape:", batch['hh'].shape)
    print("HV shape:", batch['hv'].shape)
    print("HH/HV Ratio shape:", batch['hh_hv_ratio'].shape)
    print("Longitude shape:", batch['longitude'].shape)  # Should be [batch_size, 256, 256]
    print("Latitude shape:", batch['latitude'].shape)  # Should be [batch_size, 256, 256]

    # Visualize with coordinate arrays
    class_names = {0: 'first_year_ice', 1: 'old_ice', 2: 'water'}
    visualize_batch_with_coordinates(batch, class_names)
    num_classes = Train_setting_params['num_classes']
    hidden_dim = model_params['hidden_dim']
    d_conv = model_params['d_conv']
    Num_channel = data_info_params['Num_channel']

    import torchvision.models.segmentation as seg_models
    model = seg_models.deeplabv3_resnet50(weights=False).to('cuda:0')

    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=3, stride=1).to('cuda:0')

    model = Canadian_mapping_model_func(num_classes=num_classes, dim=hidden_dim, d_conv=d_conv, in_channel=Num_channel)
    model.to(device)

    MC_dropout_model = MonteCarloConsistency(model, num_samples=2, num_classes=num_classes)
    uncertainty_estimator = UncertaintyEstimator(model, num_samples=10)
    criterion = nn.CrossEntropyLoss(ignore_index=Train_setting_params['ignore_index'])
    optimizer = torch.optim.Adam(model.parameters(), lr=Train_setting_params['learning_rate'])
    num_epochs = Train_setting_params['Epoch']

    ignore_index = Train_setting_params['ignore_index']

    model_save_path = os.path.join(args.results_dir, "model_weights.pth")
    Train_func = Trainer_container(Train_setting_params, model_save_path, model, MC_dropout_model, criterion, num_epochs, optimizer,
                                   train_loader, val_loader, test_loader, device, num_classes, ignore_index, 0.0)
    
    Train_func.train()
    #Train_func.test_()

    vis_results = visualization_func(uncertainty_estimator, model, model_save_path, args.results_dir, test_dataset, 15, device, "Figure")
    vis_results.vis_func()
    if Train_setting_params["predict_whole_map"] == True:
        print("start to predict whole map...")
        # Uncertainty_Scene = np.load("/mnt/storage/benchmark_datasets/output_Alberta/Uncertanity_map.npy", mmap_mode='r')
        # plt.imshow(Uncertainty_Scene, cmap='jet')
        # plt.savefig("A.png", dpi=150)
        # pdb.set_trace()
        class_label_list = [0, 1, 2]
        map_generation(device, model, uncertainty_estimator, num_classes,Train_setting_params["accuracy_report_name"], args.results_dir,
                       Train_setting_params["prediction_map_name"], model_save_path,
                       Train_setting_params['RCM_mosaic_image'], Train_setting_params["Uncertanity_map_name"],class_label_list)






# python3 new_mapping.py --config params.json --data_dir /beluga/Hack12_multi_type_seaice --num_workers 4 --gpu 0
if __name__ == "__main__":
    main()




