import pdb

import torch
import numpy as np
import yaml
import os
import rasterio
from collections import OrderedDict
from tqdm import tqdm
import traceback
import time
from osgeo import gdal, osr
from skimage.util import img_as_float
from datasketch import HyperLogLog
import seaborn as sns


def pot_CM(save_dir, confusion_matrix, confusion_classes):
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.1)
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    # Plot raw counts
    ax = sns.heatmap(confusion_matrix,annot=True,fmt='.2%', cmap='Blues',cbar_kws={'label': 'Count'},xticklabels=confusion_classes,yticklabels=confusion_classes)

    plt.title('Confusion Matrix (Raw Counts)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_dir + '/confusion_matrix2.png')

def write_classified_geotiff_with_colormap(output_path, Classified_Scene, Uncertainty_Scene, uncertainty_map_path, gdal_transform, projection_wkt):
    """
    Write classification results to GeoTIFF with embedded color table

    Args:
        output_path: Output file path
        Classified_Scene: 2D numpy array of class values
        gdal_transform: GDAL geotransform
        projection_wkt: Projection in WKT format
        CLASS_DEFINITIONS: Dictionary of class colors
    """
    # Get driver
    driver = gdal.GetDriverByName('GTiff')

    # Set up the output dataset with LZW compression
    options = [
        'COMPRESS=LZW',
        'PREDICTOR=2',  # Good for categorical data
        'TILED=YES',
        'NUM_THREADS=ALL_CPUS'
    ]

    # Get dimensions
    H, W = Classified_Scene.shape

    # Create output dataset
    out_ds = driver.Create(
        output_path,
        W,
        H,
        1,  # Single band
        gdal.GDT_Byte,  # Unsigned 8-bit
        options=options
    )

    # Set georeferencing
    out_ds.SetGeoTransform(gdal_transform)
    out_ds.SetProjection(projection_wkt)

    # Write data
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(Classified_Scene.astype(np.uint8))

    # Create color table
    # color_table = gdal.ColorTable()
    #
    # # Add colors for all defined classes
    # for class_val in CLASS_DEFINITIONS:
    #     r, g, b = CLASS_DEFINITIONS[class_val]['color']
    #     color_table.SetColorEntry(class_val, (r, g, b, 255))  # 255 = fully opaque
    #
    # # Set color table
    # out_band.SetRasterColorTable(color_table)
    out_band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

    # Set nodata value if needed (optional)
    out_band.SetNoDataValue(0)  # Assuming 0 is your nodata/unknown class

    # Set metadata for class names
    # class_names = [f"{k}:{v['name']}" for k, v in CLASS_DEFINITIONS.items()]
    # out_band.SetMetadata({'CLASS_NAMES': ','.join(class_names)})

    # Cleanup
    out_band.FlushCache()
    out_ds = None


    # out_uncertainty = driver.Create(
    #     uncertainty_map_path,
    #     W,
    #     H,
    #     1,  # Single band
    #     gdal.GDT_UInt16,  # Unsigned 8-bit
    # )
    #
    # # Set georeferencing
    # out_uncertainty.SetGeoTransform(gdal_transform)
    # out_uncertainty.SetProjection(projection_wkt)
    #
    # # Write data
    # out_band = out_uncertainty.GetRasterBand(1)
    # out_band.WriteArray((Uncertainty_Scene*10000).astype(np.uint16))
    # # out_band.WriteArray(scaled_data)
    #
    # # out_band.WriteArray(Uncertainty_Scene.astype(np.float32))
    #
    # # Cleanup
    # out_band.FlushCache()
    # out_uncertainty = None


def patch_generator(merged_scene, longtitude, latitude, patch_size):
    """
    Generator that yields patches from the merged scene.
    Args:
        merged_scene (np.ndarray): (H, W, C)
        patch_size (int): Size of the square patch
    Yields:
        (row, col, patch): Top-left row, col, and patch array
    """
    H, W, _ = merged_scene.shape
    for r in range(0, H, patch_size):
        for c in range(0, W, patch_size):
            patch = merged_scene[
                r:min(r+patch_size, H),
                c:min(c+patch_size, W),
                :
            ]
            long = longtitude[r:min(r+patch_size, H), c:min(c+patch_size, W)]

            lat = latitude[r:min(r+patch_size, H), c:min(c+patch_size, W)]
            # H, W, C = patch.shape[0], patch.shape[1], patch.shape[2]
            # data_reshaped = sentinel_img.reshape(-1, C)
            #
            # # 3. PCA降维
            # n_components = 3  # 降维到3个主成分
            # pca = PCA(n_components=n_components)
            # principal_components = pca.fit_transform(data_reshaped)  # [16384, 3]
            #
            # # 4. 恢复空间结构 [128, 128, n_components]
            # result = principal_components.reshape(H, W, n_components)
            # num_segments = 500  # 超像素数量，可根据需要调整
            # compactness = 10  # 紧凑度参数，控制超像素形状
            # segments = slic(result,
            #                 n_segments=num_segments,
            #                 compactness=compactness,
            #                 sigma=1,
            #                 start_label=1)

            yield r, c, patch, long, lat

def classify_scene_with_generator(
    merged_scene,
    longtitude,
    latitude,
    trained_model,
    uncertanity_estimator,
    patch_size=129,
    num_classes=14,
    device=None,
    batch_size=32,
    max_patches=None  # New argument for limiting number of patches
):
    H, W, _ = merged_scene.shape
    classified_scene = np.zeros((H, W), dtype=np.uint8)
    uncertainty_scene = np.zeros((H, W), dtype=np.float32)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model.to(device)
    trained_model.eval()

    # Calculate total number of patches for progress bar
    total_patches = (H // patch_size + (1 if H % patch_size > 0 else 0)) * (W // patch_size + (1 if W % patch_size > 0 else 0))
    if max_patches is not None:
        total_patches = min(total_patches, max_patches)

    patches = []
    coords = []
    spx = []
    longtitude_set = []
    latitude_set = []
    patch_counter = 0

    for r, c, patch, long_, lat_ in tqdm(patch_generator(merged_scene, longtitude, latitude, patch_size),
                           total=total_patches, 
                           desc="Classifying patches"):
        if max_patches is not None and patch_counter >= max_patches:
            break
        # Pad patch if at border
        ph, pw, _ = patch.shape
        if ph < patch_size or pw < patch_size:
            pad_h = patch_size - ph
            pad_w = patch_size - pw
            patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            long_ = np.pad(long_, ((0, pad_h), (0, pad_w)), mode='reflect')
            lat_ = np.pad(lat_, ((0, pad_h), (0, pad_w)), mode='reflect')
        patches.append(patch)
        longtitude_set.append(long_)
        latitude_set.append(lat_)
        coords.append((r, c, ph, pw))

        patch_counter += 1
        if len(patches) == batch_size:
            batch = torch.from_numpy(np.stack(patches).transpose(0, 3, 1, 2)).float().to(device)
            longtitude_ = torch.from_numpy(np.stack(longtitude_set)).float().to(device)
            latitude_ = torch.from_numpy(np.stack(latitude_set)).float().to(device)
            #spx_ = torch.from_numpy(np.stack(spx)).long().to(device)
            with torch.no_grad():
                logits = trained_model(batch, long=longtitude_, lat=latitude_)[0]
                # uncertainty_map = uncertanity_estimator.get_uncertainty(batch)[1]
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                # uncertainty_map = uncertainty_map.squeeze().cpu().numpy()
            for idx, (row, col, ph, pw) in enumerate(coords):
                classified_scene[row:row+ph, col:col+pw] = preds[idx][:ph, :pw]
                # uncertainty_scene[row:row + ph, col:col + pw] = uncertainty_map[idx][:ph, :pw]
            patches = []
            coords = []
            latitude_set = []
            longtitude_set = []

    # Handle remaining patches
    if patches:
        batch = torch.from_numpy(np.stack(patches).transpose(0, 3, 1, 2)).float().to(device)
        longtitude_ = torch.from_numpy(np.stack(longtitude_set)).float().to(device)
        latitude_ = torch.from_numpy(np.stack(latitude_set)).float().to(device)
        # spx_ = torch.from_numpy(np.stack(spx)).long().to(device)
        with torch.no_grad():
            logits = trained_model(batch, long=longtitude_, lat=latitude_)[0]
            # uncertainty_map = uncertanity_estimator.get_uncertainty(batch)[1]
            # uncertainty_map = uncertainty_map.squeeze().cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            # preds = LandCoverDataset.remap_to_original(preds)
        for idx, (row, col, ph, pw) in enumerate(coords):
            classified_scene[row:row+ph, col:col+pw] = preds[idx][:ph, :pw]
            # uncertainty_scene[row:row + ph, col:col + pw] = uncertainty_map[idx][:ph, :pw]
    return classified_scene

# Add band-specific ranges for preprocessing
band_ranges = {
    0: (900 , 3000),   # Band B02
    1: (1000, 3000),  # Band B03
    2: (1000, 3000),  # Band B04
    3: (800 , 8000),  # Band B08
    4: (1000, 4000),  # Band B05
    5: (900 , 6000),  # Band B06
    6: (900 , 8000),  # Band B07
    7: (900 , 8000),  # Band B8A
    8: (1000, 6000),  # Band B11
    9: (1000, 4000),  # Band B12
}


def _calculate_hh_hv_ratio(hh_patch, hv_patch):
    """Calculate HH/HV ratio with safety against division by zero and invalid values."""
    # Convert to float32 for better numerical stability
    hh_safe = hh_patch.astype(np.float32)
    hv_safe = hv_patch.astype(np.float32)

    # Replace zeros and very small values in HV to avoid division by zero
    hv_safe[hv_safe < 1e-6] = 1e-6

    # Calculate ratio
    ratio = hh_safe / hv_safe

    # Handle infinite and NaN values
    ratio[~np.isfinite(ratio)] = 0.0  # Replace inf and NaN with 0

    # Apply log transform if desired (often better for ratio data)
    # ratio = np.log1p(ratio)  # log(1 + ratio) to handle zeros

    return ratio.astype(np.float32)

def _normalize_band(band_data, band_idx):
    min_val, max_val = band_ranges[band_idx]
    clipped = np.clip(band_data, min_val, max_val)
    normalized = clipped / max_val
    return normalized

def _normalize_image(image):
    # image shape: (H, W, C)
    H, W, C = image.shape
    channels = []
    hh_patch, hv_patch = image[:, :, 0], image[:, :, 1]

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
        # Avoid division by zero by adding epsilon
    hh_hv_ratio = _calculate_hh_hv_ratio(hh_patch, hv_patch)

    #     # Normalize ratio channel
    # if np.any(hh_hv_ratio > 0):
    #     hh_hv_ratio = (hh_hv_ratio - np.mean(hh_hv_ratio)) / np.std(hh_hv_ratio)
    channels.append(hh_hv_ratio*255)

    # Stack all channels
    multi_channel = np.stack(channels, axis=2)

    return multi_channel


def generate_coordinate_grid(transform, width, height):
    """
    Generate longitude and latitude grids from affine transform
    """
    # Create arrays of pixel indices
    x = np.arange(width)
    y = np.arange(height)

    # Convert to coordinate arrays
    xx, yy = np.meshgrid(x, y)

    # Apply affine transformation to get coordinates
    lon_array = transform[0] + xx * transform[1] + yy * transform[2]
    lat_array = transform[3] + xx * transform[4] + yy * transform[5]

    return lon_array.astype(np.float32), lat_array.astype(np.float32)


def map_generation(device, model, uncertainty_estimator, num_classes,accuracy_filename, output_dir, output_map_name, model_path, sentinel2_img, uncertanity_map_name, class_label_list):

    device = device
    num_classes = num_classes
    output_filename = accuracy_filename
    output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_map_name)
    uncertainty_map_path = os.path.join(output_dir, uncertanity_map_name)
    # Load model weights
    ckpt_path = model_path
    model = model.to(device)
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    scene_dir = sentinel2_img
    ref_ds = gdal.Open(scene_dir)
    if ref_ds is None:
        raise ValueError(f"Could not open reference image: {scene_dir}")

    # Get dimensions
    W = ref_ds.RasterXSize
    H_total = ref_ds.RasterYSize
    # W = W // 32
    # H_total = H_total // 32
    # Get geotransform (equivalent to transform in rasterio)
    gdal_transform = ref_ds.GetGeoTransform()
    projection_wkt = ref_ds.GetProjection()  # This is already in WKT format
    # Get CRS
    crs = osr.SpatialReference()
    crs.ImportFromWkt(ref_ds.GetProjectionRef())

    print(f"Full scene dimensions: H={H_total}, W={W}")
    meta = {
        'driver': 'GTiff',  # Or whatever driver you'll use for output
        'dtype': 'uint8',  # Assuming your output will be uint8
        'nodata': None,  # Set if you have nodata values
        'width': ref_ds.RasterXSize,
        'height': ref_ds.RasterYSize,
        'count': 1,  # For your single-band output
        'crs': osr.SpatialReference(ref_ds.GetProjection()),
        'transform': ref_ds.GetGeoTransform(),
        'compress': 'LZW',  # Since you want LZW compression
        'tiled': True,  # If you want tiled output
        'blockxsize': 256,  # Typical tile size
        'blockysize': 256,
    }
    # Prepare full output array
    Classified_Scene = np.zeros((H_total, W), dtype=np.uint8)
    Uncertainty_Scene= np.zeros((H_total, W), dtype=np.float32)
    # Process in chunks for memory efficiency
    chunk_size = 300  # Process 200 rows at a time to reduce memory usage
    num_chunks = (H_total + chunk_size - 1) // chunk_size

    total_start_time = time.time()

    for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
        chunk_start_time = time.time()

        # Calculate chunk boundaries
        row_start = chunk_idx * chunk_size
        row_end = min(row_start + chunk_size, H_total)
        chunk_height = row_end - row_start

        print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks}: rows {row_start}-{row_end - 1}")

        # Load chunk from all bands
        # chunk_bands = []
        # for path in tqdm(scene_paths, desc=f"Loading bands for chunk {chunk_idx + 1}", leave=False):
        with rasterio.open(scene_dir) as src:

            window = rasterio.windows.Window(
                col_off=0, row_off=row_start, width=W, height=chunk_height
            )
            # merged_chunk = src.read(window=rasterio.windows.Window(
            #     col_off=0, row_off=row_start, width=W, height=chunk_height
            # ))
            merged_chunk = src.read(window=window)
            merged_chunk = merged_chunk.transpose(1, 2, 0)

            # Get the transform for this window
            transform = src.window_transform(window)

            # Generate coordinate arrays
            lon_array, lat_array = generate_coordinate_grid(transform, W, chunk_height)
            # print("unique", np.unique(merged_chunk))
            # chunk_bands.append(chunk)
        # Merge bands for this chunk
        # merged_chunk = np.stack(chunk_bands, axis=-1)
        # del chunk_bands  # Free memory
        # Preprocess: clip and normalize
        merged_chunk = _normalize_image(merged_chunk)
        print(f" Chunk shape after normalization: {merged_chunk.shape}")
        # Classify this chunk
        try:
            classified_chunk = classify_scene_with_generator(
                merged_scene=merged_chunk,
                longtitude=lon_array,
                latitude=lat_array,
                trained_model=model,
                uncertanity_estimator=uncertainty_estimator,
                patch_size=256,
                num_classes=num_classes,
                batch_size=32,
                max_patches=None
            )

            # Save chunk results to full output array
            Classified_Scene[row_start:row_end, :] = classified_chunk
            # Uncertainty_Scene[row_start:row_end, :] = uncertainty_chunk
            chunk_time = time.time() - chunk_start_time
            print(f"Chunk {chunk_idx + 1} completed in {chunk_time:.2f} seconds")

        except Exception as e:
            print(f"[ERROR] Failed to classify chunk {chunk_idx}: {e}")
            traceback.print_exc()
            continue

        # Free memory
        del merged_chunk, classified_chunk

    total_time = (time.time() - total_start_time) / 60
    print(f"\n[INFO] Total classification completed in {total_time:.2f} minutes")
    print(f"[INFO] Average time per chunk: {(total_time / num_chunks):.2f} minutes")
    # Classified_Scene = np.load("/mnt/storage/benchmark_datasets/Alberta/Map/Alberta_cloudfree_unfiltered_MCdropout.npy", mmap_mode='r')
    # n_rows, n_cols = Classified_Scene.shape
    #
    # # 2. Initialize HyperLogLog. The error rate is approx 1.04/sqrt(2^p)
    # # p=12 -> ~2.5KB memory, error rate ~1.6%
    # # p=14 -> ~10KB memory, error rate ~0.8% (better accuracy)
    # hll = HyperLogLog(p=14)
    #
    # # 3. Process the array in manageable chunks to avoid memory overload
    # chunk_size = 1000  # Process 10,000 rows at a time. Adjust based on your RAM.
    #
    # for i in range(0, n_rows, chunk_size):
    #     # Load a chunk of rows into actual RAM
    #     chunk = Classified_Scene[i:i + chunk_size, :]
    #
    #     # Flatten the chunk to a 1D array and update the HLL
    #     flattened_chunk = chunk.ravel()
    #     # Update HLL with each element. HLL handles the hashing internally.
    #     for item in flattened_chunk:
    #         hll.update(item.tobytes())  # Convert number to bytes for hashing
    #
    # # 4. Get the estimated cardinality (number of unique values)
    # estimated_unique_count = hll.count()
    # print(f"Estimated number of unique values: {estimated_unique_count}")
    # save the classified scene to a tif file
    # meta.update({"count": 1, "dtype": "uint8", "transform": transform, "crs": crs,  "nodata": 99})
    np.save(output_path.replace('.tif', '.npy'), Classified_Scene)
    np.save(uncertainty_map_path.replace('.tif', '.npy'), Uncertainty_Scene)
    print(f"Saving to: {output_path}")
    print(f"Classified_Scene shape: {Classified_Scene.shape}")

    print(f"Uncertainty Map Saving to: {uncertainty_map_path}")
    # print(f"Meta info: {meta}")
    print(f"Data type: {Classified_Scene.dtype}")

    # Save as GeoTIFF
    # Create the output GeoTIFF file
    #Uncertainty_Scene = np.load("/mnt/storage/benchmark_datasets/output_Alberta/Uncertanity_map.npy", mmap_mode='r')
    write_classified_geotiff_with_colormap(output_path, Classified_Scene, Uncertainty_Scene, uncertainty_map_path, gdal_transform, projection_wkt)
    # driver = gdal.GetDriverByName('GTiff')
    #
    # # Set up the output dataset with LZW compression
    # options = [
    #     'COMPRESS=LZW',
    #     'PREDICTOR=2',  # Good for categorical data
    #     'TILED=YES',
    #     'NUM_THREADS=ALL_CPUS'
    # ]
    #
    # out_ds = driver.Create(
    #     output_path,
    #     W,
    #     H_total,
    #     1,  # Number of bands
    #     gdal.GDT_Byte,  # Unsigned 8-bit type (uint8)
    #     options=options
    # )
    #
    # # Set the geotransform and projection
    # out_ds.SetGeoTransform(gdal_transform)
    # out_ds.SetProjection(projection_wkt)  # Use the WKT string directly
    #
    # # Write the data (assuming Classified_Scene is 2D)
    # out_band = out_ds.GetRasterBand(1)
    # out_band.WriteArray(Classified_Scene.astype(np.uint8))
    #
    # # Set NoData value if needed
    # # out_band.SetNoDataValue(0)
    #
    # # Close the dataset to write to disk
    # out_band.FlushCache()
    # out_ds = None
    # ref_ds = None
    # with rasterio.open(output_path, 'w', **meta) as dst:
    #     dst.write(Classified_Scene[np.newaxis, :, :].astype(np.uint8))
    print(f"[INFO] Final classified map saved with geospatial info: {output_path}")

    # --- Calculate Accuracy Report with Chunked Per-Class Metrics ---
    # print("\n[INFO] Calculating accuracy report...")
    #
    # # Class names for report
    # CLASS_NAMES = [
    #     'Unknown',  # 0
    #     'Temperate needleleaf forest',  # 1
    #     'Sub-polar taiga forest',  # 2
    #     'Unknown',
    #     'Unknown',
    #     'Temperate broadleaf forest',  # 5
    #     'Mixed forest',  # 6
    #     'Unknown',
    #     'Temperate shrubland',  # 8
    #     'Unknown',
    #     'Temperate grassland',  # 10
    #     'Polar shrubland-lichen',  # 11
    #     'Polar grassland-lichen',  # 12
    #     'Polar barren-lichen',  # 13
    #     'Wetland',  # 14
    #     'Cropland',  # 15
    #     'Barren lands',  # 16
    #     'Urban',  # 17
    #     'Water',  # 18
    #     'Snow/ice',  # 19
    # ]
    #
    # with rasterio.open(gt_path) as src:
    #     gt_labels = src.read(1).astype(np.uint8)
    # with rasterio.open(scl_mask_path) as src:
    #     scl_mask = src.read(1).astype(np.uint8)
    # classified_scene = Classified_Scene.astype(np.uint8)
    # if gt_labels.shape != classified_scene.shape or scl_mask.shape != classified_scene.shape:
    #     print("[WARNING] Shape mismatch detected. Cropping to smallest common size.")
    #     min_h = min(gt_labels.shape[0], scl_mask.shape[0], classified_scene.shape[0])
    #     min_w = min(gt_labels.shape[1], scl_mask.shape[1], classified_scene.shape[1])
    #     gt_labels = gt_labels[:min_h, :min_w]
    #     scl_mask = scl_mask[:min_h, :min_w]
    #     classified_scene = classified_scene[:min_h, :min_w]
    #     print(f"Cropped to shape: {classified_scene.shape}")
    #
    # valid_scl_values = [4, 5, 6]
    # block_size = 100
    # height, width = classified_scene.shape
    # class_labels = np.array(class_label_list)
    #
    # # Exclude background (0) for confusion matrix, but keep it for other calculations
    # confusion_classes = class_labels[class_labels != 0]
    # n_confusion_classes = len(confusion_classes)
    #
    # # class_labels = np.unique(np.concatenate([np.unique(gt_labels), np.unique(classified_scene)]))
    # class_labels = class_labels[class_labels != 0]
    # print("Class_labels:", class_labels)
    # n_classes = class_labels.max() + 1
    # correct = np.zeros(n_classes, dtype=np.uint64)
    # pred_count = np.zeros(n_classes, dtype=np.uint64)
    # gt_count = np.zeros(n_classes, dtype=np.uint64)
    # total_valid = 0
    # total_correct = 0
    # # Initialize confusion matrix only for non-background classes
    # confusion_matrix = np.zeros((n_confusion_classes, n_confusion_classes), dtype=np.uint64)
    #
    # # Create mappings for fast lookup
    # class_to_idx = {class_val: idx for idx, class_val in enumerate(class_labels)}
    # confusion_class_to_idx = {class_val: idx for idx, class_val in enumerate(confusion_classes)}
    #
    # for row_start in tqdm(range(0, height, block_size), desc="Processing blocks"):
    #     for col_start in range(0, width, block_size):
    #         row_end = min(row_start + block_size, height)
    #         col_end = min(col_start + block_size, width)
    #         gt_block = gt_labels[row_start:row_end, col_start:col_end]
    #         pred_block = classified_scene[row_start:row_end, col_start:col_end]
    #         scl_block = scl_mask[row_start:row_end, col_start:col_end]
    #         valid_mask = np.isin(scl_block, valid_scl_values)
    #         valid_classified_mask = (pred_block != -99)
    #         valid_gt_mask = (gt_block > 0) & (gt_block < 255)
    #         final_valid_mask = valid_mask & valid_gt_mask & valid_classified_mask
    #         valid_gt = gt_block[final_valid_mask]
    #         valid_pred = pred_block[final_valid_mask]
    #         total_valid += valid_gt.size
    #         total_correct += np.sum(valid_gt == valid_pred)
    #         #
    #         # # Update confusion matrix (only for non-background classes)
    #         # for true_class in confusion_classes:
    #         #     for pred_class in confusion_classes:
    #         #         true_idx = confusion_class_to_idx[true_class]
    #         #         pred_idx = confusion_class_to_idx[pred_class]
    #         #         count = np.sum((valid_gt == true_class) & (valid_pred == pred_class))
    #         #         confusion_matrix[true_idx, pred_idx] += 1
    #         # Alternative: Vectorized approach (much faster)
    #         for true_class in confusion_classes:
    #             true_mask = (valid_gt == true_class)
    #             if np.any(true_mask):
    #                 true_idx = confusion_class_to_idx[true_class]
    #                 for pred_class in confusion_classes:
    #                     pred_idx = confusion_class_to_idx[pred_class]
    #                     count = np.sum((valid_gt == true_class) & (valid_pred == pred_class))
    #                     confusion_matrix[true_idx, pred_idx] += count
    #
    #         for c in class_labels:
    #             gt_c = (valid_gt == c)
    #             pred_c = (valid_pred == c)
    #             correct[c] += np.sum(gt_c & pred_c)
    #             pred_count[c] += np.sum(pred_c)
    #             gt_count[c] += np.sum(gt_c)
    # overall_accuracy = total_correct / total_valid if total_valid > 0 else 0.0
    # precision = np.zeros(n_classes)
    # recall = np.zeros(n_classes)
    # f1 = np.zeros(n_classes)
    # support = gt_count
    # for c in class_labels:
    #     if pred_count[c] > 0:
    #         precision[c] = correct[c] / pred_count[c]
    #     if gt_count[c] > 0:
    #         recall[c] = correct[c] / gt_count[c]
    #     if precision[c] + recall[c] > 0:
    #         f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])
    # print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)")
    # print(f"\nPer-class metrics:")
    # print(f"{'Class':<35} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    # print("-" * 75)
    # for c in class_labels:
    #     class_name = CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else f"Unknown ({c})"
    #     print(f"{class_name:<35} {precision[c]:<10.4f} {recall[c]:<10.4f} {f1[c]:<10.4f} {support[c]:<10}")
    # accuracy_report_path = os.path.join(output_dir, output_filename)
    #
    # with open(accuracy_report_path, 'w') as f:
    #     f.write(f"Accuracy Report for LandCover Mapping\n")
    #     f.write("=" * 50 + "\n\n")
    #     f.write(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy * 100:.2f}%)\n\n")
    #     f.write("Per-class metrics:\n")
    #     f.write(f"{'Class':<45} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
    #     f.write("-" * 75 + "\n")
    #     for c in class_labels:
    #         class_name = CLASS_NAMES[c] if 0 <= c < len(CLASS_NAMES) else f"Unknown ({c})"
    #         f.write(f"{class_name:<45} {precision[c]:<10.4f} {recall[c]:<10.4f} {f1[c]:<10.4f} {support[c]:<10}\n")
    #     f.write(f"\nValid pixels used: {total_valid}\n")
    #     f.write(f"Ground truth unique values: {class_labels}\n")
    #     f.write(f"Predicted unique values: {class_labels}\n")
    # print(f"\n[INFO] Accuracy report saved to: {accuracy_report_path}")
    #
    # pot_CM(output_dir, confusion_matrix, confusion_classes)
    # print(f"\n[INFO] Confusion Matrix plotted")
    # --- Save Output Files ---
    # Print unique values and their counts before saving
    # unique_vals, counts = np.unique(Classified_Scene, return_counts=True)
    # print(f"\nUnique values in Classified_Scene before saving: {unique_vals}")
    # print("Value counts:")
    # for val, count in zip(unique_vals, counts):
    #     print(f"  Class {val}: {count} pixels")
