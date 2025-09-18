import os
import pdb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from model import Canadian_mapping_model_func, PrototypicalContrastiveLoss, MonteCarloConsistency, UnsupervisedPixelContrastLoss
import torch
import torch.nn as nn
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score
import random
import rasterio
from rasterio.windows import Window
import h5py
from torch.utils.data import Dataset, DataLoader
from skimage.util import img_as_float
import numpy as np
from sklearn.decomposition import PCA
from itertools import islice

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ==================== CONFIGURATION ====================

def valid_mask(mask, ignore_index=-99):
    # Checks if there are any pixels that are NOT the ignore_index
    return (mask != ignore_index).any()

def calculate_metrics(predictions, targets, num_classes, ignore_index=-99):
    """
    Calculates segmentation metrics including overall accuracy, mean accuracy, IoU, and Cohen's Kappa.
    Only calculates metrics for classes that exist in the ground truth data.

    Args:
        predictions (torch.Tensor): Model's predicted class indices (e.g., from argmax).
                                    Shape: (batch_size, H, W)
        targets (torch.Tensor): Ground truth class indices.
                                Shape: (batch_size, H, W)
        num_classes (int): Total number of possible classes (excluding ignore_index).
        ignore_index (int, optional): The class index to ignore in metric calculations. Defaults to -1.

    Returns:
        dict: A dictionary containing overall accuracy, mean accuracy, mean IoU,
              Cohen's Kappa, and per-class IoU and accuracy (only for present classes).
    """
    # Ensure tensors are on CPU for numpy operations
    if predictions.is_cuda:
        predictions = predictions.cpu()
    if targets.is_cuda:
        targets = targets.cpu()

    # Flatten tensors
    predictions_flat = predictions.view(-1)
    targets_flat = targets.view(-1)

    # Filter out ignore_index pixels
    valid_pixels_mask = (targets_flat != ignore_index)
    predictions_valid = predictions_flat[valid_pixels_mask]
    targets_valid = targets_flat[valid_pixels_mask]

    # 1. Overall Accuracy
    correct_predictions_overall = (predictions_valid == targets_valid).sum().item()
    total_valid_pixels = valid_pixels_mask.sum().item()
    overall_accuracy_valid = correct_predictions_overall / total_valid_pixels if total_valid_pixels > 0 else 0.0

    # 2. Cohen's Kappa - WITH ROBUSTNESS FIX
    # Get the unique classes that are present in the ground truth OR the predictions.
    # This is crucial to define the shape of the confusion matrix correctly.
    all_possible_classes_in_data = np.union1d(np.unique(targets_valid), np.unique(predictions_valid))
    # If there's only one class, Kappa is undefined (it's always perfect agreement or total disagreement)
    if len(all_possible_classes_in_data) < 2:
        kappa = 0.0 # or np.nan, but 0.0 is safer for logging
    else:
        # Use sklearn's function but ensure labels are defined to include all present classes.
        # This prevents the function from assuming a different shape for the matrix.
        kappa = cohen_kappa_score(targets_valid.numpy(),
                                 predictions_valid.numpy(),
                                 labels=all_possible_classes_in_data)

    # 3. Find classes that actually exist in the data
    present_classes = torch.unique(targets_valid)
    present_classes = present_classes[present_classes != ignore_index].tolist()

    # 4. Per-Class Metrics (only for present classes)
    per_class_accuracy = {}
    per_class_iou = {}
    valid_class_accuracies = []
    valid_class_ious = []

    for class_id in present_classes:
        # Accuracy calculation
        target_class_pixels = (targets_valid == class_id)
        correct_predictions_class = (predictions_valid[target_class_pixels] == class_id).sum().item()
        total_target_class_pixels = target_class_pixels.sum().item()

        acc_class = correct_predictions_class / total_target_class_pixels if total_target_class_pixels > 0 else 0.0
        per_class_accuracy[class_id] = acc_class

        # IoU calculation
        intersection = ((predictions_valid == class_id) & (targets_valid == class_id)).sum().item()
        union = ((predictions_valid == class_id) | (targets_valid == class_id)).sum().item()

        iou_class = intersection / union if union > 0 else 0.0
        per_class_iou[class_id] = iou_class

        valid_class_accuracies.append(acc_class)
        valid_class_ious.append(iou_class)

    # 5. Mean Accuracy and Mean IoU (only for present classes)
    mean_accuracy_valid_classes = np.mean(valid_class_accuracies) if valid_class_accuracies else 0.0
    mean_iou_valid_classes = np.mean(valid_class_ious) if valid_class_ious else 0.0

    metrics = {
        "Overall Accuracy (valid classes)": overall_accuracy_valid,
        "Cohen's Kappa": kappa,
        "Mean Accuracy (valid classes)": mean_accuracy_valid_classes,
        "Mean IoU (valid classes)": mean_iou_valid_classes,
        "Per-Class Accuracy": per_class_accuracy,
        "Per-Class IoU": per_class_iou,
        "Present Classes": present_classes  # Add which classes were actually present
    }

    return metrics

def compute_superpixel_gt_mode_torch(gt_labels, superpixels):
    """
    Args:
        gt_labels: GT标签图, shape=[batch_size, H, W], 类别从0开始，背景=-1
        superpixels: 超像素分割图, shape=[batch_size, H, W], 超像素ID从1开始
    Returns:
        superpixel_gt_modes: shape=[batch_size, max_num_superpixels], 无效区域填充-1
    """
    batch_size, H, W = gt_labels.shape
    device = gt_labels.device

    # 1. 计算每张影像的超像素数量（标签从1开始，直接取max）
    num_superpixels_per_image = [superpixels[i].max().item() for i in range(batch_size)]
    max_num_superpixels = max(num_superpixels_per_image)

    # 2. 初始化结果矩阵（默认填充-1）
    superpixel_gt_modes = torch.full(
        (batch_size, max_num_superpixels),
        -99,
        dtype=torch.long,
        device=device
    )

    for i in range(batch_size):
        # 3. 展平GT和超像素标签
        gt_flat = gt_labels[i].view(-1)  # [H*W]
        sp_flat = superpixels[i].view(-1)  # [H*W]
        current_num_superpixels = num_superpixels_per_image[i]

        # 4. 预处理GT：将背景类（-1）临时映射为max_gt_label+1（不参与众数竞争）
        max_gt_label = gt_flat.max().item() + 1  # 有效类别是0,1,...,max_gt_label-1
        gt_processed = torch.where(gt_flat == -99, max_gt_label, gt_flat)

        # 5. 向量化计算众数
        # 5.1 生成线性索引：(sp_id - 1) * (max_gt_label + 1) + gt_processed
        linear_indices = (sp_flat - 1) * (max_gt_label + 1) + gt_processed

        # 5.2 统计频次（排除背景类的影响）
        counts = torch.bincount(
            linear_indices,
            minlength=current_num_superpixels * (max_gt_label + 1)
        )
        counts = counts.view(current_num_superpixels, max_gt_label + 1)

        # 5.3 找到众数（忽略临时背景类max_gt_label）
        valid_counts = counts[:, :max_gt_label]  # [current_num_superpixels, max_gt_label]
        modes = torch.argmax(valid_counts, dim=1)  # [current_num_superpixels]

        # 5.4 处理全背景的超像素块（众数为max_gt_label时，填充-1）
        bg_mask = (valid_counts.sum(dim=1) == 0)  # 该超像素块全是背景
        modes[bg_mask] = -99

        # 6. 存储结果
        superpixel_gt_modes[i, :current_num_superpixels] = modes

    return superpixel_gt_modes



class Trainer_container:
    def __init__(self, param, model_save_path, model, MC_dropout_model, criterion, num_epochs, optimizer, train_loader, val_loader, test_loader, device, num_classes, ignore_index, best_val):

        self.param = param
        self.model_save_path = model_save_path
        self.model = model
        self.MC_dropout_model = MC_dropout_model
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.criterion = criterion
        self.best_val = best_val

    def train(self):

        for epoch in range(self.num_epochs):
            # Print epoch header first
            print(f"\n{'=' * 30}")
            print(f"Epoch {epoch + 1}/{self.num_epochs}")
            print(f"{'=' * 30}")
            # Train and validate
            self.train_one_epoch(epoch)
            if (epoch + 1) % 5 == 0:
                val_metrics = self.validate()
                if val_metrics['acc'] > self.best_val:
                    self.best_val = val_metrics['acc']
                    torch.save(self.model.state_dict(), self.model_save_path)
                    print(f"✅ Model saved to {self.model_save_path}")

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        count = 0
        metrics_accumulator = {
            'overall_acc': 0.0,
            'mean_acc': 0.0,
            'mean_iou': 0.0,
            'kappa': 0.0,
            'batch_count': 0
        }
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, sample in enumerate(pbar):
            hh = sample['hh'].to(self.device).unsqueeze(dim=1)
            hv = sample['hv'].to(self.device).unsqueeze(dim=1)
            hhhv = sample['hh_hv_ratio'].to(self.device).unsqueeze(dim=1)
            mask = sample['label'].to(self.device)
            longitude = sample['longitude'].to(self.device)
            latitude = sample['latitude'].to(self.device)

            images = torch.cat([hh, hv], dim=1)
            images = torch.cat([images, hhhv], dim=1)

            mc_outputs, centers1, centers2 = self.MC_dropout_model(images, mask, longitude, latitude, epoch)
            # updata class center
            losses = self.MC_dropout_model.compute_total_loss(epoch, mc_outputs, centers1, centers2, mask, images,
                                                 loss_weights={'ce': self.param["ce_weight"], 'mse': self.param["mse_weight"], 'edge': self.param["edge_weight"],
                                                               'js_kl': self.param["js_kl_weight"]})
            loss = losses['total_loss']
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            pbar.set_postfix({
                'CE': f"{losses['ce_loss'].item():.4f}",
                'KL': f"{losses['js_consistency'].item():.4f}",
                'MSE': f"{losses['mse_consistency'].item():.4f}",
                "prototype_loss": f"{losses['prototype_loss']:.4f}",
            })

            _, preds = torch.max(mc_outputs.mean(dim=0), dim=1)  # shape: [B, H, W]

            # Calculate metrics
            batch_metrics = calculate_metrics(
                preds, mask,
                num_classes=self.num_classes,
                ignore_index=self.ignore_index
            )

            # Accumulate metrics for epoch average
            metrics_accumulator['overall_acc'] += batch_metrics["Overall Accuracy (valid classes)"]
            metrics_accumulator['mean_acc'] += batch_metrics["Mean Accuracy (valid classes)"]
            metrics_accumulator['mean_iou'] += batch_metrics["Mean IoU (valid classes)"]
            metrics_accumulator['kappa'] += batch_metrics["Cohen's Kappa"]
            metrics_accumulator['batch_count'] += 1
            running_loss += loss.item()
            count += 1

        # Calculate epoch averages
        epoch_loss = running_loss / count if count > 0 else float("nan")
        avg_metrics = {
            'loss': epoch_loss,
            'OA': metrics_accumulator['overall_acc'] / metrics_accumulator['batch_count'] if
            metrics_accumulator[
                'batch_count'] > 0 else 0.0,
            'mean_acc': metrics_accumulator['mean_acc'] / metrics_accumulator['batch_count'] if
            metrics_accumulator[
                'batch_count'] > 0 else 0.0,
            'mean_iou': metrics_accumulator['mean_iou'] / metrics_accumulator['batch_count'] if
            metrics_accumulator[
                'batch_count'] > 0 else 0.0,
            'kappa': metrics_accumulator['kappa'] / metrics_accumulator['batch_count'] if
            metrics_accumulator[
                'batch_count'] > 0 else 0.0
        }

        # Print metrics
        print(f"Train Metrics - Loss: {epoch_loss:.4f}, "
              f"Overall Acc: {avg_metrics['OA']:.4f}, "
              f"Mean Acc: {avg_metrics['mean_acc']:.4f}, "
              f"Mean IoU: {avg_metrics['mean_iou']:.4f}, "
              f"Kappa: {avg_metrics['kappa']:.4f}")



    def validate(self):

        self.model.eval()
        running_loss = 0.0
        count = 0

        # Initialize metrics accumulators
        metrics_accumulator = {
            'overall_acc': 0.0,
            'mean_acc': 0.0,
            'mean_iou': 0.0,
            'kappa': 0.0,
            'batch_count': 0
        }

        with torch.no_grad():
            for sample in tqdm(self.val_loader, desc="Validation"):

                hh = sample['hh'].to(self.device).unsqueeze(dim=1)
                hv = sample['hv'].to(self.device).unsqueeze(dim=1)
                hhhv = sample['hh_hv_ratio'].to(self.device).unsqueeze(dim=1)
                masks = sample['label'].to(self.device)
                longitude = sample['longitude'].to(self.device)
                latitude = sample['latitude'].to(self.device)

                images = torch.cat([hh, hv], dim=1)
                images = torch.cat([images, hhhv], dim=1)

                if not valid_mask(masks, self.ignore_index):
                    continue

                outputs = self.model(images, masks, longitude, latitude)[0]
                _, preds = torch.max(outputs, dim=1)

                # Calculate metrics
                batch_metrics = calculate_metrics(
                    preds, masks,
                    num_classes=self.num_classes,
                    ignore_index=self.ignore_index
                )

                # Accumulate metrics
                metrics_accumulator['overall_acc'] += batch_metrics["Overall Accuracy (valid classes)"]
                metrics_accumulator['mean_acc'] += batch_metrics["Mean Accuracy (valid classes)"]
                metrics_accumulator['mean_iou'] += batch_metrics["Mean IoU (valid classes)"]
                metrics_accumulator['kappa'] += batch_metrics["Cohen's Kappa"]
                metrics_accumulator['batch_count'] += 1

                # Compute loss
                loss = self.criterion(outputs, masks)

                if torch.isnan(loss) or loss.item() == 0.0:
                    continue

                running_loss += loss.item()
                count += 1

        # Calculate epoch averages
        epoch_loss = running_loss / count if count > 0 else float("nan")
        avg_metrics = {
            'loss': epoch_loss,
            'acc': metrics_accumulator['overall_acc'] / metrics_accumulator['batch_count'] if metrics_accumulator[
                                                                                                  'batch_count'] > 0 else 0.0,
            'mean_acc': metrics_accumulator['mean_acc'] / metrics_accumulator['batch_count'] if metrics_accumulator[
                                                                                                    'batch_count'] > 0 else 0.0,
            'mean_iou': metrics_accumulator['mean_iou'] / metrics_accumulator['batch_count'] if metrics_accumulator[
                                                                                                    'batch_count'] > 0 else 0.0,
            'kappa': metrics_accumulator['kappa'] / metrics_accumulator['batch_count'] if metrics_accumulator[
                                                                                              'batch_count'] > 0 else 0.0
        }

        # Print metrics
        print(f"Val Metrics - Loss: {epoch_loss:.4f}, "
              f"Acc: {avg_metrics['acc']:.4f}, "
              f"Mean Acc: {avg_metrics['mean_acc']:.4f}, "
              f"Mean IoU: {avg_metrics['mean_iou']:.4f}, "
              f"Kappa: {avg_metrics['kappa']:.4f}")

        return avg_metrics


    def test_(self):

        self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()
        # mae_model.load_state_dict(torch.load("mae_s2_pretrained.pth"))
        # --- Evaluate on test set with updated metrics ---
        print("\nEvaluating model on test set...")
        all_predictions = []
        all_targets = []
        self.model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for inference
            for sample in tqdm(self.test_loader, desc="Testing"):
                hh = sample['hh'].to(self.device).unsqueeze(dim=1)
                hv = sample['hv'].to(self.device).unsqueeze(dim=1)
                hhhv = sample['hh_hv_ratio'].to(self.device).unsqueeze(dim=1)
                masks = sample['label'].to(self.device)
                longitude = sample['longitude'].to(self.device)
                latitude = sample['latitude'].to(self.device)

                images = torch.cat([hh, hv], dim=1)
                images = torch.cat([images, hhhv], dim=1)

                outputs = self.model(images, masks, longitude, latitude)[0]
                predicted_classes = torch.argmax(outputs, dim=1)

                all_predictions.append(predicted_classes.cpu())
                all_targets.append(masks.cpu())

        # Concatenate all predictions and targets
        final_predictions = torch.cat(all_predictions, dim=0)
        final_targets = torch.cat(all_targets, dim=0)

        # Calculate and print metrics

        metrics_results = calculate_metrics(final_predictions, final_targets, num_classes=self.num_classes, ignore_index=self.ignore_index)

        print("\n📊 Final Evaluation Metrics:")
        print("=" * 50)
        print(f"{'Overall Accuracy:':<25} {metrics_results['Overall Accuracy (valid classes)']:.4f}")
        print(f"{'Mean Accuracy:':<25} {metrics_results['Mean Accuracy (valid classes)']:.4f}")
        print(f"{'Mean IoU:':<25} {metrics_results['Mean IoU (valid classes)']:.4f}")

        # Handle the Kappa metric with proper key access
        kappa_key = [k for k in metrics_results.keys() if 'kappa' in k.lower()][0]  # Find the correct key
        print(f"{'Kappa Score:':<25} {metrics_results[kappa_key]:.4f}")
        print("-" * 50)

        # Print per-class metrics with improved formatting
        print("\n📈 Per-Class Metrics:")
        print(f"{'Class':<10} {'IoU':<10} {'Accuracy':<10}")
        print("-" * 30)
        for cls_id in range(self.num_classes):
            iou_val = metrics_results['Per-Class IoU'].get(cls_id, 0.0)
            acc_val = metrics_results['Per-Class Accuracy'].get(cls_id, 0.0)
            print(f"{cls_id:<10} {iou_val:.4f}    {acc_val:.4f}")

        print("\n✅ Evaluation complete.")
        print("=" * 50)




class visualization_func:
    def __init__(self, uncertainty_estimator, model, model_save_path, fig_save_dir, test_dataset, plot_num, device, fig_name):

        self.uncertainty_estimator = uncertainty_estimator
        self.model = model
        self.model_save_path = model_save_path
        self.test_dataset = test_dataset
        self.plot_num = plot_num
        self.device = device
        self.fig_name = fig_name
        self.fig_save_dir = fig_save_dir

    def vis_func(self):

        """Visualize model predictions on test samples"""
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()

        color_map = {
            -1: [0, 0, 0],  # Black - ignore/no data
            0: [1, 0, 0],  # Red - first year ice
            1: [0, 0, 1],  # Blue - young ice
            2: [0, 1, 0],  # Green - water
            3: [1, 1, 0]  # Yellow - background
        }
        # Group samples by original_class


        # Select random samples for this class

        # Process each sample
        test_loader = DataLoader(
            self.test_dataset, batch_size=1, shuffle=False
        )
        num = len(test_loader)

        pbar = tqdm(test_loader, desc="Visualization")
        for j, sample in enumerate(pbar):
            fig, axes = plt.subplots(
                1, 5,
                figsize=(25, 5),
                squeeze=False  # Ensure axes is always 2D
            )
            hh = sample['hh'].to(self.device).unsqueeze(dim=1)
            hv = sample['hv'].to(self.device).unsqueeze(dim=1)
            hhhv = sample['hh_hv_ratio'].to(self.device).unsqueeze(dim=1)
            masks = sample['label'].to(self.device)
            longitude = sample['longitude'].to(self.device)
            latitude = sample['latitude'].to(self.device)

            images = torch.cat([hh, hv], dim=1)
            images = torch.cat([images, hhhv], dim=1)

            # Get prediction
            with torch.no_grad():
                local_map = self.model(images, masks, longitude, latitude)[0]
                predictive_mean, uncertainty, all_predictions = self.uncertainty_estimator.get_uncertainty(images, masks, longitude, latitude)
                local_pred_mask = local_map.argmax(dim=1).squeeze().cpu().numpy()
                uncertainty = uncertainty.squeeze().cpu().numpy()

            # For visualization - get first 3 channels
            image_numpy = images.squeeze(0).cpu().numpy()
            C, H, W = image_numpy.shape[0], image_numpy.shape[1], image_numpy.shape[2]
            #
            # Apply standard deviation stretching for better contrast

            #vis_image = stretch_image(vis_image, percent=2)

            # Normalize image for display
            # for c in range(vis_image.shape[-1]):
            #     channel = vis_image[..., c]
            #     vis_image[..., c] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
            # Plot input image

            axes[0, 0].imshow(hh.squeeze(0).squeeze(0).cpu().numpy())
            axes[0, 0].set_title('HH')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(hv.squeeze(0).squeeze(0).cpu().numpy())
            axes[0, 1].set_title('HV')
            axes[0, 1].axis('off')

            # Plot ground truth (shift classes by 1 for display)
            label_rgb = np.zeros((local_pred_mask.shape[0], local_pred_mask.shape[1], 3))
            for class_id, color in color_map.items():
                mask = (masks.squeeze(0).cpu().numpy() == class_id)
                for c in range(3):
                    label_rgb[..., c][mask] = color[c]

            axes[0, 2].imshow(label_rgb)
            axes[0, 2].set_title('Ground Truth')
            axes[0, 2].axis('off')

            # Plot prediction (shift classes by 1 for display)
            label_rgb = np.zeros((local_pred_mask.shape[0], local_pred_mask.shape[1], 3))
            for class_id, color in color_map.items():
                mask = (local_pred_mask == class_id)
                for c in range(3):
                    label_rgb[..., c][mask] = color[c]
            axes[0, 3].imshow(label_rgb)
            axes[0, 3].set_title('Local Map')
            axes[0, 3].axis('off')

            # colored_mask = np.zeros((*global_map_.shape, 3))
            # for class_val_, color_info in CLASS_DEFINITIONS.items():
            #     color = np.array(color_info['color']) / 255
            #     colored_mask[global_map_ == class_val_] = color

            axes[0, 4].imshow(uncertainty, cmap='PRGn')
            axes[0, 4].set_title('Uncertainty Map')
            axes[0, 4].axis('off')
            plt.tight_layout()
            plt.savefig(
                os.path.join(self.fig_save_dir, self.fig_name + str(j) + '.png'),
                bbox_inches='tight'
            )
            plt.close(fig)  # Critical: Close figure to free memory



