# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import pdb

import numpy as np
from functools import partial
from HR_Mamba import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from timm.models.vision_transformer import PatchEmbed, Block
from segmentation_models_pytorch.losses import FocalLoss
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

# --- Semantic Segmentation Decoder ---
# Helper Block for the Decoder

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean', ignore_index=-100, num_classes=None, device='cuda:0'):
        """
        Focal Loss for segmentation tasks

        Args:
            alpha: Weighting factor for class balancing (None, float, or list/tensor)
            gamma: Focusing parameter (higher gamma => more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
            ignore_index: Index to ignore in loss calculation
            num_classes: Number of classes for automatic alpha calculation
            device: Device for tensors
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.device = device

        # Handle alpha parameter
        if alpha is None:
            self.alpha = self.compute_class_weights()
            print("HERE")
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha] * num_classes, device=device)
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha, device=device)
        else:
            self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits tensor of shape [B, C, H, W] or [B, C, ...]
            targets: Ground truth tensor of shape [B, H, W] or [B, ...]
        """
        # Flatten tensors for easier processing

        B, C, H, W = inputs.shape
        spatial_loss = torch.zeros(B * H * W, device=inputs.device)
        inputs_flat = inputs.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [B*H*W, C]
        targets_flat = targets.view(-1)  # [B*H*W]

        # Handle ignore_index
        if self.ignore_index is not None:
            valid_mask = targets_flat != self.ignore_index
            inputs_flat = inputs_flat[valid_mask]
            targets_flat = targets_flat[valid_mask]

        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs_flat, targets_flat, reduction='none')

        # Get probabilities
        pt = torch.exp(-ce_loss)  # p_t = probability of true class

        # Apply focal weighting
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply class balancing if alpha is provided
        #if self.alpha is None:
            # Get alpha values for each target class
        alpha_t = self.alpha[targets_flat]
        focal_loss = alpha_t * focal_loss

        spatial_loss[valid_mask] = focal_loss
        # Reshape back to spatial dimensions
        spatial_loss = spatial_loss.view(B, H, W)  # shape: [B, H, W]

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return spatial_loss

    def compute_class_weights(self, targets=None):
        """
        Compute class weights based on inverse frequency for Focal Loss
        Using the pixel distribution data provided
        """
        # Pixel counts for each class (from your data)
        class_pixels = {
            0: 26725132,  # Temperate needleleaf forest
            1: 3479552,  # Sub-polar taiga forest
            2: 7151388,  # Temperate broadleaf forest
            3: 21473911,  # Mixed forest
            4: 17896294,  # Temperate shrubland
            5: 26564557,  # Temperate grassland
            6: 41038,  # Polar grassland-lichen
            7: 18559014,  # Wetland
            8: 19812106,  # Cropland
            9: 23706607,  # Barren lands
            10: 10075031,  # Urban
            11: 21486339,  # Water
            12: 5231321  # Snow/ice
        }

        class_pixels = {
            0: 100,
            1: 100,
            2: 100
        }

        # Convert to tensor and ensure proper ordering
        total_pixels = sum(class_pixels.values())
        class_counts = torch.zeros(self.num_classes, device=self.device)
        print(f"TEST: {self.num_classes}")

        for class_idx, count in class_pixels.items():
            if class_idx < self.num_classes:
                class_counts[class_idx] = count

        # Compute inverse frequency weights
        # Higher weight for rare classes, lower weight for frequent classes
        class_weights = total_pixels / (class_counts + 1e-6)

        # Normalize weights so they sum to num_classes (common practice)
        class_weights = torch.log(class_weights) / torch.log(class_weights).sum() * self.num_classes

        # Optional: Apply smoothing or clipping to avoid extreme weights
        class_weights = torch.clamp(class_weights, min=0.1, max=10.0)

        return class_weights

class LearnableSinusoidalEmbedding(nn.Module):
    def __init__(self, embedding_dim, scale=1.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.scale = scale  # Optional scaling factor

        # Learnable frequency and phase parameters
        self.freq = nn.Parameter(torch.exp(
            torch.arange(0, embedding_dim, 2, dtype=torch.float) *
            (-math.log(10000.0) / embedding_dim)
        ))
        self.phase = nn.Parameter(torch.zeros(embedding_dim // 2))

    def forward(self, x):
        # x: (B, H, W) normalized coordinates in [-1, 1]
        x = x.unsqueeze(-1)  # (B, H, W, 1)

        # Compute sinusoidal embeddings with learnable parameters
        angles = x * self.freq.unsqueeze(0).unsqueeze(0) + self.phase.unsqueeze(0).unsqueeze(0)
        embeddings = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        # Reshape to (B, H, W, embedding_dim)
        return embeddings * self.scale

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

class SparseDeformableMambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, sparsity_ratio=0.3):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = dim * expand
        self.sparsity_ratio = sparsity_ratio

        self.norm = RMSNorm(dim)
        self.proj_in = nn.Linear(dim, self.expanded_dim)
        self.proj_out = nn.Linear(self.expanded_dim, dim)

        self.A = nn.Parameter(torch.zeros(d_state, d_state))
        self.B = nn.Parameter(torch.zeros(1, 1, d_state))
        self.C = nn.Parameter(torch.zeros(self.expanded_dim, d_state))

        self.conv = nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.expanded_dim,
            bias=False
        )

    def _build_controllable_matrix(self, n):
        A = torch.zeros(n, n)
        for i in range(n - 1):
            A[i, i + 1] = 1.0
        A[-1, :] = torch.randn(n) * 0.02
        return A

    def forward(self, x):
        B, L, C = x.shape
        # L = H * W
        residual = x

        # Flatten spatial dimensions
        x_flat = x  # x.reshape(B, L, C)

        # Normalize and project
        x_norm = self.norm(x_flat)
        x_proj = self.proj_in(x_norm)  # [B, L, expanded_dim]

        # Token selection
        center_idx = L // 2
        center = x_proj[:, center_idx:center_idx + 1, :]

        x_proj_norm = F.normalize(x_proj, p=2, dim=-1)  # [B, L, D]
        center_norm = F.normalize(center, p=2, dim=-1)  # [B, 1, D]
        sim = torch.matmul(x_proj_norm, center_norm.transpose(-1, -2)).squeeze(-1)
        # im = torch.matmul(x_proj, center.transpose(-1, -2)).squeeze(-1)  # [B, L]
        sim = torch.softmax(sim, dim=-1)  # Normalized probabilities

        k = max(1, int(L * self.sparsity_ratio))
        _, topk_idx = torch.topk(sim, k=k, dim=-1)

        x_sparse = batched_index_select(x_proj, 1, topk_idx)  # [B, k, expanded_dim]

        # Conv processing
        x_conv = x_sparse.transpose(1, 2)
        x_conv = self.conv(x_conv)[..., :k]
        x_conv = x_conv.transpose(1, 2)

        # SSM processing
        h = torch.zeros(B, self.expanded_dim, self.d_state, device=x.device)
        outputs = []

        for t in range(k):
            x_t = x_conv[:, t].unsqueeze(-1)
            Bx = torch.sigmoid(self.B.to(x.device)) * x_t
            h = torch.matmul(h, self.A.to(x.device).T) + Bx
            out_t = (h * torch.sigmoid(self.C.to(x.device).unsqueeze(0))).sum(-1)
            outputs.append(out_t)

        x_processed = torch.stack(outputs, dim=1)
        x_processed = self.proj_out(x_processed)

        # Combine with residual
        # x_processed = x_processed + batched_index_select(residual.reshape(B, L, C), 1, topk_idx)

        # Scatter back to original positions
        output = torch.zeros(B, L, C, device=x.device)
        output.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, C), x_processed)

        # return output.reshape(B, H, W, C) + x

        return output + x

class SimplifiedMambaBlock(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = dim * expand

        #self.norm = DyT(dim)
        self.norm = RMSNorm(dim)
        self.proj_in = nn.Linear(dim, self.expanded_dim)
        self.proj_out = nn.Linear(self.expanded_dim, dim)

        # SSM parameters
        self.A = nn.Parameter(torch.zeros(self.expanded_dim, d_state))
        self.B = nn.Parameter(torch.zeros(self.expanded_dim, d_state))
        self.C = nn.Parameter(torch.zeros(self.expanded_dim, d_state))

        # Convolution layer
        self.conv = nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.expanded_dim,
            bias=False
        )

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.proj_in(x)

        # Conv branch
        x_conv = x.transpose(1, 2)
        x_conv = self.conv(x_conv)[..., :x.shape[1]]
        x_conv = x_conv.transpose(1, 2)

        # SSM branch
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.expanded_dim, self.d_state, device=x.device)
        outputs = []

        for t in range(seq_len):
            x_t = x_conv[:, t].unsqueeze(-1)
            Bx = torch.sigmoid(self.B) * x_t
            h = torch.sigmoid(self.A.unsqueeze(0)) * h + Bx
            out_t = (h * torch.sigmoid(self.C.unsqueeze(0))).sum(-1)
            outputs.append(out_t)

        x = torch.stack(outputs, dim=1)
        x = self.proj_out(x)
        return x + residual


class MambaConv2D(nn.Module):
    """Replacement for Conv2D using Mamba blocks with parallel processing"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels

        # Projection layers
        self.proj_in  = nn.Conv2d(in_channels,  out_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # Mamba block (now processes all positions in parallel)
        self.mamba = Mamba(
            d_model=out_channels, # Dimension of input features
            d_state=16,           # State dimension
            d_conv=4,             # Conv kernel size
            expand=2,             # Expansion factor
        )

    def forward(self, x):
        B, C, H, W = x.shape
        res = x
        # Apply initial projection
        x = self.proj_in(x)

        # Unfold patches (similar to convolution)
        x_unfold = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        # Reshape to (B, C, K*K, L) where L is H'*W'
        x_unfold = x_unfold.view(B, self.out_channels, self.kernel_size * self.kernel_size, -1)
        # Permute to (B, L, K*K, C)
        x_unfold = x_unfold.permute(0, 3, 1, 2)

        # Get dimensions
        L = x_unfold.shape[1]  # Number of positions (H' * W')
        K_sq = self.kernel_size * self.kernel_size

        # Reshape for parallel processing: combine batch and position dimensions
        # New shape: (B*L, K*K, C)
        x_reshaped = x_unfold.reshape(-1, K_sq, self.out_channels)

        # Process ALL positions in parallel with Mamba
        x_processed = self.mamba(x_reshaped)

        # Reshape back to (B, L, K*K, C)
        x_processed = x_processed.reshape(B, L, K_sq, self.out_channels)

        # Permute back to (B, C, K*K, L) for folding
        x_processed = x_processed.permute(0, 3, 2, 1)

        # Fold back to original spatial dimensions
        x_processed = x_processed.reshape(B, -1, L)  # B, C*K*K, L

        output = F.fold(x_processed, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)

        # Final projection
        output = self.proj_out(output)
        return output

class Canadian_mapping_model_func(nn.Module):
    def __init__(self, num_classes, dim, d_conv, in_channel, ema_decay=0.99):
        super().__init__()

        self.in_channel = in_channel
        self.num_classes = num_classes


        self.ss_ctype = 'ce'
        self.dim = dim
        self.d_conv = d_conv
        self.lon_embed = LearnableSinusoidalEmbedding(embedding_dim=10)
        self.lat_embed = LearnableSinusoidalEmbedding(embedding_dim=10)
        self.cnn_backbone = L2HNet(width=64, image_band=in_channel+10)
        self.decode = DecoderCup()

        self.global_mamba = nn.ModuleList([SparseDeformableMambaBlock(dim=1024, d_conv=self.d_conv) for i in range(2)])

        self.conv = nn.Sequential(
            nn.Conv2d(640, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.cls1 = nn.Conv2d(64+1, self.num_classes, kernel_size=3, padding=1)
        self.cls2 = nn.Conv2d(256, self.num_classes, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.3)
        # Projection Head: maps encoder features to contrastive space
        self.projection_head = nn.Sequential(
            nn.Conv2d(64+1, 256, 1),  # Adjust input dim to match backbone
            nn.ReLU(),
            nn.Conv2d(256, 128, 1)
        )
        self.centers1 = nn.Parameter(torch.zeros(num_classes, 128))
        self.centers2 = nn.Parameter(torch.zeros(num_classes, 128))
        # self.register_buffer('prototypes', torch.zeros(num_classes, 128))
        self.register_buffer('counts1', torch.zeros(num_classes))
        self.register_buffer('counts2', torch.zeros(num_classes))
        self.ema_decay = ema_decay
        self.num_classes = num_classes

        # self.fuse_GL = FeatureFusion(
        #     p=self.dim,
        #     spatial_dim=self.num_superpixel,
        #     spectral_dim=self.total_pixel,
        #     d_model=256,
        #     nhead=8,
        #     num_layers=2,
        #     dropout=0.1
        # )

        self.global_seg = nn.Linear(self.dim, self.num_classes)

        # self.Gl_seg = DipResNet2Layers(
        # num_input_channels=self.dim,
        # num_output_channels=self.num_classes,
        # num_channels=self.dim,
        # act_fun="LeakyReLU",
        # pad="reflection"
        #     )
        # #
        # self.Lg_seg = DipResNet2Layers(
        # num_input_channels=self.dim,
        # num_output_channels=self.num_classes,
        # num_channels=self.dim,
        # act_fun="LeakyReLU",
        # pad="reflection"
        #     )

    def forward(self, x, labels=None, long=None, lat=None):
        B, C, H, W = x.shape
        long_pe = self.lon_embed(long)
        lat_pe = self.lat_embed(lat)
        position_enc = long_pe + lat_pe
        position_enc = position_enc.permute(0, 3, 1, 2)
        x, features = self.cnn_backbone(torch.cat([x, position_enc], dim=1))
        B, C, H, W = x.shape
        x_reshape = x.reshape(B, C, -1).permute(0, 2, 1)
        # x = self.conv(x)
        # feat1, _ = self.stem1(x)
        # feat2, _ = self.stem2(feat1)
        # feat3, local_map = self.stem3(feat2)
        #
        # fusion_feat = torch.concatenate((feat1, feat2, feat3), dim=1)
        # features = self.fusion(fusion_feat)
        #num_channels = x.shape[1]# B, 256, 128, 128

        # batch_size, num_channels, h, w = x.shape
        for blk in self.global_mamba:
           x_reshape = blk(x_reshape)
        x = x_reshape.permute(0, 2, 1).reshape(B, C, H, W)
        # global_feat = self.global_mamba(x)
        # 展平特征和标签


        # features_flat = x.permute(0, 2, 3, 1).reshape(batch_size, -1, num_channels)  # [b, h*w, 256]
        # segments_flat = segments.reshape(batch_size, -1)  # [b, h*w]
        #
        # assert segments.max() <= target_num
        # padded_results = torch.zeros(batch_size, target_num, num_channels,
        #                              device=x.device, dtype=x.dtype)

        # for b in range(batch_size):
        #     # 获取当前batch的超像素ID和数量
        #     unique_ids = torch.unique(segments_flat[b])
        #     actual_num = len(unique_ids)
        #
        #     # 计算实际存在的超像素特征
        #     masks = (segments_flat[b].unsqueeze(1) == unique_ids.unsqueeze(0))  # [h*w, actual_num]
        #     sum_features = torch.matmul(masks.T.float(), features_flat[b])  # [actual_num, 256]
        #     pixel_counts = masks.sum(dim=0).float()  # [actual_num]
        #     mean_features = sum_features / (pixel_counts.unsqueeze(1) + 1e-7)  # [actual_num, 256]
        #     padded_results[b, :actual_num] = mean_features
        #     # # 填充到target_num
        #     # if actual_num >= target_num:
        #     #     padded_results[b] = mean_features[:target_num]  # 截断超出的部分
        #     # else:
        #     #     padded_results[b, :actual_num] = mean_features  # 不足部分保持为0
        # for blk in self.global_mamba:
        #     padded_results = blk(padded_results)

        # mamba_output = padded_results
        # batch_size, num_sp, feat_dim = mamba_output.shape
        # # device = mamba_output.device
        # mamba_output = F.softmax(mamba_output, dim=2)


        # 获取每个batch的实际超像素数量
        # actual_nums = [len(torch.unique(segments[b])) for b in range(batch_size)]

        # 创建扩展后的segments用于索引 [batch, 128, 128, 1]
        #segments_expanded = segments.unsqueeze(-1)  # [batch, 128, 128, 1]

        # 创建输出张量

        #
        #global_map = self.global_seg(mamba_output)

        # remap_output = torch.zeros(batch_size, feat_dim, *segments.shape[1:], device=device)
        #
        # for b in range(batch_size):
        #     # 获取有效特征 [actual_num, 256]
        #     valid_features = mamba_output[b, :actual_nums[b]]
        #
        #     # 创建查找表 [max_sp_id + 1, 256]
        #     max_id = segments[b].max().item()
        #     lookup_table = torch.zeros(max_id+1, feat_dim, device=device)
        #     unique_ids = torch.unique(segments[b])
        #     lookup_table[unique_ids] = valid_features[:len(unique_ids)]
        #
        #     # 向量化映射
        #     remap_output[b] = lookup_table[segments[b]].permute(2, 0, 1)  # [256, 128, 128]
        # #
        # # '''
        # #     G = torch.randn(b, config['total_pixel'], p)  # num super pixels
        # #     L = torch.randn(b, config['num_superpixel'], p) # total pixels
        # # '''
        # remap_output = remap_output * x + x
        x1, x2 = self.decode(x, features)
        xx = self.dropout(x1)
        local_map = self.cls1(xx)
        # 2. Project features for contrastive learning
        projections = self.projection_head(x1)  # shape [B, D, H, W]
        # L2 normalize along the channel dimension to place on unit hypersphere
        projections = F.normalize(projections, p=2, dim=1)  # shape [B, D, H, W]

        # 3. Update prototypes if we are training and have labels
        # if self.training and labels is not None:
        #     self.update_prototypes(projections, labels)
        # global_ = self.cls2(x2)
        return local_map, projections

    def update_classcenter(self, ss_masks_lbl, labels, projections):

        if self.training and labels is not None:
            self.update_prototypes1(projections[0], labels, ss_masks_lbl[0])
            self.update_prototypes2(projections[1], labels, ss_masks_lbl[1])

        return self.centers1, self.centers2

    def _get_single_pred_confidence_mask(self, pred_logits):
        # get confidence
        pred_probs = F.softmax(pred_logits.detach(), dim=1)
        pred_logprobs = pred_probs.log()
        unc_max = np.log(self.num_classes)
        # pred_logprobs = F.log_softmax(pred_logits.detach(), dim=1)
        # pred_probs = pred_logprobs.exp()
        if self.ss_ctype == 'ce':
            unc_all = -(pred_probs*pred_logprobs)
            unc_max = np.log(self.num_classes)
            # # # # # # DEBUG: saving nan value cases # # # # #
            # if torch.isnan(unc_all).sum()>0:
            #     print(f"EP{self.current_epoch}-i{batch_idx}-m{i}|Encounter nan in uncertainty. Number: all-{pred_logits.numel()}, logits-{torch.isnan(pred_logits).sum()}, uncertainty-{torch.isnan(unc_all).sum()}")
            #     if self.ls:
            #         if len(self.lsf)>1:
            #             fn = f'logits_ent_ls{int(self.lsf[0]*100)}_{int(self.lsf[1]*100)}_m{i}_{batch_idx}.h5'
            #         else:
            #             fn = f'logits_ent_ls{int(self.lsf[0]*100)}_m{i}_{batch_idx}.h5'
            #     else:
            #         fn = f'logits_ent_m{i}_{batch_idx}.h5'
            #     with h5py.File(fn, 'w') as f:
            #         f.create_dataset('logits', data=pred_logits.detach().cpu().numpy())
            #         f.create_dataset('pred_logprobs', data=pred_logprobs.cpu().numpy())
            #         f.create_dataset('unc_all', data=unc_all.cpu().numpy())
            unc = unc_all.sum(dim=1)
            conf = 1-unc/unc_max
        elif self.ss_ctype == 'gini':
            conf_all = (pred_probs**2)
            if torch.isnan(conf_all).sum()>0:
                tmp = pred_probs[torch.isnan(conf_all)]
                print(f"EP{self.current_epoch}|Encounter nan in confidence: min and max of prob in nan positions are {tmp.min()} - {tmp.max()}")
                conf_all[torch.isnan(conf_all)] = 0
            conf = conf_all.sum(dim=1)
        # remove nan values
        if torch.isnan(conf).sum()>0:
            # tmp = pred_logits*torch.isnan(conf)[:,None]
            # tmp = tmp[tmp!=0]
            #print(f"EP{self.current_epoch}|nan in final confidence: n-{torch.isnan(conf).sum()}, logits min-{tmp.min()}, logits max-{tmp.max()}")
            conf[torch.isnan(conf)] = 0
        # else:
        #       assert conf.min()>=0 and conf.max()<=1, f"Confidence value is out of range: {conf.min()} - {conf.max()}"
        return conf.clamp(min=0, max=1)

    def _get_single_label_confidence_mask(self, pred_logits, y):
        # Ensure y is on the same device as pred_logits
        y = y.to(pred_logits.device)

        # Create mask for ignore_index
        ignore_mask = (y == -1)

        # Replace out-of-bounds and negative values with 0 (temporary for one-hot)
        y_safe = y.clone()
        y_safe[y < 0] = 0
        y_safe[y >= self.num_classes] = 0

        # get confidence
        pred_probs = F.softmax(pred_logits.detach(), dim=1)

        # Convert y to one-hot encoding
        y_one_hot = F.one_hot(y_safe, num_classes=self.num_classes).float()
        y_one_hot = y_one_hot.permute((0, 3, 1, 2))

        # Calculate confidence
        conf = (pred_probs * y_one_hot).sum(dim=1)

        # Set confidence to 0 for ignored pixels and invalid classes
        invalid_mask = ignore_mask | (y < 0) | (y >= self.num_classes)
        conf = conf.masked_fill(invalid_mask, 0.0)

        # Handle NaN values
        if torch.isnan(conf).sum() > 0:
            conf = torch.nan_to_num(conf, nan=0.0)

        return conf

    def condidence_select(self, epoch, min=0.6):

        a_0 = 0.9  # 初始阈值
        lambda_ = 0.1  # 衰减系数

        a_t = np.maximum(a_0 * np.exp(-lambda_ * epoch), min)

        return a_t

    def _get_selection_mask_by_label_confidence(self, epoch, conf_mask, y, class_balanced=False):
        # get rumpup sample selection ratio
        # if self.current_epoch <= self.ss_nep:
        #     ss_ratio = 1 - self.ss_rmup_func(self.current_epoch, self.ss_nep) * (1 - self.ss_prop)
        # else:
        #     ss_ratio = self.ss_prop
        ss_ratio = self.condidence_select(epoch, min=0.6)
        # # # # # for debugging # # # # #
        # if DEBUG and self.current_epoch < 5:
        #     print(f"EP{self.current_epoch}|label-seg, ss_ratio={ss_ratio}")
        # # # # # for debugging # # # # #
        if ss_ratio > 0 and ss_ratio < 1:
            # sample selection
            # weights for the first proportion (ss_ratio) are set to 1,
            # while the weights for the rest uncertainty ones are set to (0-1) normalized by the maximum confidence in this part
            if not class_balanced:
                # select from all
                nss = int(y.numel() * ss_ratio)
                large_confs, _ = torch.topk(conf_mask, nss, largest=True)  # select nss smallest losses
                thrd_conf = large_confs[-1]
                if thrd_conf > 0:
                    mask = torch.clip(conf_mask / thrd_conf, min=0., max=1.)
                else:
                    mask = torch.ones_like(conf_mask, device=y.device, requires_grad=False)
            else:
                # select from each class
                mask = torch.zeros_like(conf_mask, device=y.device, requires_grad=False)
                for c in range(self.num_classes):
                    cind_mask = (y == c)
                    nc = cind_mask.sum()
                    nss = int(nc * ss_ratio)
                    if nss > 0:
                        c_conf = conf_mask[cind_mask]  # shape: (nc,)
                        c_large_conf, _ = torch.topk(c_conf, nss, largest=True)
                        c_thrd_conf = c_large_conf[-1]
                        if c_thrd_conf > 0:
                            c_conf_mask = conf_mask * cind_mask
                            cmask = torch.clip(c_conf_mask / c_thrd_conf, min=0., max=1.)
                            mask += cmask
                        else:
                            mask += cind_mask
            # if self.current_epoch<3:
            #     print(f"EP{self.current_epoch}|{mask.sum()} samples are selected for training!")
        elif ss_ratio == 1:
            mask = torch.ones_like(y, device=y.device, requires_grad=False)
        elif ss_ratio == 0:
            mask = conf_mask
        else:
            raise ValueError(f"Sample selection ratio is out of range: {ss_ratio}!")
        return mask

    def _get_selection_mask_by_pred_confidence(self, conf_mask):
        # get rumpup sample selection ratio
        if self.current_epoch <= self.ss_nep:
            ss_ratio = self.ss_rmup_func(self.current_epoch, self.ss_nep)
            mask = (1 - ss_ratio) * torch.ones_like(conf_mask, device=self.device,
                                                    requires_grad=False) + ss_ratio * conf_mask
        else:
            mask = conf_mask
            # # # # # for debugging # # # # #
        if DEBUG and self.current_epoch < 5:
            print(f"EP{self.current_epoch}|pred-consist, ss_ratio={ss_ratio}")
        # # # # # for debugging # # # # #
        return mask

    def _get_selection_mask_by_pred_confidence(self, conf_mask):
        # get rumpup sample selection ratio
        if self.current_epoch<=self.ss_nep:
            ss_ratio = self.ss_rmup_func(self.current_epoch, self.ss_nep)
            mask = (1-ss_ratio)*torch.ones_like(conf_mask, device=self.device, requires_grad=False)+ss_ratio*conf_mask
        else:
            mask = conf_mask
        # # # # # for debugging # # # # #
        if DEBUG and self.current_epoch<5:
            print(f"EP{self.current_epoch}|pred-consist, ss_ratio={ss_ratio}")
        # # # # # for debugging # # # # #
        return mask

    def _get_common_enhanced_confidence_masks(self, confidence_masks):
        cmask1, cmask2 = confidence_masks
        commask = cmask1*cmask2
        enhmask1 = (cmask1+commask)/2
        enhmask2 = (cmask2+commask)/2
        enhmasks = [enhmask1, enhmask2]
        return enhmasks

    def update_prototypes1(self, projections, labels, conf_mask):
        """
        Updated version that correctly filters out ignored pixels (label = -1).
        """
        projections_flat = projections.permute(0, 2, 3, 1).reshape(-1, projections.size(1))  # [N, D]
        labels_flat = labels.reshape(-1)  # [N]
        conf_mask_flat = conf_mask.reshape(-1)

        # --- CRITICAL FIX: Create a mask for valid (non-ignored) pixels ---
        # We only want to consider pixels whose label is NOT -1
        valid_pixel_mask = (labels_flat != -1)
        valid_projections = projections_flat[valid_pixel_mask]
        valid_labels = labels_flat[valid_pixel_mask]
        valid_conf_mask = conf_mask_flat[valid_pixel_mask]
        # -----------------------------------------------------------------

        # Now we only work with valid_projections and valid_labels
        # This ensures we never try to process a pixel with label=-1

        for class_id in range(self.num_classes):
            # Find valid pixels that belong to this specific class
            # We are now only searching through 'valid_labels', which contain only 0-12
            mask = (valid_labels == class_id)
            num_pixels = mask.sum().item()

            if num_pixels == 0:
                # It's normal for some classes to be missing from a small batch.
                # Skip the update for this class in this batch.
                continue

            class_embeddings = valid_projections[mask]
            batch_prototype = ((class_embeddings*valid_conf_mask[mask].unsqueeze(1))/valid_conf_mask[mask].sum()).mean(dim=0)

            # EMA update
            if self.counts1[class_id] == 0:
                self.centers1.data[class_id] = batch_prototype
            else:
                updated_prototype = (
                            self.ema_decay * self.centers1[class_id] + (1 - self.ema_decay) * batch_prototype)
                self.centers1.data[class_id] =  F.normalize(updated_prototype.unsqueeze(0), p=2, dim=1).squeeze(0)

            self.counts1[class_id] += 1
            # Check if the prototype has been initialized (is not all zeros)
            # updated_prototype = (self.ema_decay * self.prototypes[class_id] + (1 - self.ema_decay) * batch_prototype)
            # L2 normalize the updated prototype to keep it on the unit sphere
            # self.prototypes[class_id].data = F.normalize(updated_prototype.unsqueeze(0), p=2, dim=1).squeeze(0)

    def update_prototypes2(self, projections, labels, conf_mask):
        """
        Updated version that correctly filters out ignored pixels (label = -1).
        """
        projections_flat = projections.permute(0, 2, 3, 1).reshape(-1, projections.size(1))  # [N, D]
        labels_flat = labels.reshape(-1)  # [N]
        conf_mask_flat = conf_mask.reshape(-1)
        # --- CRITICAL FIX: Create a mask for valid (non-ignored) pixels ---
        # We only want to consider pixels whose label is NOT -1
        valid_pixel_mask = (labels_flat != -1)
        valid_projections = projections_flat[valid_pixel_mask]
        valid_labels = labels_flat[valid_pixel_mask]
        valid_conf_mask = conf_mask_flat[valid_pixel_mask]

        # -----------------------------------------------------------------

        # Now we only work with valid_projections and valid_labels
        # This ensures we never try to process a pixel with label=-1

        for class_id in range(self.num_classes):
            # Find valid pixels that belong to this specific class
            # We are now only searching through 'valid_labels', which contain only 0-12
            mask = (valid_labels == class_id)
            num_pixels = mask.sum().item()

            if num_pixels == 0:
                # It's normal for some classes to be missing from a small batch.
                # Skip the update for this class in this batch.
                continue

            class_embeddings = valid_projections[mask]
            batch_prototype = ((class_embeddings*valid_conf_mask[mask].unsqueeze(1))/valid_conf_mask[mask].sum()).mean(dim=0)

            # EMA update
            if self.counts2[class_id] == 0:
                self.centers2.data[class_id] = batch_prototype
            else:
                updated_prototype = (
                            self.ema_decay * self.centers2[class_id] + (1 - self.ema_decay) * batch_prototype)
                self.centers2.data[class_id] =  F.normalize(updated_prototype.unsqueeze(0), p=2, dim=1).squeeze(0)

            self.counts2[class_id] += 1


class PrototypicalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, projections, prototypes, labels):
        """
        Args:
            projections: L2-normalized feature maps [B, D, H, W]
            prototypes: L2-normalized prototype vectors [num_classes, D]
            labels: Ground truth map [B, H, W]
        Returns:
            loss: The contrastive loss scalar
        """
        b, d, h, w = projections.shape
        projections_flat = projections.permute(0, 2, 3, 1).reshape(-1, d)  # [N, D]
        labels_flat = labels.reshape(-1)  # [N]

        # Calculate similarity between all embeddings and all prototypes
        # Uses matrix multiplication because vectors are normalized -> sim = cosine similarity
        similarity_matrix = torch.matmul(projections_flat, prototypes.T)  # [N, num_classes]
        similarity_matrix /= self.temperature

        # The "positive" prototype for each pixel is its own class prototype
        positives = similarity_matrix[torch.arange(len(labels_flat)), labels_flat]  # [N]

        # The loss for each pixel is: -log(exp(positive) / sum(exp(all_similarities)))
        # This tries to make the pixel's similarity to its correct prototype much higher than to any other.
        loss = F.cross_entropy(similarity_matrix, labels_flat, ignore_index=-1)  # Use ignore_index for masked pixels

        return loss

class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class RPBlock(nn.Module):
    def __init__(self, input_chs, ratios=[1, 0.5, 0.25], bn_momentum=0.1):
        super(RPBlock, self).__init__()
        self.branches = nn.ModuleList()
        dilations = [1,2,4]
        paddings = [0,2,8]
        for i, ratio in enumerate(ratios):
            conv = nn.Sequential(
                nn.Conv2d(input_chs, int(input_chs * ratio), kernel_size=(2 * i + 1), stride=1, padding=i),
                nn.BatchNorm2d(int(input_chs * ratio), momentum=bn_momentum),
                nn.ReLU()
            )
            self.branches.append(conv)

        self.fuse_conv = nn.Sequential(  # + input_chs // 64
            nn.Conv2d(int(input_chs * sum(ratios)), input_chs, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(input_chs, momentum=bn_momentum),
            nn.ReLU()
        )

        self.edge = EdgeEnhancementModuleSobel(in_channels=input_chs)

    def forward(self, x):
        branches = torch.cat([branch(x) for branch in self.branches], dim=1)
        fused = self.fuse_conv(branches)
        edge_enhanced = self.edge(fused)
        output = edge_enhanced + x
        return output


class L2HNet(nn.Module):
    def __init__(self,
                 width,  # width=64 for light mode; width=128 for normal mode
                 image_band=4,
                 # image_band genenral is 3 (RGB) or 4 (RGB-NIR) for high-resolution remote sensing images
                 output_chs=128,
                 length=5,
                 ratios=[1, 0.5, 0.25],
                 bn_momentum=0.1):
        super(L2HNet, self).__init__()
        self.width = width
        # self.startconv = nn.Conv2d(image_band, self.width, kernel_size=3, stride=1, padding=1)

        self.startconv = nn.Sequential(
                StdConv2d(image_band, self.width, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.width),
                nn.ReLU()
            )

        self.rpblocks = nn.ModuleList()
        for _ in range(length):
            rpblock = RPBlock(self.width, ratios, bn_momentum)
            self.rpblocks.append(rpblock)

        # self.out_conv1 = nn.Sequential(
        #     StdConv2d(self.width * length, output_chs * length, kernel_size=3, stride=2, bias=False, padding=1, groups=32),
        #     nn.GroupNorm(32, output_chs * 5, eps=1e-6),
        #     nn.ReLU()
        # )
        # self.out_conv2 = nn.Sequential(
        #     StdConv2d(output_chs * length, 1024, kernel_size=3, stride=2, bias=False, padding=1, groups=32),
        #     nn.GroupNorm(32, 1024, eps=1e-6),
        #     nn.ReLU()
        # )
        # self.out_conv3 = nn.Sequential(
        #     StdConv2d(1024, 1024, kernel_size=5, stride=4, bias=False, padding=1, groups=32),
        #     nn.GroupNorm(32, 1024, eps=1e-6),
        #     nn.ReLU()
        # )

        # self.out_conv1 = nn.Sequential(
        #     nn.Conv2d(self.width * length, output_chs * length, kernel_size=3, stride=2,
        #               bias=False, padding=2, dilation=2, groups=32),  # 同时使用stride=2和dilation=2
        #     nn.GroupNorm(32, output_chs * length, eps=1e-6),
        #     nn.ReLU()
        # )
        #
        # self.out_conv2 = nn.Sequential(
        #     nn.Conv2d(output_chs * length, 1024, kernel_size=3, stride=2,
        #               bias=False, padding=4, dilation=4, groups=32),  # 同时使用stride=2和dilation=4
        #     nn.GroupNorm(32, 1024, eps=1e-6),
        #     nn.ReLU()
        # )
        #
        # self.out_conv3 = nn.Sequential(
        #     nn.Conv2d(1024, 1024, kernel_size=5, stride=4,
        #               bias=False, padding=8, dilation=4, groups=32),  # 同时使用stride=4和dilation=4
        #     nn.GroupNorm(32, 1024, eps=1e-6),
        #     nn.ReLU()
        # )

        # A more standard and robust design
        self.out_conv1 = nn.Sequential(
            # First, do a conservative downsampling
            # Use stride=1 first
            nn.Conv2d(self.width * length, output_chs * length, kernel_size=3, stride=1, bias=False, padding=1, groups=32),
            nn.GroupNorm(32, output_chs * length, eps=1e-6),
            nn.ReLU(),
            # Downsample using a dedicated layer (e.g., pooling or stride=2 conv)
            nn.Conv2d(output_chs * length, output_chs * length, kernel_size=3, stride=2, bias=False, padding=1, groups=32),
            nn.GroupNorm(32, output_chs * length, eps=1e-6),
            nn.ReLU()
        )

        self.out_conv2 = nn.Sequential(
            # Another conservative conv followed by downsampling
            # Increase channels gradually
            nn.Conv2d(output_chs * length, 1024, kernel_size=3, stride=1, bias=False, padding=1, groups=32),
            nn.GroupNorm(32, 1024, eps=1e-6),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, bias=False, padding=1, groups=32),
            nn.GroupNorm(32, 1024, eps=1e-6),
            nn.ReLU()
        )

        self.out_conv3 = nn.Sequential(
            # Layer 1: Dilated Conv for Context (stride=1)
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, bias=False, padding=4, dilation=4, groups=32),
            nn.GroupNorm(32, 1024, eps=1e-6),
            nn.ReLU(inplace=True),  # Add inplace=True to save memory

            # Layer 2: First 2x Downsample (e.g., with Strided Conv)
            # Downsample here
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, bias=False, padding=1, groups=32),  # dilation=1 is default
            nn.GroupNorm(32, 1024, eps=1e-6),
            nn.ReLU(inplace=True),

            # Layer 3: Second 2x Downsample
            # Final downsample
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, bias=False, padding=1, groups=32),
            nn.GroupNorm(32, 1024, eps=1e-6),
            nn.ReLU(inplace=True)
        )

        #
        # self.edge1 = EdgeEnhancementModuleSobel(in_channels=self.width * length)
        # self.edge2 = EdgeEnhancementModuleSobel(in_channels=1024)
        # self.edge3 = EdgeEnhancementModuleSobel(in_channels=1024)

    def forward(self, x):
        x = self.startconv(x)
        output_d1 = []
        for rpblk in self.rpblocks:
            x = rpblk(x)
            output_d1.append(x)
        output_d1 = self.out_conv1(torch.cat(output_d1, dim=1))
        output_d2 = self.out_conv2(output_d1)
        output_d3 = self.out_conv3(output_d2)
        features = [output_d1, output_d2, output_d3, x]
        return output_d3, features[::-1]

class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True,):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not(use_batchnorm),)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)

class DecoderCup(nn.Module):
    def __init__(self,):
        super().__init__()
        head_channels = 512
        self.conv_more = Conv2dReLU(1024, head_channels, kernel_size=3, padding=1, use_batchnorm=True,)
        self.conv1 = Conv2dReLU(1536,512, kernel_size=3, padding=1, use_batchnorm=True,)
        self.conv2 = Conv2dReLU(1536, 640, kernel_size=3, padding=1, use_batchnorm=True,)
        self.conv3 = Conv2dReLU(1280, 1, kernel_size=3, padding=1, use_batchnorm=True,)
        self.conv4 = Conv2dReLU(1280, 256, kernel_size=3, padding=1, use_batchnorm=True,)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, hidden_states, features=None):
        # B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        # x = hidden_states.permute(0, 2, 1)
        # x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(hidden_states)
        #X is the output of ViT branch
        x = torch.cat([x, features[1]], dim=1)
        x = self.conv1(x)
        x = self.up4(x)
        x = torch.cat([x, features[2]], dim=1)
        x = self.conv2(x)
        x = self.up2(x)
        x = torch.cat([x, features[3]], dim=1)
        x1 = self.conv3(x)
        x1 = self.up2(x1)
        x2 = self.conv4(x)
        x2 = self.up2(x2)
        x1 = torch.cat([x1, features[0]], dim=1) #Concat the 1-channal output_map of ViT branch to assist CNN branch training
        return x1,x2

class MonteCarloConsistency(nn.Module):
    def __init__(self, base_model, num_samples=2, temperature=3.0, num_classes=4):
        super().__init__()
        self.base_model = base_model
        self.num_samples = num_samples
        self.temperature = temperature
        self.losses_seg = {}
        # Cross Entropy
        self.CELoss = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.losses_seg['c'] = self.CELoss

        # self.FLoss = FocalLoss(mode='multiclass', reduction='none', ignore_index=-1)
        self.losses_seg['f'] = FocalLoss(reduction='none', ignore_index=-1, num_classes=num_classes)
        self.ContrastiveLoss = ClassCenterContrastiveLoss()

    def forward(self, x, labels, long, lat, epoch):
        self.enable_dropout()
        outputs = []
        projection = []
        for _ in range(self.num_samples):
            out_, proj_ = self.base_model(x, labels, long, lat)
            outputs.append(out_)
            projection.append(proj_)

        LOGITS = torch.stack(outputs, dim=0)
        PROJECTIONS = torch.stack(projection, dim=0)
        lbl_confs = [self.base_model._get_single_label_confidence_mask(LOGITS[i], labels) for i in range(2)]
        lbl_confs_mask = self.base_model._get_common_enhanced_confidence_masks(lbl_confs)
        ss_masks_lbl = [self.base_model._get_selection_mask_by_label_confidence(epoch, lbl_confs_mask[i], labels, class_balanced=True) for i in range(2)]
        centers1, centers2 = self.base_model.update_classcenter(ss_masks_lbl, labels, PROJECTIONS)
        return LOGITS, centers1, centers2

    def enable_dropout(self):
        for m in self.base_model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def compute_total_loss(self, epoch, mc_outputs, c1, c2, targets, img, loss_weights=None):
        """
        Compute complete loss: Cross Entropy + Consistency losses

        Args:
            mc_outputs: [num_samples, batch_size, num_classes]
            targets: [batch_size] ground truth labels
            loss_weights: Dictionary with weights for each loss component
        """
        if loss_weights is None:
            loss_weights = {
                'ce': 1.0,  # Cross entropy weight
                'mse': 0.3,  # MSE consistency weight
                'edge': 0.001,
                'js_kl': 0.0  # JS consistency weight (optional)
            }

        # 1. Cross Entropy Loss (main task loss)
        ce_loss = self._compute_cross_entropy_loss(epoch, mc_outputs, targets)

        # 2. Consistency Losses
        consistency_losses = self._compute_consistency_losses(mc_outputs, img)
        consistency_weight = self.get_current_consistency_weights(epoch)
        prototype_loss = self.ContrastiveLoss(c1, c2)
        # 3. Total weighted loss
        total_loss = (
                loss_weights['ce'] * ce_loss +
                loss_weights['mse'] * consistency_losses['mse'] +
                loss_weights['js_kl'] * consistency_losses['js_kl']
        )

        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'mse_consistency': consistency_losses['mse'],
            'js_consistency': consistency_losses['js_kl'],
            'edge': consistency_losses['edge'],
            'consistency_weight': consistency_weight,
            'prototype_loss': prototype_loss,
        }


    def get_current_consistency_weights(self, epoch):

        consistency_max = 10.0
        rampup_length = 10
        if epoch < rampup_length:
            phase = 1.0 - (epoch / rampup_length)
            return consistency_max * np.exp(-5.0 * phase * phase)
        else:
            return consistency_max

    def _compute_cross_entropy_loss(self, epoch, mc_outputs, targets):
        """Compute cross entropy loss - multiple strategies"""
        num_samples = mc_outputs.size(0)

        # Strategy 1: Average predictions then compute CE
        pred_logits = mc_outputs.mean(dim=0)

        #pred_confs = self.base_model._get_single_pred_confidence_mask(pred_logits)
        pred_confs = [self.base_model._get_single_pred_confidence_mask(mc_outputs[i]) for i in range(2)]

        # label confidence
        #lbl_confs = self.base_model._get_single_label_confidence_mask(pred_logits, targets)
        lbl_confs = [self.base_model._get_single_label_confidence_mask(mc_outputs[i], targets) for i in range(2)]
        # common information enhanced confidence

        #pred_confs = self.base_model._get_common_enhanced_confidence_masks(pred_confs)
        lbl_confs = self.base_model._get_common_enhanced_confidence_masks(lbl_confs)
        # sample selection masks
        # for label confidence (segmentation part): weights for the first proportion (ss_ratio) are set to 1,
        # while the weights for the rest uncertainty ones are set to (0-1) normalized by the maximum confidence in this part
        #ss_masks_lbl = self.base_model._get_selection_mask_by_label_confidence(epoch, lbl_confs, targets, class_balanced=True)
        ss_masks_lbl = [self.base_model._get_selection_mask_by_label_confidence(epoch, lbl_confs[i], targets, class_balanced=True) for i in
                        range(2)]
        # for prediction confidence (consistency part)
        # ss_masks_pred = [self._get_selection_mask_by_pred_confidence(pred_confs[i]) for i in range(2)]


        # calculate weighted losses
        # loss = torch.tensor(0.0, device=pred_logits.device)
        # # segmentation losses
        # loss_branch = torch.tensor(0.0, device=pred_confs.device)
        # for lk in self.losses_seg:
        #     L = self.losses_seg[lk]
        #     loss_ = L(pred_logits, targets)
        #     if lk == 'f':
        #         loss_ = (loss_*ss_masks_lbl).sum()/ss_masks_lbl.sum()
        #         loss_ = 0.1 * loss_
        #     else:
        #         loss_ = (loss_ * ss_masks_lbl).sum() / ss_masks_lbl.sum()
        #     # # # # # for debugging # # # # #
        #     # if DEBUG and self.current_epoch<5:
        #     #     print(f"EP{self.current_epoch}|loss_{lk}_{pi+1}: {loss_}")
        #     # # # # # for debugging # # # # #
        #     loss_branch += loss_
        # loss += loss_branch

        # calculate weighted losses
        loss = torch.tensor(0.0, device=mc_outputs.device)
        # segmentation losses
        for pi, pred_logits_ in enumerate(mc_outputs):
            # calculate losses
            loss_branch = torch.tensor(0.0, device=mc_outputs.device)
            for lk in self.losses_seg:
                L = self.losses_seg[lk]
                loss_ = L(pred_logits_, targets)
                loss_ = (loss_ * ss_masks_lbl[pi]).sum() / ss_masks_lbl[pi].sum()
                # # # # # for debugging # # # # #
                # if DEBUG and self.current_epoch<5:
                #     print(f"EP{self.current_epoch}|loss_{lk}_{pi+1}: {loss_}")
                # # # # # for debugging # # # # #
                loss_branch += loss_
            loss += loss_branch

        #ce_loss = F.cross_entropy(avg_predictions, targets, ignore_index=-1)

        # Strategy 2: Average CE across all samples
        # ce_losses = [F.cross_entropy(mc_outputs[i], targets) for i in range(num_samples)]
        # ce_loss = torch.stack(ce_losses).mean()

        # Strategy 3: Use first sample only (simplest)
        # ce_loss = F.cross_entropy(mc_outputs[0], targets)
        return loss

    def _compute_consistency_losses(self, mc_outputs, img):
        """Compute all consistency losses"""
        num_samples = mc_outputs.size(0)

        if num_samples < 2:
            return {'mse': 0.0, 'js_kl': 0.0}

        mse_loss, js_loss, edge_loss = 0.0, 0.0, 0.0
        count = 0

        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                mse_loss += F.mse_loss(mc_outputs[i], mc_outputs[j])
                # kl_loss += self._kl_divergence(mc_outputs[i], mc_outputs[j])
                js_loss += self._js_divergence(mc_outputs[i], mc_outputs[j])
                count += 1
        # edge_loss = self._edge_loss(mc_outputs.mean(dim=0), img)

        return {
            'mse': mse_loss / count,
            'js_kl': js_loss / count,
            'edge': edge_loss
        }

    def _kl_divergence(self, logits1, logits2):
        probs1 = F.log_softmax(logits1 / self.temperature, dim=1)
        probs2 = F.log_softmax(logits2 / self.temperature, dim=1)

        # print(f"Logits shape: {logits1.shape}")
        # print(f"Logits min: {logits1.min().item()}")
        # print(f"Logits max: {logits1.max().item()}")
        # print(f"Logits mean: {logits1.mean().item()}")
        # print(f"Logits std: {logits1.std().item()}")
        # print(f"Has NaN: {torch.isnan(logits1).any().item()}")
        # print(f"Has Inf: {torch.isinf(logits1).any().item()}")
        #
        #
        # print(f"Logits shape: {logits2.shape}")
        # print(f"Logits min: {logits2.min().item()}")
        # print(f"Logits max: {logits2.max().item()}")
        # print(f"Logits mean: {logits2.mean().item()}")
        # print(f"Logits std: {logits2.std().item()}")
        # print(f"Has NaN: {torch.isnan(logits2).any().item()}")
        # print(f"Has Inf: {torch.isinf(logits2).any().item()}")

        kl_pq = F.kl_div(probs1 + 1e-8, probs2, reduction='batchmean')
        kl_qp = F.kl_div(probs2 + 1e-8, probs1, reduction='batchmean')

        return (kl_pq + kl_qp) / 2

    def _js_divergence(self, logits1, logits2):
        probs1 = F.softmax(logits1 / self.temperature, dim=1)
        probs2 = F.softmax(logits2 / self.temperature, dim=1)
        M = 0.5 * (probs1 + probs2)

        # print(f"Logits shape: {probs1.shape}")
        # print(f"Logits min: {probs1.min().item()}")
        # print(f"Logits max: {probs1.max().item()}")
        # print(f"Logits mean: {probs1.mean().item()}")
        # print(f"Logits std: {probs1.std().item()}")
        # print(f"Has NaN: {torch.isnan(probs1).any().item()}")
        # print(f"Has Inf: {torch.isinf(probs1).any().item()}")
        #
        #
        # print(f"Logits shape: {probs2.shape}")
        # print(f"Logits min: {probs2.min().item()}")
        # print(f"Logits max: {probs2.max().item()}")
        # print(f"Logits mean: {probs2.mean().item()}")
        # print(f"Logits std: {probs2.std().item()}")
        # print(f"Has NaN: {torch.isnan(probs2).any().item()}")
        # print(f"Has Inf: {torch.isinf(probs2).any().item()}")

        kl1 = F.kl_div(torch.log(probs1 + 1e-8), M, reduction='batchmean')
        kl2 = F.kl_div(torch.log(probs2 + 1e-8), M, reduction='batchmean')

        return 0.5 * (kl1 + kl2)

    def _edge_loss(self, logist, net_input):

        edge_width = 1

        # dh0 = torch.min( torch.pow(net_input[:,:,:,edge_width:] - net_input[:,:,:,:-edge_width], 2), dim=1)[1]
        # dw0 = torch.min( torch.pow(net_input[:,:,edge_width:,:] - net_input[:,:,:-edge_width,:], 2), dim=1)[1]

        dh0 = torch.sum(torch.pow(net_input[:, :, :, edge_width:] - net_input[:, :, :, :-edge_width], 2), dim=1)
        dw0 = torch.sum(torch.pow(net_input[:, :, edge_width:, :] - net_input[:, :, :-edge_width, :], 2), dim=1)
        input_diff = (dh0[:, :-edge_width, :] + dw0[:, :, :-edge_width]) / torch.sum(
            dh0[:, :-edge_width, :] + dw0[:, :, :-edge_width])  # sum for all classes
        # input_diff = torch.unsqueeze(input_diff, dim=1)
        B, H, W = input_diff.shape
        input_diff = F.softmax(input_diff.reshape(1, B * H * W), dim=1)

        S = logist

        dh = torch.sum(torch.pow(S[:, :, :, edge_width:] - S[:, :, :, :-edge_width], 2), dim=1)
        dw = torch.sum(torch.pow(S[:, :, edge_width:, :] - S[:, :, :-edge_width, :], 2), dim=1)
        sparse_loss = torch.sum(dh[:, :-edge_width, :] + dw[:, :, :-edge_width])
        S_diff = ((dh[:, :-edge_width] + dw[:, :, :-edge_width]) / sparse_loss)  # sum for all classes
        # S_diff = torch.unsqueeze(S_diff, dim=1)
        S_diff = F.softmax(S_diff.reshape(1, B * H * W), dim=1)

        mse_div_loss = F.kl_div(S_diff.log(), input_diff, reduction='sum') + F.kl_div(input_diff.log(), S_diff, reduction='sum')

        return mse_div_loss / 2


class UncertaintyEstimator:
    def __init__(self, model, num_samples=50):
        self.model = model
        self.num_samples = num_samples

    def enable_dropout(self):
        """Set dropout layers to training mode to keep them active"""
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def get_uncertainty(self, x, label, long, lat):
        """
        Correct uncertainty calculation for segmentation tasks
        Shape: predictions [num_samples, batch_size, num_classes, height, width]
        """
        self.enable_dropout()

        # Collect multiple predictions
        predictions = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                pred = self.model(x, label, long, lat)[0]
                predictions.append(pred)

        predictions = torch.stack(predictions)  # [num_samples, batch_size, num_classes, height, width]

        # Calculate mean prediction
        predictive_mean = predictions.mean(dim=0)  # [batch_size, num_classes, height, width]

        # Calculate uncertainty - CORRECT DIMENSION FOR SEGMENTATION
        if predictions.size(2) == 1:  # Regression (unlikely for segmentation)
            predictive_variance = predictions.var(dim=0)
            uncertainty = predictive_variance.sqrt()
        else:  # Classification - USE dim=2 for class dimension!
            # Method 1: Variance of probabilities across samples
            probabilities = F.softmax(predictions, dim=2)  # dim=2 is class dimension!
            predictive_variance = probabilities.var(dim=0)  # variance across samples
            uncertainty_variance = predictive_variance.mean(dim=1)  # average over classes

            # Method 2: Entropy of mean prediction (more common)
            mean_probs = F.softmax(predictive_mean, dim=1)  # dim=1 is class dimension
            uncertainty_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)

            # You can use either method, or combine them
            uncertainty = uncertainty_entropy  # Typically use entropy

        return predictive_mean, uncertainty, predictions


class UnsupervisedPixelContrastLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2): # z1, z2 are projections from two augmented views
        b, d, h, w = z1.shape
        num_pixels = b * h * w

        # Reshape to [N, D]
        z1 = z1.permute(0, 2, 3, 1).reshape(-1, d)
        z2 = z2.permute(0, 2, 3, 1).reshape(-1, d)

        # Normalize
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        # Compute similarity matrix between all pixels in both views
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature # [N, N]

        # The positives are the diagonal of this large matrix
        labels = torch.arange(num_pixels, device=z1.device) # Positives are at index [i, i]

        loss = F.cross_entropy(sim_matrix, labels)
        return loss


class EdgeEnhancementModule(nn.Module):
    """
    Edge Enhancement Module for preserving fine spatial details in high-resolution land cover mapping.
    This module explicitly models boundary information using spatial gradients.
    """

    def __init__(self, in_channels, gamma_init=0.1):
        """
        Args:
            in_channels: Number of input channels
            gamma_init: Initial value for the learnable parameter gamma
        """
        super(EdgeEnhancementModule, self).__init__()

        # Learnable parameter controlling the strength of edge enhancement
        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))

        # 3x3 convolution for processing the gradient magnitudes
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                              padding=1, bias=False)

        # Initialize convolution weights (optional: identity initialization)
        nn.init.xavier_uniform_(self.conv.weight)

    def compute_spatial_gradients(self, x):
        """
        Compute spatial gradients in horizontal and vertical directions.

        Args:
            x: Input feature map of shape (batch_size, channels, height, width)

        Returns:
            Tuple of (grad_x, grad_y) gradient tensors
        """
        # Using central difference for gradient computation
        # For horizontal gradient (x-direction)
        grad_x = x[:, :, :, 2:] - x[:, :, :, :-2]  # Central difference in x
        # Pad to maintain original spatial dimensions
        grad_x = F.pad(grad_x, (1, 1, 0, 0), mode='replicate')

        # For vertical gradient (y-direction)
        grad_y = x[:, :, 2:, :] - x[:, :, :-2, :]  # Central difference in y
        # Pad to maintain original spatial dimensions
        grad_y = F.pad(grad_y, (0, 0, 1, 1), mode='replicate')

        return grad_x, grad_y

    def forward(self, F_fused):
        """
        Forward pass of the edge enhancement module.

        Args:
            F_fused: Fused feature map of shape (batch_size, channels, height, width)

        Returns:
            Revised feature map with enhanced edges
        """
        # Compute spatial gradients
        grad_x, grad_y = self.compute_spatial_gradients(F_fused)

        # Compute absolute gradients and sum them
        gradient_magnitude = torch.abs(grad_x) + torch.abs(grad_y)

        # Apply 3x3 convolution (Equation 5)
        E = self.conv(gradient_magnitude)

        # Apply edge enhancement (Equation 6)
        F_revised = F_fused + self.gamma * E

        return F_revised


# Alternative implementation using Sobel filters for gradient computation
class EdgeEnhancementModuleSobel(nn.Module):
    """
    Alternative implementation using Sobel filters for more robust gradient computation.
    """

    def __init__(self, in_channels, gamma_init=0.1):
        super(EdgeEnhancementModuleSobel, self).__init__()

        self.gamma = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                              padding=1, bias=False)

        # Precompute Sobel kernels
        self.sobel_x = nn.Parameter(torch.tensor([[[[1, 0, -1],
                                                    [2, 0, -2],
                                                    [1, 0, -1]]]], dtype=torch.float32).repeat(in_channels, 1, 1, 1),
                                    requires_grad=False)

        self.sobel_y = nn.Parameter(torch.tensor([[[[1, 2, 1],
                                                    [0, 0, 0],
                                                    [-1, -2, -1]]]], dtype=torch.float32).repeat(in_channels, 1, 1, 1),
                                    requires_grad=False)

        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, F_fused):
        # Compute gradients using Sobel filters
        grad_x = F.conv2d(F_fused, self.sobel_x, padding=1, groups=F_fused.size(1))
        grad_y = F.conv2d(F_fused, self.sobel_y, padding=1, groups=F_fused.size(1))

        # Compute absolute gradients and sum them
        gradient_magnitude = torch.abs(grad_x) + torch.abs(grad_y)

        # Apply 3x3 convolution
        E = self.conv(gradient_magnitude)

        # Apply edge enhancement
        F_revised = F_fused + self.gamma * E

        return F_revised

class ClassCenterContrastiveLoss(nn.Module):
    """
    Contrastive loss for class centers generated from two MC Dropout samples.
    This loss operates on the entire set of class centers.
    """

    def __init__(self, temperature=0.07, reduction='mean'):
        """
        Args:
            temperature (float): Scaling factor for the contrastive loss.
            reduction (str): 'mean' or 'sum' for loss reduction.
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, centers_1, centers_2):
        """
        Args:
            centers_1 (torch.Tensor): Class centers from first MC pass.
                                      Shape: [num_classes, feat_dim]
            centers_2 (torch.Tensor): Class centers from second MC pass.
                                      Shape: [num_classes, feat_dim]
        Returns:
            torch.Tensor: The contrastive loss.
        """
        num_classes, feat_dim = centers_1.shape
        device = centers_1.device

        # Normalize the class centers to unit hypersphere
        # centers_1 = F.normalize(centers_1, dim=1)  # [num_classes, feat_dim]
        # centers_2 = F.normalize(centers_2, dim=1)  # [num_classes, feat_dim]

        # Compute similarity matrix between all centers
        # This gives [num_classes, num_classes] matrix
        similarity_matrix = torch.matmul(centers_1, centers_2.T) / self.temperature

        # The positive pairs are the diagonal elements: center_i from pass1 vs center_i from pass2
        positives = similarity_matrix.diagonal()  # [num_classes]

        # The loss for each class: -log(exp(positive) / sum(exp(all similarities)))
        # We need to mask out the positive itself for the denominator
        # Create mask where diagonal is False (to exclude positive during negative summation)
        mask = ~torch.eye(num_classes, dtype=torch.bool, device=device)

        # For each class i, the negatives are all centers j where j != i
        loss_per_class = -positives + torch.logsumexp(similarity_matrix, dim=1)

        # Alternative more numerically stable implementation:
        # exp_logits = torch.exp(similarity_matrix)
        # log_sum_exp = torch.log(exp_logits.sum(dim=1, keepdim=True))
        # loss_per_class = -positives + log_sum_exp.squeeze()

        if self.reduction == 'mean':
            loss = loss_per_class.mean()
        elif self.reduction == 'sum':
            loss = loss_per_class.sum()
        else:
            loss = loss_per_class

        return loss

