import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from Mamba2D import SS2D
from mamba_ssm import Mamba
import math
device = torch.device("cuda" if 1 else "cpu")

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Root Mean Square Layer Normalization (RMSNorm).

        RMSNorm normalizes the input tensor along the last dimension using
        the root mean square (RMS) of the elements instead of the variance
        (as used in standard LayerNorm). This normalization technique is
        computationally more efficient and has been used in various transformer-based models.

        Formula:
            RMSNorm(x) = gamma * x / (RMS(x) + eps)
            where RMS(x) = sqrt(mean(x ** 2))

        Args:
            dim (int): The number of features in the input (i.e., size of the last dimension).
            eps (float): A small constant added to the denominator for numerical stability. Default: 1e-6.

        Attributes:
            weight (nn.Parameter): Learnable scaling parameter of shape (dim,).

        Shape:
            - Input: (N, ..., dim)
            - Output: (N, ..., dim) — same shape as input
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True)  # Compute L2 norm of each feature vector
        rms_x = norm_x * (x.shape[-1] ** -0.5)  # Convert L2 norm to RMS value
        return self.gamma * (x / (rms_x + self.eps))

class SimplifiedMambaBlock(nn.Module):
    """
    Simplified Mamba Block for Sequence Modeling.

    This block is inspired by the Mamba architecture and combines
    convolutional processing with a simplified state-space model (SSM)
    to capture both local and long-range dependencies in sequences.

    The architecture includes:
        - RMSNorm normalization on the input
        - Linear projection to an expanded feature space
        - A depthwise 1D convolutional branch to capture local context
        - A state-space recurrence branch to model long-term dependencies
        - A final linear projection and residual connection

    State-Space Model:
        The recurrence is governed by learnable parameters A, B, and C:
            h[t] = sigmoid(A) * h[t-1] + sigmoid(B) * x[t]
            y[t] = sum(sigmoid(C) * h[t])
        where h is the hidden state and x[t] is the input at timestep t.

    Args:
        dim (int): The input and output feature dimension.
        d_state (int): The number of internal SSM states per feature. Default: 16.
        d_conv (int): The kernel size of the depthwise convolution. Default: 4.
        expand (int): Factor to expand the feature dimension internally. Default: 2.

    Attributes:
        norm (RMSNorm): RMS-based normalization layer.
        proj_in (nn.Linear): Linear layer projecting input to expanded dimension.
        proj_out (nn.Linear): Linear layer projecting back to input dimension.
        A (nn.Parameter): Learnable state-transition weights (used in SSM).
        B (nn.Parameter): Learnable input weights (used in SSM).
        C (nn.Parameter): Learnable output weights (used in SSM).
        conv (nn.Conv1d): Depthwise convolutional layer.

    Input Shape:
        (batch_size, sequence_length, dim)

    Output Shape:
        (batch_size, sequence_length, dim)
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.expand = expand
        self.expanded_dim = dim * expand
        self.norm = RMSNorm(dim)
        self.proj_in = nn.Linear(dim, self.expanded_dim)
        self.proj_out = nn.Linear(self.expanded_dim, dim)
        # SSM parameters
        self.A = nn.Parameter(torch.zeros(self.expanded_dim, d_state))  # Shape: (expanded_dim, d_state)
        self.B = nn.Parameter(torch.zeros(self.expanded_dim, d_state))  # Shape: (expanded_dim, d_state)
        self.C = nn.Parameter(torch.zeros(self.expanded_dim, d_state))  # Shape: (expanded_dim, d_state)
        # Convolution layer
        self.conv = nn.Conv1d(
            in_channels=self.expanded_dim,
            out_channels=self.expanded_dim,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.expanded_dim,  # # Shape: (expanded_dim, d_state)
            bias=False
        )

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.proj_in(x)  # Exapnd the input feature
        # Conv branch --> depthwise 1D convolution over the sequence
        x_conv = x.transpose(1, 2)  # (B, D, T)
        x_conv = self.conv(x_conv)[..., :x.shape[1]]  # Trims output to match input sequence length.
        x_conv = x_conv.transpose(1, 2)  # (B, T, D)
        # SSM branch
        batch_size, seq_len, _ = x.shape
        # Initializes hidden state
        h = torch.zeros(batch_size, self.expanded_dim, self.d_state, device=x.device)
        outputs = []
        for t in range(seq_len):
            x_t = x_conv[:, t].unsqueeze(-1)  # Get input at time step t
            Bx = torch.sigmoid(self.B) * x_t
            h = torch.sigmoid(self.A.unsqueeze(0)) * h + Bx  # Updates hidden state
            out_t = (h * torch.sigmoid(self.C.unsqueeze(0))).sum(
                -1)  # Computes output using parameter C OR Weighted sum of hidden state
            outputs.append(out_t)
        x = torch.stack(outputs, dim=1)  # Reassemble the sequence
        x = self.proj_out(x)  # back to original dim
        return x + residual  # Add the residual connection

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]
        self.aspp1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.aspp2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=dilations[1], dilation=dilations[1], bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.aspp3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=dilations[2], dilation=dilations[2], bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.aspp4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=dilations[3], dilation=dilations[3], bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(out_channels * 5, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1, x2, x3, x4 = self.aspp1(x), self.aspp2(x), self.aspp3(x), self.aspp4(x)
        x5 = F.interpolate(self.global_avg_pool(x), size=x4.size()[2:], mode='bilinear', align_corners=True)
        return self.dropout(self.conv1(torch.cat((x1, x2, x3, x4, x5), dim=1)))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn1 = LayerNorm(planes, data_format="channels_first")
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.GELU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn2 = LayerNorm(planes, data_format="channels_first")
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion)
        self.relu = nn.GELU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.GELU()

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion),
                #LayerNorm(num_channels[branch_index] * block.expansion, data_format="channels_first")
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i]),
                        #LayerNorm(num_inchannels[i], data_format="channels_first"),
                        nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                #PatchMerging(num_inchannels[j], out_dim=num_outchannels_conv3x3),
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                #LayerNorm(num_outchannels_conv3x3, data_format="channels_first")))
                                nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                #LayerNorm(num_outchannels_conv3x3, data_format="channels_first"),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.GELU()))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

class HighResolutionNet(nn.Module):
    def __init__(self, config, **kwargs):
        super(HighResolutionNet, self).__init__()

        # stem net
        #self.conv1 = PatchMerging(10, out_dim=64)
        self.conv1 = nn.Conv2d(10, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn1 = LayerNorm(64, data_format="channels_first")
        self.bn1 = nn.BatchNorm2d(64)
        #self.conv2 = PatchMerging(64, out_dim=64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn2 = LayerNorm(64, data_format="channels_first")
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.GELU()

        self.stage1_cfg = config['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = Bottleneck if self.stage1_cfg['BLOCK'] == 'BOTTLENECK' else BasicBlock
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = config['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = BasicBlock
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = config['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = BasicBlock
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = config['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = BasicBlock
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels)


        # Decoder Part
        self.high_mamba = SimplifiedMambaBlock(
            dim=4*4,  # Model dimension d_model
        )


        self.aspp = ASPP(in_channels=256, out_channels=256)

        # --- Decoder with multiple skip connections ---
        # 1x1 conv for each feature stream to unify their channel dimensions
        self.project_low = nn.Conv2d(32, 128, kernel_size=1, bias=False)
        self.project_mid = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.project_mid2 = nn.Conv2d(128, 128, kernel_size=1, bias=False)  # ASPP output is 256
        self.project_high = nn.Conv2d(256, 128, kernel_size=1, bias=False) # ASPP output is 256

        # Final fusion and classifier
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(128 * 3, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            #LayerNorm(256, data_format="channels_first"),
            nn.GELU(),
        )

        self.fusion_conv1 = nn.Sequential(
            nn.Conv2d(128 * 2, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            #LayerNorm(256, data_format="channels_first"),
            nn.GELU(),
        )
        self.fusion_conv2 = nn.Sequential(
            nn.Conv2d(128 * 2, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            #LayerNorm(256, data_format="channels_first"),
            nn.GELU(),
        )
        self.fusion_conv3 = nn.Sequential(
            nn.Conv2d(128 * 2, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            #LayerNorm(256, data_format="channels_first"),
            nn.GELU(),
        )
        self.classifier = nn.Conv2d(256, config['NUM_CLASSES'], kernel_size=1)
        self.recon = nn.Conv2d(256, 10, kernel_size=1)
        # Classification Head
        self.incre_modules, self.downsamp_modules, \
            self.final_layer = self._make_head(pre_stage_channels)

        # self.classifier = nn.Linear(2048, config['NUM_CLASSES'])

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
                #LayerNorm(planes * block.expansion, data_format="channels_first"),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        SS2D(d_model=num_channels_pre_layer[i], d_state=16, d_conv=3, device=device),
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  1,
                                  bias=False),
                        # nn.Conv2d(num_channels_pre_layer[i],
                        #           num_channels_cur_layer[i],
                        #           3,
                        #           1,
                        #           1,
                        #           bias=False),
                        #LayerNorm(num_channels_cur_layer[i], data_format="channels_first"),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.GELU()))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(

                        # nn.Conv2d(inchannels,
                        #           outchannels,
                        #           1,
                        #           bias=False),
                        #PatchMerging(inchannels, out_dim=outchannels),
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        SS2D(d_model=outchannels, d_state=16, d_conv=3, device=device),
                        #LayerNorm(outchannels, data_format="channels_first"),
                        nn.BatchNorm2d(outchannels),
                        nn.GELU()))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = BasicBlock
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                      block,
                                      num_blocks,
                                      num_inchannels,
                                      num_channels,
                                      fuse_method,
                                      reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,
                                            channels,
                                            head_channels[i],
                                            1,
                                            stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels)-1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i+1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                #LayerNorm(out_channels, data_format="channels_first"),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            #LayerNorm(2048, data_format="channels_first"),
            nn.BatchNorm2d(2048),
            nn.GELU()
        )

        return incre_modules, downsamp_modules, final_layer

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)


        #hr_feature = y_list[0]  # [batch, 32, H/4, W/4]
        high_level_feat = y_list[3]
        #mid2_level_feat = y_list[2]
        mid_level_feat = y_list[1]
        low_level_feat = y_list[0]
        bs, c, h, w = high_level_feat.shape
        #high_level_feat = self.high_mamba(high_level_feat.reshape(bs,c,-1))
        #mid_level_feat = self.mid_mamba(mid_level_feat.reshape(mid_level_feat.shape[0],mid_level_feat.shape[1],-1).permute(0,2,1))
        #high_level_feat = high_level_feat.reshape(bs, c, h, w)
        # OCR Decoder

        aspp_feat = self.aspp(high_level_feat)
        low_projected = self.project_low(low_level_feat)
        mid_projected = self.project_mid(mid_level_feat)
        #mid2_projected = self.project_mid2(mid2_level_feat)
        high_projected = self.project_high(aspp_feat)


        mid_upsampled = F.interpolate(mid_projected, size=low_projected.shape[-2:], mode='bilinear', align_corners=False)
        high_upsampled = F.interpolate(high_projected, size=low_projected.shape[-2:], mode='bilinear', align_corners=False)
        fused = torch.cat([low_projected, mid_upsampled, high_upsampled], dim=1)

        # 4. Refine the fused features
        fused = self.fusion_conv(fused)

        # 5. Final upsample to original input size and classify
        #fused = F.interpolate(fused, size=input_size, mode='bilinear', align_corners=False)
        output = self.classifier(fused)

        return output

class PatchMerging(nn.Module):
    """
    Patch Merging Layer for downsampling (B, C, H, W) inputs.
    Args:
        dim (int): Input feature dimension (C).
        reduction_ratio (int): Patch merging ratio (default=2 for 2x2 merging).
        out_dim (int, optional): Output dimension. If None, set to 2*dim.
    """

    def __init__(self, dim, reduction_ratio=2, out_dim=None):
        super().__init__()
        self.reduction_ratio = reduction_ratio
        self.out_dim = out_dim if out_dim is not None else dim * (reduction_ratio ** 2)

        # Linear projection to adjust channels
        self.proj = nn.Linear(dim * (reduction_ratio ** 2), self.out_dim)

        # Optional normalization (e.g., LayerNorm as in Swin)
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, x):
        """
        Input:
            x: (B, C, H, W)
        Output:
            (B, out_dim, H//r, W//r), where r=reduction_ratio
        """
        B, C, H, W = x.shape
        r = self.reduction_ratio

        # Check if H and W are divisible by r
        assert H % r == 0 and W % r == 0, f"Input size {(H, W)} must be divisible by {r}"

        # Step 1: Reshape to extract patches
        x = x.view(B, C, H // r, r, W // r, r)  # (B, C, H/r, r, W/r, r)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, H/r, W/r, r, r, C)
        x = x.reshape(B, H // r, W // r, -1)  # (B, H/r, W/r, r*r*C)

        # Step 2: Linear projection + normalization
        x = self.proj(x)  # (B, H/r, W/r, out_dim)
        x = self.norm(x)

        # Step 3: Permute back to (B, C, H, W) format
        x = x.permute(0, 3, 1, 2)  # (B, out_dim, H/r, W/r)

        return x

def get_hrnet_config():
    config = {
        'STAGE1': {
            'NUM_MODULES': 1,
            'NUM_BRANCHES': 1,
            'NUM_BLOCKS': [4],
            'NUM_CHANNELS': [64],
            'BLOCK': 'BOTTLENECK',
            'FUSE_METHOD': 'SUM'
        },
        'STAGE2': {
            'NUM_MODULES': 1,
            'NUM_BRANCHES': 2,
            'NUM_BLOCKS': [4, 4],
            'NUM_CHANNELS': [32, 64],
            'FUSE_METHOD': 'SUM'
        },
        'STAGE3': {
            'NUM_MODULES': 4,
            'NUM_BRANCHES': 3,
            'NUM_BLOCKS': [4, 4, 4],
            'NUM_CHANNELS': [32, 64, 128],
            'FUSE_METHOD': 'SUM'
        },
        'STAGE4': {
            'NUM_MODULES': 3,
            'NUM_BRANCHES': 4,
            'NUM_BLOCKS': [4, 4, 4, 4],
            'NUM_CHANNELS': [32, 64, 128, 256],
            'FUSE_METHOD': 'SUM'
        },
        'NUM_CLASSES': 20  # 根据你的任务修改
    }
    return config


def hrnet(num_classes=1000, **kwargs):
    config = get_hrnet_config()
    config['NUM_CLASSES'] = num_classes
    model = HighResolutionNet(config, **kwargs)
    return model


class OCRBlock(nn.Module):
    def __init__(self, in_channels, key_channels, out_channels):
        super(OCRBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)
        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels + key_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        # x: 高分辨率特征图 (e.g., from HRNet's last stage)
        context = self.conv1x1(x)
        attention = self.softmax(context)
        context = torch.sum(x * attention, dim=(2, 3), keepdim=True)
        output = torch.cat([x, context.expand_as(x)], dim=1)
        return self.last_conv(output)


class HRNetDecoder(nn.Module):
    def __init__(self, hrnet_backbone, num_classes):
        super(HRNetDecoder, self).__init__()
        self.backbone = hrnet_backbone
        self.ocr = OCRBlock(in_channels=720, key_channels=256, out_channels=512)  # 调整通道数
        self.cls_head = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, x):
        # HRNet Encoder
        features = self.backbone(x)  # 假设返回多尺度特征图

        # 选择最高分辨率特征图 (e.g., 1/4尺度)
        hr_feature = features[0]  # [batch, 32, H/4, W/4]

        # OCR Decoder
        ocr_output = self.ocr(hr_feature)
        logits = self.cls_head(ocr_output)
        return F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=True)


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels, height=128, width=128):
        """
        完全修正的2D位置编码

        Args:
            channels: 位置编码的通道数 (必须是偶数)
            height: 图像高度
            width: 图像宽度
        """
        super(PositionalEncoding2D, self).__init__()

        if channels % 2 != 0:
            raise ValueError("通道数必须是偶数")

        self.channels = channels
        self.height = height
        self.width = width

        # 创建位置编码张量
        pe = torch.zeros(channels, height, width)

        # 每个通道的除数项
        div_term = torch.exp(torch.arange(0., channels // 2, 1) *  # 修改为1步长
                             -(math.log(10000.0) / (channels // 2)))

        # 生成高度方向的位置编码
        for h in range(height):
            for c in range(0, channels // 2):
                pe[2 * c, h, :] = torch.sin(h * div_term[c])
                pe[2 * c + 1, h, :] = torch.cos(h * div_term[c])

        # 生成宽度方向的位置编码
        for w in range(width):
            for c in range(0, channels // 2):
                pe[2 * c, :, w] += torch.sin(w * div_term[c])
                pe[2 * c + 1, :, w] += torch.cos(w * div_term[c])

        # 标准化
        pe = pe / pe.max()

        # 注册为buffer (不参与训练)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, channels, height, width]

    def forward(self, x):
        """
        添加位置编码到输入

        Args:
            x: 输入张量 [bs, C, H, W]

        Returns:
            添加位置编码后的张量 [bs, C, H, W]
        """
        return x + self.pe[:, :self.channels, :self.height, :self.width]


class MLPSegmentationWithPE(nn.Module):
    def __init__(self, input_channels=10, num_classes=5, hidden_dim=512, pe_dim=32):
        """
        带修正位置编码的MLP分割网络

        Args:
            input_channels: 输入通道数 (default: 10)
            num_classes: 输出类别数 (default: 5)
            hidden_dim: 隐藏层维度 (default: 512)
            pe_dim: 位置编码维度 (default: 32)
        """
        super(MLPSegmentationWithPE, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.pe_dim = pe_dim

        # 位置编码层 (独立于输入通道)
        self.pos_encoder = PositionalEncoding2D(pe_dim)

        # 计算展平后的维度 (输入通道 + 位置编码通道)
        self.flattened_dim = 128 * 128 * (input_channels + pe_dim)

        # MLP层
        self.mlp = nn.Sequential(
            nn.Linear(self.flattened_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128 * 128 * num_classes)
        )

    def forward(self, x):
        bs = x.shape[0]

        # 生成位置编码 [bs, pe_dim, 128, 128]
        pe = self.pos_encoder(torch.zeros(bs, self.pe_dim, 128, 128, device=x.device))

        # 拼接输入和位置编码 [bs, 10+pe_dim, 128, 128]
        x_with_pe = torch.cat([x, pe], dim=1)

        # 展平输入 [bs, (10+pe_dim)*128*128]
        x_flat = x_with_pe.view(bs, -1)

        # 通过MLP [bs, 128*128*num_classes]
        out_flat = self.mlp(x_flat)

        # 重塑为 [bs, num_classes, 128, 128]
        out = out_flat.view(bs, self.num_classes, 128, 128)

        return out

# 使用示例
if __name__ == '__main__':
    model = hrnet(num_classes=20)
    model = model.to(device=device)
    input_tensor = torch.randn(1, 10, 128, 128).to(device=device)
    output = model(input_tensor)
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")