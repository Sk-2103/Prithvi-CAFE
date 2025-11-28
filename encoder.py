import torch
import torch.nn as nn
import torch.nn.functional as F
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY
from terratorch import BACKBONE_REGISTRY
### Adpaptd Prithvi v2 600 M backbone for efficent fin training####
from terratorch.models.necks import SelectIndices, ReshapeTokensToImage
#prithvi_backbone = BACKBONE_REGISTRY.build("prithvi_eo_v2_600", pretrained=False)
class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net


class MultiScaleAttentionFAM(nn.Module):
    def __init__(self, sam_channels_list, cnn_channels_list, initial_bias=0.8): ## original is 0.8, now testing with different attention
        """
        Args:
            sam_channels_list (list[int]): channels of each SAM feature map (len=4, e.g., [128, 256, 512, 1024])
            cnn_channels_list (list[int]): channels of each CNN feature map (len=4, e.g., [64, 128, 256, 512])
            initial_bias (float): initial bias factor for attention weighting
        """
        super(MultiScaleAttentionFAM, self).__init__()

        assert len(cnn_channels_list) == 4 and len(sam_channels_list) == 4, "Expected 4 CNN and 4 SAM feature maps"

        # Attention conv for each scale
        self.attention_convs = nn.ModuleList([
            nn.Conv2d(sam_c + cnn_c, sam_c, kernel_size=1)
            for sam_c, cnn_c in zip(sam_channels_list, cnn_channels_list)
        ])

        # Projection conv to match CNN channels to SAM channels
        self.cnn_projs = nn.ModuleList([
            nn.Conv2d(cnn_c, sam_c, kernel_size=1)
            for sam_c, cnn_c in zip(sam_channels_list, cnn_channels_list)
        ])

        self.sigmoid = nn.Sigmoid()
        self.bias_factor = nn.Parameter(torch.tensor(initial_bias))

    def forward(self, sam_features, cnn_features):
        """
        Args:
            sam_features: list of 4 tensors, each (B, C_sam_i, H_i, W_i)
            cnn_features: list of 4 tensors, each (B, C_cnn_i, H'_i, W'_i)
        Returns:
            fused_features_list: list of 4 tensors, each (B, C_sam_i, H_i, W_i)
        """
        assert len(sam_features) == 4 and len(cnn_features) == 4, "Need 4 SAM and 4 CNN features"

        fused_outputs = []
        for i in range(4):
            feature_sam = sam_features[i]
            feature_cnn = cnn_features[i]

            # Upsample CNN feature to SAM size
            feature_cnn = F.interpolate(feature_cnn, size=feature_sam.shape[-2:], mode="bilinear", align_corners=False)

            # Project CNN feature to match SAM channels
            feature_cnn_proj = self.cnn_projs[i](feature_cnn)

            # Compute attention
            combined = torch.cat([feature_sam, feature_cnn], dim=1)
            attention = self.sigmoid(self.attention_convs[i](combined))

            # Apply bias factor
            attention = attention * (1 - self.bias_factor) + self.bias_factor

            # Fuse
            fused = attention * feature_sam + (1 - attention) * feature_cnn_proj
            fused_outputs.append(fused)

        return fused_outputs


# FAT_Net
class FAT_Net(nn.Module):
    def __init__(self):
        super(FAT_Net, self).__init__()
        self.cnn = CNN_Module()

        self.multi_fam = MultiScaleAttentionFAM(sam_channels_list = [160, 320, 640, 1280], cnn_channels_list = [128, 256, 512, 1024])

    def forward(self, feature_sam, image):

        cnn_feats = self.cnn(image)


        outputs = self.multi_fam(feature_sam, cnn_feats)

        return outputs


# CBAM
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        channel_weights = self.channel_attention(x)
        x = x * channel_weights


        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_weights = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_weights

        return x



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_cbam=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )


        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(out_channels)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        if self.use_cbam:
            out = self.cbam(out)
        out = self.relu(out)
        return out


# CNN
class CNN_Module(nn.Module):
    def __init__(self):
        super(CNN_Module, self).__init__()
        self.block1 = ResidualBlock(7, 128, use_cbam=True)  # 添加 CBAM   ##cnn channal will be remaining 7 
        self.block2 = ResidualBlock(128, 256, stride=2, use_cbam=True)
        self.block3 = ResidualBlock(256, 512, stride=2, use_cbam=True)
        self.block4 = ResidualBlock(512, 1024, stride=2, use_cbam=True)

    def forward(self, x):
        # 提取多尺度特征
        scale1 = self.block1(x)  # (B, 64, H, W)
        scale2 = self.block2(scale1)  # (B, 128, H/2, W/2)
        scale3 = self.block3(scale2)  # (B, 256, H/4, W/4)
        scale4 = self.block4(scale3)  # (B, 512, H/8, W/8)

        return scale1, scale2, scale3, scale4


#class CNN_Module(nn.Module):
#    def __init__(self):
#        super(CNN_Module, self).__init__()
#        self.block1 = ResidualBlock(7, 64, use_cbam=True)  # 添加 CBAM   ##cnn channal will be remaining 7 
#        self.block2 = ResidualBlock(64, 128, stride=2, use_cbam=True)
#        self.block3 = ResidualBlock(128, 256, stride=2, use_cbam=True)
#        self.block4 = ResidualBlock(256, 512, stride=2, use_cbam=True)

@TERRATORCH_BACKBONE_REGISTRY.register
class AdaptedPrithvi(nn.Module):
    def __init__(self, in_channels=6, **kwargs):
        super(AdaptedPrithvi, self).__init__()
        self.encoder = BACKBONE_REGISTRY.build("prithvi_eo_v2_600", in_channels=in_channels, pretrained=True)
        self.fat_net = FAT_Net()
        #self.decoder = PT2Decoder()

        # Freeze original backbone parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Wrap each block with an Adapter
        adapted_blocks = [Adapter(block) for block in self.encoder.blocks]
        self.encoder.blocks = nn.Sequential(*adapted_blocks)

        # Forward out_channels from the base encoder
        if hasattr(self.encoder, "out_channels"):
            self.out_channels = self.encoder.out_channels
        else:
            raise AttributeError("The wrapped Prithvi encoder must have `out_channels` defined.")

        self.select_feat = SelectIndices(indices = [7,15,23,31], channel_list=[1280, 1280, 1280, 1280])

        # You'll likely want to reshape the tokens to an image format next
        self.reshape_tokens = ReshapeTokensToImage(channel_list=[1280, 1280, 1280, 1280])
        embed_dim = [1280, 1280, 1280, 1280]
        self.fpn1 = nn.Sequential(
          nn.ConvTranspose2d(embed_dim[0], embed_dim[0] // 2, kernel_size=2, stride=2),
          nn.BatchNorm2d(embed_dim[0] // 2),
          nn.GELU(),
          nn.ConvTranspose2d(embed_dim[0] // 2, embed_dim[0] // 4, kernel_size=2, stride=2),
          nn.BatchNorm2d(embed_dim[0] // 4),
          nn.GELU(),
          nn.ConvTranspose2d(embed_dim[0] // 4, embed_dim[0] // 8, kernel_size=2, stride=2)
          )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim[1], embed_dim[1] // 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(embed_dim[0] // 2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim[1] // 2, embed_dim[1] // 4, kernel_size=2, stride=2)
        )

        # self.fpn3 = nn.Sequential(
        #     nn.Conv2d(embed_dim[2], embed_dim[2]//2, kernel_size=1),
        #     nn.BatchNorm2d(embed_dim[0] // 2),
        #     nn.GELU(),  # optional channel adjust
        # )
        self.fpn3 = nn.Sequential(
          nn.ConvTranspose2d(embed_dim[2], embed_dim[2]//2, kernel_size=2, stride=2),
          nn.BatchNorm2d(embed_dim[2]//2),
          nn.GELU()
        )

        self.fpn4 = nn.Sequential(
            nn.Conv2d(embed_dim[3], embed_dim[3], kernel_size=1)  # keep spatial as is
        )

        #self.embed_dim = [embed_dim[0] // 4, embed_dim[1] // 2, embed_dim[2], embed_dim[3]]



    def forward(self, x):
        # Define which channels go to encoder
        encoder_idx = [1, 2, 3, 7, 11, 12]    ##Prthivi selected indices
        
        # Compute complementary indices (channels not in encoder_idx)
        all_idx = list(range(x.shape[1]))
        fatnet_idx = [i for i in all_idx if i not in encoder_idx]
        
        # Split inputs
        x_enc = x[:, encoder_idx, :, :]   # Channels for prthvi encoder
        x_fat = x[:, fatnet_idx, :, :]    # Remaining channels for cnn backbone
        
        # Encoder forward
        feat = self.encoder(x_enc)
        features = self.select_feat(feat)
        features = self.reshape_tokens(features)

        # FPN processing
        f7  = self.fpn1(features[0])   # 8x up, channel reduced
        f15 = self.fpn2(features[1])   # 2x up, channel reduced
        f23 = self.fpn3(features[2])   # 1x up, channel reduced
        f31 = self.fpn4(features[3])   # identity or 1x1 conv

        features = [f7, f15, f23, f31]

        # Pass remaining channels to fat_net along with features
        fused_embedding = self.fat_net(features, x_fat)

        return fused_embedding
