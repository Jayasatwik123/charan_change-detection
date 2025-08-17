#  Change Guiding Network: Incorporating Change Prior to Guide Change Detection in Remote Sensing Imagery,
#  IEEE J. SEL. TOP. APPL. EARTH OBS. REMOTE SENS., PP. 1–17, 2023, DOI: 10.1109/JSTARS.2023.3310208. C. HAN, C. WU, H. GUO, M. HU, J.Li AND H. CHEN,


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChangeGuideModule(nn.Module):
    def __init__(self, in_dim):
        super(ChangeGuideModule, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, guiding_map0):
        m_batchsize, C, height, width = x.size()

        guiding_map0 = F.interpolate(guiding_map0, x.size()[2:], mode='bilinear', align_corners=True)

        guiding_map = torch.sigmoid(guiding_map0)

        query = self.query_conv(x) * (1 + guiding_map)
        proj_query = query.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        key = self.key_conv(x) * (1 + guiding_map)
        proj_key = key.view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        self.energy = energy
        self.attention = attention

        value = self.value_conv(x) * (1 + guiding_map)
        proj_value = value.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x

        return out


class HCGMNet(nn.Module):
    def __init__(self, backbone_name='vgg16'):
        super(HCGMNet, self).__init__()

        if backbone_name == 'vgg16':
            # Use VGG16 without pretrained weights to avoid internet download
            vgg16_bn = models.vgg16_bn(pretrained=False)
            
            self.inc = vgg16_bn.features[:5]  # Output: 64 channels
            self.down1 = vgg16_bn.features[5:12]  # Output: 128 channels
            self.down2 = vgg16_bn.features[12:22]  # Output: 256 channels
            self.down3 = vgg16_bn.features[22:32]  # Output: 512 channels
            self.down4 = vgg16_bn.features[32:42]  # Output: 512 channels

            # conv_reduce layers for VGG16
            self.conv_reduce_1 = BasicConv2d(128*2, 128, 3, 1, 1)
            self.conv_reduce_2 = BasicConv2d(256*2, 256, 3, 1, 1)
            self.conv_reduce_3 = BasicConv2d(512*2, 512, 3, 1, 1)
            self.conv_reduce_4 = BasicConv2d(512*2, 512, 3, 1, 1)

            # up_layer modules for VGG16
            self.up_layer4 = BasicConv2d(512, 512, 3, 1, 1)
            self.up_layer3 = BasicConv2d(512, 512, 3, 1, 1)
            self.up_layer2 = BasicConv2d(256, 256, 3, 1, 1)

            # Decoder input channels for VGG16: 128 (layer1) + 256 (layer2_1) + 512 (layer3_1) + 512 (layer4_1) = 1408
            decoder_input_channels = 1408
            
            # ChangeGuideModules for VGG16
            # Inputs are layer2, layer3, layer4 after conv_reduce and up_layer
            # layer2 (from up_layer2) is 256 channels
            # layer3 (from up_layer3) is 512 channels
            # layer4 (from up_layer4) is 512 channels
            self.cgm_2 = ChangeGuideModule(256)
            self.cgm_3 = ChangeGuideModule(512)
            self.cgm_4 = ChangeGuideModule(512)

        elif backbone_name == 'resnet34':
            # Use ResNet34 without pretrained weights to avoid internet download
            resnet = models.resnet34(pretrained=False)
            
            self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) # Output: 64 channels
            self.down1 = resnet.layer1 # Output: 64 channels
            self.down2 = resnet.layer2 # Output: 128 channels
            self.down3 = resnet.layer3 # Output: 256 channels
            self.down4 = resnet.layer4 # Output: 512 channels

            # conv_reduce layers for ResNet34
            self.conv_reduce_1 = BasicConv2d(64*2, 64, 3, 1, 1)
            self.conv_reduce_2 = BasicConv2d(128*2, 128, 3, 1, 1)
            self.conv_reduce_3 = BasicConv2d(256*2, 256, 3, 1, 1)
            self.conv_reduce_4 = BasicConv2d(512*2, 512, 3, 1, 1)

            # up_layer modules for ResNet34
            # Output of conv_reduce_4 is 512, conv_reduce_3 is 256, conv_reduce_2 is 128
            self.up_layer4 = BasicConv2d(512, 512, 3, 1, 1) 
            self.up_layer3 = BasicConv2d(256, 256, 3, 1, 1) 
            self.up_layer2 = BasicConv2d(128, 128, 3, 1, 1)

            # Decoder input channels for ResNet34: 64 (layer1) + 128 (layer2_1) + 256 (layer3_1) + 512 (layer4_1) = 960
            decoder_input_channels = 960
            
            # ChangeGuideModules for ResNet34
            # Inputs are layer2, layer3, layer4 after conv_reduce and up_layer
            # layer2 (from up_layer2) is 128 channels
            # layer3 (from up_layer3) is 256 channels
            # layer4 (from up_layer4) is 512 channels
            self.cgm_2 = ChangeGuideModule(128)
            self.cgm_3 = ChangeGuideModule(256)
            self.cgm_4 = ChangeGuideModule(512)
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose 'vgg16' or 'resnet34'.")

        # Common decoder structure, channel sizes depend on backbone
        self.deocde = nn.Sequential(BasicConv2d(decoder_input_channels, 512, 3, 1, 1), BasicConv2d(512, 256, 3, 1, 1),BasicConv2d(256, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))
        self.deocde_final = nn.Sequential(BasicConv2d(decoder_input_channels, 512, 3, 1, 1), BasicConv2d(512, 256, 3, 1, 1),BasicConv2d(256, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))

    def forward(self,A,B):
        size = A.size()[2:]
        layer1_pre = self.inc(A)
        layer1_A = self.down1(layer1_pre)
        layer2_A = self.down2(layer1_A)
        layer3_A = self.down3(layer2_A)
        layer4_A = self.down4(layer3_A)

        layer1_pre = self.inc(B)
        layer1_B = self.down1(layer1_pre)
        layer2_B = self.down2(layer1_B)
        layer3_B = self.down3(layer2_B)
        layer4_B = self.down4(layer3_B)

        layer1 = torch.cat((layer1_B,layer1_A),dim=1)

        layer2 = torch.cat((layer2_B,layer2_A),dim=1)

        layer3 = torch.cat((layer3_B,layer3_A),dim=1)

        layer4 = torch.cat((layer4_B,layer4_A),dim=1)

        layer1 = self.conv_reduce_1(layer1)
        layer2 = self.conv_reduce_2(layer2)
        layer3 = self.conv_reduce_3(layer3)
        layer4 = self.conv_reduce_4(layer4)

        layer4 = self.up_layer4(layer4)

        layer3 = self.up_layer3(layer3)

        layer2 = self.up_layer2(layer2)

        layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer3_1 = F.interpolate(layer3, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer2_1 = F.interpolate(layer2, layer1.size()[2:], mode='bilinear', align_corners=True)

        feature_fuse = torch.cat((layer1,layer2_1,layer3_1,layer4_1),dim=1)
        change_map = self.deocde(feature_fuse)

        layer2 = self.cgm_2(layer2, change_map)
        layer3 = self.cgm_3(layer3, change_map)
        layer4 = self.cgm_4(layer4, change_map)

        layer4_2 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer3_2 = F.interpolate(layer3, layer1.size()[2:], mode='bilinear', align_corners=True)
        layer2_2 = F.interpolate(layer2, layer1.size()[2:], mode='bilinear', align_corners=True)

        new_feature_fuse = torch.cat((layer1,layer2_2,layer3_2,layer4_2),dim=1)

        change_map = F.interpolate(change_map, size, mode='bilinear', align_corners=True)
        final_map = self.deocde_final(new_feature_fuse)

        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)

        return change_map, final_map

#增加decoder解码器，不concat多尺度特征生成guide map,也concat多尺度特征生成最后输出

# ---------------- ASPP Module ----------------

# --- Tuned ASPP Module for LEVIR-CD-256 ---
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=512, dilations=(1, 4, 8, 12, 24)):
        super(ASPP, self).__init__()
        self.aspp_blocks = nn.ModuleList()
        for dilation in dilations:
            self.aspp_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU()  # Use GELU for smoother activation
                )
            )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(0.3)  # Add dropout for regularization
        )

    def forward(self, x):
        aspp_outs = [block(x) for block in self.aspp_blocks]
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=True)
        aspp_outs.append(global_feat)
        x = torch.cat(aspp_outs, dim=1)
        x = self.project(x)
        return x


class CGNet(nn.Module):
    def __init__(self, backbone_name='vgg16', use_aspp=True):
        super(CGNet, self).__init__()

        self.use_aspp = use_aspp

        if backbone_name == 'vgg16':
            # Use VGG16 without pretrained weights to avoid internet download
            vgg16_bn = models.vgg16_bn(pretrained=False)
            self.inc = vgg16_bn.features[:5]  # 64
            self.down1 = vgg16_bn.features[5:12]  # 128
            self.down2 = vgg16_bn.features[12:22]  # 256
            self.down3 = vgg16_bn.features[22:32]  # 512
            self.down4 = vgg16_bn.features[32:42]  # 512

            self.conv_reduce_1 = BasicConv2d(128*2, 128, 3, 1, 1)
            self.conv_reduce_2 = BasicConv2d(256*2, 256, 3, 1, 1)
            self.conv_reduce_3 = BasicConv2d(512*2, 512, 3, 1, 1)
            self.conv_reduce_4 = BasicConv2d(512*2, 512, 3, 1, 1)

            if self.use_aspp:
                self.aspp = ASPP(512, 512)

            self.up_layer4 = BasicConv2d(512, 512, 3, 1, 1)
            self.up_layer3 = BasicConv2d(512, 512, 3, 1, 1)
            self.up_layer2 = BasicConv2d(256, 256, 3, 1, 1)

            self.decoder = nn.Sequential(BasicConv2d(512, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))
            
            self.cgm_4 = ChangeGuideModule(512)
            self.decoder_module4 = BasicConv2d(512 + 512, 512, 3, 1, 1)
            self.cgm_3 = ChangeGuideModule(512)
            self.decoder_module3 = BasicConv2d(512 + 256, 256, 3, 1, 1)
            self.cgm_2 = ChangeGuideModule(256)
            self.decoder_module2 = BasicConv2d(256 + 128, 128, 3, 1, 1)
            
            self.decoder_final = nn.Sequential(BasicConv2d(128, 64, 3, 1, 1), nn.Conv2d(64, 1, 1))

        elif backbone_name == 'resnet34':
            # Use ResNet34 without pretrained weights to avoid internet download
            resnet = models.resnet34(pretrained=False)
            self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool) # 64
            self.down1 = resnet.layer1 # 64
            self.down2 = resnet.layer2 # 128
            self.down3 = resnet.layer3 # 256
            self.down4 = resnet.layer4 # 512

            self.conv_reduce_1 = BasicConv2d(64*2, 64, 3, 1, 1)
            self.conv_reduce_2 = BasicConv2d(128*2, 128, 3, 1, 1)
            self.conv_reduce_3 = BasicConv2d(256*2, 256, 3, 1, 1)
            self.conv_reduce_4 = BasicConv2d(512*2, 512, 3, 1, 1)

            if self.use_aspp:
                self.aspp = ASPP(512, 512)

            self.decoder = nn.Sequential(BasicConv2d(512, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))

            self.cgm_4 = ChangeGuideModule(512)
            self.decoder_module4 = BasicConv2d(512 + 256, 256, 3, 1, 1)
            self.cgm_3 = ChangeGuideModule(256)
            self.decoder_module3 = BasicConv2d(256 + 128, 128, 3, 1, 1)
            self.cgm_2 = ChangeGuideModule(128)
            self.decoder_module2 = BasicConv2d(128 + 64, 64, 3, 1, 1)
            
            self.decoder_final = nn.Sequential(BasicConv2d(64, 64, 3, 1, 1), nn.Conv2d(64, 1, 1))
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose 'vgg16' or 'resnet34'.")

        # Common modules
        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, A, B):
        size = A.size()[2:]
        layer1_pre = self.inc(A)
        layer1_A = self.down1(layer1_pre)
        layer2_A = self.down2(layer1_A)
        layer3_A = self.down3(layer2_A)
        layer4_A = self.down4(layer3_A)

        layer1_pre = self.inc(B)
        layer1_B = self.down1(layer1_pre)
        layer2_B = self.down2(layer1_B)
        layer3_B = self.down3(layer2_B)
        layer4_B = self.down4(layer3_B)

        layer1 = torch.cat((layer1_B, layer1_A), dim=1)
        layer2 = torch.cat((layer2_B, layer2_A), dim=1)
        layer3 = torch.cat((layer3_B, layer3_A), dim=1)
        layer4 = torch.cat((layer4_B, layer4_A), dim=1)

        layer1 = self.conv_reduce_1(layer1)
        layer2 = self.conv_reduce_2(layer2)
        layer3 = self.conv_reduce_3(layer3)
        layer4 = self.conv_reduce_4(layer4)

        if self.use_aspp:
            layer4 = self.aspp(layer4)

        layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        feature_fuse = layer4_1  # ASPP output is used for guide map

        change_map = self.decoder(feature_fuse)

        layer4 = self.cgm_4(layer4, change_map)
        feature4 = self.decoder_module4(torch.cat([self.upsample2x(layer4), layer3], 1))
        layer3 = self.cgm_3(feature4, change_map)
        feature3 = self.decoder_module3(torch.cat([self.upsample2x(layer3), layer2], 1))
        layer2 = self.cgm_2(feature3, change_map)
        layer1 = self.decoder_module2(torch.cat([self.upsample2x(layer2), layer1], 1))

        change_map = F.interpolate(change_map, size, mode='bilinear', align_corners=True)
        final_map = self.decoder_final(layer1)
        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)
        return change_map, final_map

class CGNet_Ablation(nn.Module):
    def __init__(self, backbone_name='vgg16'):
        super(CGNet_Ablation, self).__init__()

        if backbone_name == 'vgg16':
            # Use VGG16 without pretrained weights to avoid internet download
            vgg16_bn = models.vgg16_bn(pretrained=False)
            self.inc = vgg16_bn.features[:5]
            self.down1 = vgg16_bn.features[5:12]
            self.down2 = vgg16_bn.features[12:22]
            self.down3 = vgg16_bn.features[22:32]
            self.down4 = vgg16_bn.features[32:42]

            self.conv_reduce_1 = BasicConv2d(128*2, 128, 3, 1, 1)
            self.conv_reduce_2 = BasicConv2d(256*2, 256, 3, 1, 1)
            self.conv_reduce_3 = BasicConv2d(512*2, 512, 3, 1, 1)
            self.conv_reduce_4 = BasicConv2d(512*2, 512, 3, 1, 1)

            self.up_layer4 = BasicConv2d(512,512,3,1,1)
            self.up_layer3 = BasicConv2d(512,512,3,1,1)
            self.up_layer2 = BasicConv2d(256,256,3,1,1)

            self.decoder = nn.Sequential(BasicConv2d(512, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))
            
            self.cgm_4 = ChangeGuideModule(512)
            self.decoder_module4 = BasicConv2d(512 + 512, 512, 3, 1, 1)
            self.cgm_3 = ChangeGuideModule(512)
            self.decoder_module3 = BasicConv2d(512 + 256, 256, 3, 1, 1)
            self.cgm_2 = ChangeGuideModule(256)
            self.decoder_module2 = BasicConv2d(256 + 128, 128, 3, 1, 1)
            
            self.decoder_final = nn.Sequential(BasicConv2d(128, 64, 3, 1, 1), nn.Conv2d(64, 1, 1))

        elif backbone_name == 'resnet34':
            try:
                resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            except AttributeError:
                resnet = models.resnet34(pretrained=True)
            self.inc = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
            self.down1 = resnet.layer1
            self.down2 = resnet.layer2
            self.down3 = resnet.layer3
            self.down4 = resnet.layer4

            self.conv_reduce_1 = BasicConv2d(64*2, 64, 3, 1, 1)
            self.conv_reduce_2 = BasicConv2d(128*2, 128, 3, 1, 1)
            self.conv_reduce_3 = BasicConv2d(256*2, 256, 3, 1, 1)
            self.conv_reduce_4 = BasicConv2d(512*2, 512, 3, 1, 1)

            self.decoder = nn.Sequential(BasicConv2d(512, 64, 3, 1, 1), nn.Conv2d(64, 1, 3, 1, 1))

            self.cgm_4 = ChangeGuideModule(512)
            self.decoder_module4 = BasicConv2d(512 + 256, 256, 3, 1, 1)
            self.cgm_3 = ChangeGuideModule(256)
            self.decoder_module3 = BasicConv2d(256 + 128, 128, 3, 1, 1)
            self.cgm_2 = ChangeGuideModule(128)
            self.decoder_module2 = BasicConv2d(128 + 64, 64, 3, 1, 1)
            
            self.decoder_final = nn.Sequential(BasicConv2d(64, 64, 3, 1, 1), nn.Conv2d(64, 1, 1))
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose 'vgg16' or 'resnet34'.")

        # Common modules
        self.upsample2x = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self,A,B):
        size = A.size()[2:]
        layer1_pre = self.inc(A)
        layer1_A = self.down1(layer1_pre)
        layer2_A = self.down2(layer1_A)
        layer3_A = self.down3(layer2_A)
        layer4_A = self.down4(layer3_A)

        layer1_pre = self.inc(B)
        layer1_B = self.down1(layer1_pre)
        layer2_B = self.down2(layer1_B)
        layer3_B = self.down3(layer2_B)
        layer4_B = self.down4(layer3_B)

        layer1 = torch.cat((layer1_B,layer1_A),dim=1)

        layer2 = torch.cat((layer2_B,layer2_A),dim=1)

        layer3 = torch.cat((layer3_B,layer3_A),dim=1)

        layer4 = torch.cat((layer4_B,layer4_A),dim=1)

        layer1 = self.conv_reduce_1(layer1)
        layer2 = self.conv_reduce_2(layer2)
        layer3 = self.conv_reduce_3(layer3)
        layer4 = self.conv_reduce_4(layer4)

        layer4_1 = F.interpolate(layer4, layer1.size()[2:], mode='bilinear', align_corners=True)
        feature_fuse=layer4_1 #需要注释！

        change_map = self.decoder(feature_fuse) #需要注释！

        layer4 = self.cgm_4(layer4, change_map)
        feature4=self.decoder_module4(torch.cat([self.upsample2x(layer4),layer3],1))
        feature3 = self.decoder_module3(torch.cat([self.upsample2x(feature4),layer2],1))
        layer3 = self.cgm_2(feature3, change_map)
        layer1 = self.decoder_module2(torch.cat([self.upsample2x(layer3), layer1], 1))

        change_map = F.interpolate(change_map, size, mode='bilinear', align_corners=True)
        final_map = self.decoder_final(layer1)

        final_map = F.interpolate(final_map, size, mode='bilinear', align_corners=True)

        return change_map, final_map



if __name__=='__main__':
    input_size = 256 # Example input size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for __main__ tests in network/CGNet.py.\n")

    backbone_choices = ['vgg16', 'resnet34']

    for backbone in backbone_choices:
        print(f"\n=== Testing with Backbone: {backbone.upper()} ===")
        
        # Test HCGMNet
        print("\n--- HCGMNet ---")
        try:
            model_hcgm = HCGMNet(backbone_name=backbone).to(device)
            model_hcgm.eval()
            total_params_hcgm = sum(p.numel() for p in model_hcgm.parameters())
            trainable_params_hcgm = sum(p.numel() for p in model_hcgm.parameters() if p.requires_grad)
            print(f"HCGMNet ({backbone}) Total parameters: {total_params_hcgm:,}")
            print(f"HCGMNet ({backbone}) Trainable parameters: {trainable_params_hcgm:,}")
            
            dummy_A = torch.randn(1, 3, input_size, input_size).to(device)
            dummy_B = torch.randn(1, 3, input_size, input_size).to(device)
            with torch.no_grad():
                change_map, final_map = model_hcgm(dummy_A, dummy_B)
            print(f"HCGMNet ({backbone}) output shapes: change_map={change_map.shape}, final_map={final_map.shape}")
        except Exception as e:
            print(f"HCGMNet ({backbone}) test failed: {e}")

        # Test CGNet
        print("\n--- CGNet ---")
        try:
            model_cg = CGNet(backbone_name=backbone).to(device)
            model_cg.eval()
            total_params_cg = sum(p.numel() for p in model_cg.parameters())
            trainable_params_cg = sum(p.numel() for p in model_cg.parameters() if p.requires_grad)
            print(f"CGNet ({backbone}) Total parameters: {total_params_cg:,}")
            print(f"CGNet ({backbone}) Trainable parameters: {trainable_params_cg:,}")

            dummy_A = torch.randn(1, 3, input_size, input_size).to(device)
            dummy_B = torch.randn(1, 3, input_size, input_size).to(device)
            with torch.no_grad():
                change_map, final_map = model_cg(dummy_A, dummy_B)
            print(f"CGNet ({backbone}) output shapes: change_map={change_map.shape}, final_map={final_map.shape}")
        except Exception as e:
            print(f"CGNet ({backbone}) test failed: {e}")
        
        # Test CGNet_Ablation (Optional, can be added if needed)
        # print("\n--- CGNet_Ablation ---")
        # try:
        #     model_ablation = CGNet_Ablation(backbone_name=backbone).to(device)
        #     # ... similar parameter count and forward pass ...
        # except Exception as e:
        #     print(f"CGNet_Ablation ({backbone}) test failed: {e}")

    print("\nFinished __main__ tests in network/CGNet.py.")