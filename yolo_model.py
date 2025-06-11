import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1):
        super().__init__()
        hidden_channels = out_channels // 2
        
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1)
        self.conv2 = ConvBlock(in_channels, hidden_channels, 1)
        
        self.blocks = nn.Sequential(*[
            self._make_block(hidden_channels) for _ in range(num_blocks)
        ])
        
        self.conv3 = ConvBlock(hidden_channels * 2, out_channels, 1)
    
    def _make_block(self, channels):
        return nn.Sequential(
            ConvBlock(channels, channels, 1),
            ConvBlock(channels, channels, 3)
        )
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.blocks(x2)
        return self.conv3(torch.cat([x1, x2], dim=1))

class SPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden_channels = in_channels // 2
        
        self.conv1 = ConvBlock(in_channels, hidden_channels, 1)
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2) 
            for k in [5, 9, 13]
        ])
        self.conv2 = ConvBlock(hidden_channels * 4, out_channels, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [pool(x) for pool in self.pools], dim=1))

class YOLOHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_outputs = num_anchors * (5 + num_classes)
        
        self.conv = nn.Conv2d(in_channels, self.num_outputs, 1)
    
    def forward(self, x):
        return self.conv(x)

class YOLOv5(nn.Module):
    def __init__(self, num_classes=5, input_channels=3):
        super().__init__()
        self.num_classes = num_classes
        
        self.backbone = self._build_backbone(input_channels)
        self.neck = self._build_neck()
        self.head = self._build_head()
        
        self._initialize_weights()
    
    def _build_backbone(self, input_channels):
        return nn.Sequential(
            ConvBlock(input_channels, 32, 6, 2, 2),
            ConvBlock(32, 64, 3, 2),
            CSPBlock(64, 64, 1),
            ConvBlock(64, 128, 3, 2),
            CSPBlock(128, 128, 3),
            ConvBlock(128, 256, 3, 2),
            CSPBlock(256, 256, 3),
            ConvBlock(256, 512, 3, 2),
            CSPBlock(512, 512, 1),
            ConvBlock(512, 1024, 3, 2),
            CSPBlock(1024, 1024, 1),
            SPP(1024, 1024)
        )
    
    def _build_neck(self):
        return nn.ModuleDict({
            'up1': nn.Upsample(scale_factor=2, mode='nearest'),
            'conv1': ConvBlock(1024, 512, 1),
            'csp1': CSPBlock(1024, 512, 1),
            
            'up2': nn.Upsample(scale_factor=2, mode='nearest'),
            'conv2': ConvBlock(512, 256, 1),
            'csp2': CSPBlock(512, 256, 1),
            
            'down1': ConvBlock(256, 256, 3, 2),
            'csp3': CSPBlock(512, 512, 1),
            
            'down2': ConvBlock(512, 512, 3, 2),
            'csp4': CSPBlock(1024, 1024, 1)
        })
    
    def _build_head(self):
        return nn.ModuleDict({
            'small': YOLOHead(256, self.num_classes),
            'medium': YOLOHead(512, self.num_classes),
            'large': YOLOHead(1024, self.num_classes)
        })
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = []
        
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in [4, 6, 11]:
                features.append(x)
        
        c3, c4, c5 = features
        
        p5 = c5
        p5_up = self.neck['up1'](self.neck['conv1'](p5))
        p4 = self.neck['csp1'](torch.cat([p5_up, c4], dim=1))
        
        p4_up = self.neck['up2'](self.neck['conv2'](p4))
        p3 = self.neck['csp2'](torch.cat([p4_up, c3], dim=1))
        
        p3_down = self.neck['down1'](p3)
        p4 = self.neck['csp3'](torch.cat([p3_down, p4], dim=1))
        
        p4_down = self.neck['down2'](p4)
        p5 = self.neck['csp4'](torch.cat([p4_down, p5], dim=1))
        
        outputs = [
            self.head['small'](p3),
            self.head['medium'](p4),
            self.head['large'](p5)
        ]
        
        return outputs

def create_yolo_model(num_classes=5):
    return YOLOv5(num_classes=num_classes)