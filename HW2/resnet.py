import torch
import torch.nn as nn
from torch import Tensor


class BasicBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # residual
        out = self.relu(out)
        return out


class ResNet18(nn.Module):

    def __init__(self, in_channels: int = 3, num_classes: int = 1000) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    @staticmethod  # Avoid passing whole model to it
    def _make_layer(
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet18FCN(ResNet18):
    """
    - Drop the classification head.
    - Add a 1x1 conv -> project the last feature map to num_classes.
    - Upsample 32x back to the original resolution
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 13) -> None:
        super().__init__(in_channels=in_channels, num_classes=num_classes)

        del self.avgpool
        del self.fc

        self.score = nn.Conv2d(512, num_classes, kernel_size=1)

        self.upsample = nn.Upsample(
            scale_factor=32, mode="bilinear", align_corners=False
        )

        nn.init.kaiming_normal_(self.score.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.score.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.score(x)
        x = self.upsample(x)

        return x


class DoubleConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class UNet(nn.Module):

    def __init__(self, in_channels: int = 3, num_classes: int = 13) -> None:
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4 = DoubleConv(1024 + 512, 512)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = DoubleConv(512 + 256, 256)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DoubleConv(256 + 128, 128)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = DoubleConv(128 + 64, 64)

        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        x = self.bottleneck(self.pool(s4))

        x = self.dec4(torch.cat([self.up4(x), s4], dim=1))
        x = self.dec3(torch.cat([self.up3(x), s3], dim=1))
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))

        return self.head(x)


class UNetColorization(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(1,   64)
        self.enc2 = DoubleConv(64,  128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(512, 1024)

        self.up4  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4 = DoubleConv(1024 + 512, 512)
        self.up3  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = DoubleConv(512 + 256, 256)
        self.up2  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DoubleConv(256 + 128, 128)
        self.up1  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = DoubleConv(128 + 64, 64)

        self.head = nn.Sequential(nn.Conv2d(64, 2, 1), nn.Tanh())

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))
        x  = self.bottleneck(self.pool(s4))
        x  = self.dec4(torch.cat([self.up4(x), s4], dim=1))
        x  = self.dec3(torch.cat([self.up3(x), s3], dim=1))
        x  = self.dec2(torch.cat([self.up2(x), s2], dim=1))
        x  = self.dec1(torch.cat([self.up1(x), s1], dim=1))
        return self.head(x)


class EncoderDecoderColorization(nn.Module):
    """Same architecture but no skip connections."""
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(1,   64)
        self.enc2 = DoubleConv(64,  128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = DoubleConv(512, 1024)

        self.up4  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4 = DoubleConv(1024, 512)          # no concat → half channels
        self.up3  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = DoubleConv(512,  256)
        self.up2  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DoubleConv(256,  128)
        self.up1  = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = DoubleConv(128,  64)

        self.head = nn.Sequential(nn.Conv2d(64, 2, 1), nn.Tanh())

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(self.pool(x))
        x = self.enc3(self.pool(x))
        x = self.enc4(self.pool(x))
        x = self.bottleneck(self.pool(x))
        x = self.dec4(self.up4(x))
        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        x = self.dec1(self.up1(x))
        return self.head(x)

if __name__ == "__main__":
    B, C, H, W = 2, 3, 224, 224
    x = torch.randn(B, C, H, W)

    fcn = ResNet18FCN(num_classes=13)
    out = fcn(x)
    print(f"FCN  output : {out.shape}") 
    print(f"FCN  params : {sum(p.numel() for p in fcn.parameters()):,}")

    unet = UNet(num_classes=13)
    out = unet(x)
    print(f"UNet output : {out.shape}")  
    print(f"UNet params : {sum(p.numel() for p in unet.parameters()):,}")
