import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=64):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = self.conv_block(in_channels, features)
        self.encoder2 = self.conv_block(features, features * 2)
        self.encoder3 = self.conv_block(features * 2, features * 4)
        self.encoder4 = self.conv_block(features * 4, features * 8)

        self.middle = self.conv_block(features * 8, features * 16)

        self.upconv4 = self.upconv_block(features * 16, features * 8)
        self.decoder4 = self.conv_block(features * 16, features * 8)

        self.upconv3 = self.upconv_block(features * 8, features * 4)
        self.decoder3 = self.conv_block(features * 8, features * 4)

        self.upconv2 = self.upconv_block(features * 4, features * 2)
        self.decoder2 = self.conv_block(features * 4, features * 2)

        self.upconv1 = self.upconv_block(features * 2, features)
        self.decoder1 = self.conv_block(features * 2, features)

        self.final_conv = nn.Conv2d(features, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        print(x)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder4(F.max_pool2d(enc3, 2))

        mid = self.middle(F.max_pool2d(enc4, 2))

        up4 = self.upconv4(mid)
        up4 = F.interpolate(up4, size=enc4.size()[2:], mode='bilinear', align_corners=False)  # 크기 맞추기
        up4 = torch.cat([up4, enc4], dim=1)
        up4 = self.decoder4(up4)

        up3 = self.upconv3(up4)
        up3 = F.interpolate(up3, size=enc3.size()[2:], mode='bilinear', align_corners=False)  # 크기 맞추기
        up3 = torch.cat([up3, enc3], dim=1)
        up3 = self.decoder3(up3)

        up2 = self.upconv2(up3)
        up2 = F.interpolate(up2, size=enc2.size()[2:], mode='bilinear', align_corners=False)  # 크기 맞추기
        up2 = torch.cat([up2, enc2], dim=1)
        up2 = self.decoder2(up2)

        up1 = self.upconv1(up2)
        up1 = F.interpolate(up1, size=enc1.size()[2:], mode='bilinear', align_corners=False)  # 크기 맞추기
        up1 = torch.cat([up1, enc1], dim=1)
        up1 = self.decoder1(up1)

        out = self.final_conv(up1)

        #print(out)
        return out

# 모델 인스턴스 생성
model = UNet(in_channels=1, out_channels=2)

# 가짜 입력 데이터 생성
input_tensor = torch.randn(1, 1, 572, 572) # 배치,채널,높이,너비

# 모델을 호출하여 forward 함수 실행
output = model(input_tensor)
pred = torch.argmax(output, dim=1)

print(pred)
