import torch
import numbers
import torch.nn as nn
from thop import profile
from einops import rearrange
from attention import joint_attention, AOTBlock, Attention, Top_Attention

def convrelu(in_channels, out_channels, kernel, padding, pool):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(pool, stride=pool, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    )

def convreluT(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=2, padding=padding),
        nn.ReLU(inplace=True)
    )

class Multi_model(nn.Module):

    def __init__(self, heads=(1, 2, 4, 8), bias=False, inputs=(6, 1)):

        super().__init__()

        # modal 1
        self.conv10 = convrelu(inputs[0], 64, 3, 1, 1)
        self.conv11 = convrelu(64, 128, 5, 2, 2)
        self.conv12 = convrelu(128, 256, 5, 2, 2)
        self.conv13 = convrelu(256, 512, 5, 2, 2)

        self.att10 = nn.Sequential(*[AOTBlock(64, [1, 2, 4, 8]) for _ in range(1)])
        self.att11 = nn.Sequential(*[AOTBlock(128, [1, 2, 4, 8]) for _ in range(1)])
        self.att12 = nn.Sequential(*[AOTBlock(256, [1, 2, 4, 8]) for _ in range(1)])
        self.att13 = nn.Sequential(*[AOTBlock(512, [1, 2, 4, 8]) for _ in range(1)])

        # modal 2
        self.conv20 = convrelu(inputs[1], 64, 3, 1, 1)
        self.conv21 = convrelu(64, 128, 5, 2, 2)
        self.conv22 = convrelu(128, 256, 5, 2, 2)
        self.conv23 = convrelu(256, 512, 5, 2, 2)

        self.att20 = Top_Attention(64, heads[0], bias)
        self.att21 = Top_Attention(128, heads[1], bias)
        self.att22 = Top_Attention(256, heads[2], bias)
        self.att23 = Top_Attention(512, heads[3], bias)

        # fusion
        self.att0 = Attention(64)
        self.att1 = Attention(128)
        self.att2 = Attention(256)

        self.up_conv00 = convreluT(512, 256, 4, 1)
        self.up_conv11 = convreluT(256 + 256, 128, 4, 1)
        self.up_conv22 = convreluT(128 + 128, 64, 4, 1)
        self.up_conv33 = convrelu(64 + 64, 1, 5, 2, 1)

    def forward(self, img, lvm_img, sample):

        x = torch.cat([img, lvm_img], 1)

        # stage 1
        layer10 = self.conv10(x)
        layer10 = self.att10(layer10)

        layer20 = self.conv20(sample)
        layer20 = self.att20(layer20)

        fusion_210 = self.att0(layer10, layer20)

        # stage 2
        layer11 = self.conv11(layer10)
        layer11 = self.att11(layer11)

        layer21 = self.conv21(layer20)
        layer21 = self.att21(layer21)

        fusion_211 = self.att1(layer11, layer21)

        # stage 3
        layer12 = self.conv12(layer11)
        layer12 = self.att12(layer12)

        layer22 = self.conv22(layer21)
        layer22 = self.att22(layer22)

        fusion_212 = self.att2(layer12, layer22)

        # stage 4
        layer13 = self.conv13(layer12)
        layer13 = self.att13(layer13)

        layer23 = self.conv23(layer22)
        layer23 = self.att23(layer23)

        # decoder
        up_layer0 = self.up_conv00(layer13 + layer23)
        cat0 = torch.cat([up_layer0, fusion_212], dim=1)
        up_layer1 = self.up_conv11(cat0)
        cat1 = torch.cat([up_layer1, fusion_211], dim=1)
        up_layer2 = self.up_conv22(cat1)
        cat2 = torch.cat([up_layer2, fusion_210], dim=1)
        up_layer3 = self.up_conv33(cat2)

        return up_layer3

# Debug
def test():
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    z = torch.randn((1, 1, 256, 256))

    model = Multi_model()
    flops, params = profile(model, (x, y, z))
    print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))

    model.cuda()
    preds = model(x.cuda(), y.cuda(), z.cuda())
    print(preds.shape)


if __name__ == "__main__":
    test()
