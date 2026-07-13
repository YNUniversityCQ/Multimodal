import torch
import numbers
import torch.nn as nn
from thop import profile
from einops import rearrange
from attention import joint_attention, AOTBlock, Attention, Top_Attention

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

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

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)

        x = self.project_out(x)

        return x

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Top_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x

class Multi_model(nn.Module):

    def __init__(self,
                 num_blocks=(4, 6, 6, 8),  # 原始(4, 6, 6, 8)
                 heads=(1, 2, 4, 8),
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 inputs=(3, 1)):

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

        self.att20 = joint_attention(64, 3)
        self.att21 = joint_attention(128, 3)
        self.att22 = joint_attention(256, 3)
        self.att23 = joint_attention(512, 3)

        # self.att20 = nn.Sequential(*[
        #     TransformerBlock(dim=64, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
        #                      LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        # self.att21 = nn.Sequential(*[
        #     TransformerBlock(dim=128, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
        #                      LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        # self.att22 = nn.Sequential(*[
        #     TransformerBlock(dim=256, num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
        #                      LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        # self.att23 = nn.Sequential(*[
        #     TransformerBlock(dim=512, num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
        #                      LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        # self.att20 = Top_Attention(64, heads[0], bias)
        # self.att21 = Top_Attention(128, heads[1], bias)
        # self.att22 = Top_Attention(256, heads[2], bias)
        # self.att23 = Top_Attention(512, heads[3], bias)

        # fusion
        self.att0 = Attention(64)
        self.att1 = Attention(128)
        self.att2 = Attention(256)

        self.up_conv00 = convreluT(512, 256, 4, 1)
        self.up_conv11 = convreluT(256 + 256, 128, 4, 1)
        self.up_conv22 = convreluT(128 + 128, 64, 4, 1)
        self.up_conv33 = convrelu(64 + 64, 1, 5, 2, 1)

    def forward(self, img, lvm_img, sample):

        # x = torch.cat([img, lvm_img], 1)

        # stage 1
        layer10 = self.conv10(img)
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


        # # single modal
        # x = torch.cat([img, lvm_img, sample], 1)
        #
        # # stage 1
        # layer10 = self.conv10(x)
        # layer10 = self.att10(layer10)
        #
        # # layer20 = self.conv20(sample)
        # # layer20 = self.att20(layer20)
        #
        # # stage 2
        # layer11 = self.conv11(layer10)
        # layer11 = self.att11(layer11)
        #
        # # layer21 = self.conv21(layer20)
        # # layer21 = self.att21(layer21)
        #
        # # stage 3
        # layer12 = self.conv12(layer11)
        # layer12 = self.att12(layer12)
        #
        # # layer22 = self.conv22(layer21)
        # # layer22 = self.att22(layer22)
        #
        # # stage 4
        # layer13 = self.conv13(layer12)
        # layer13 = self.att13(layer13)
        #
        # # layer23 = self.conv23(layer22)
        # # layer23 = self.att23(layer23)
        #
        # # decoder
        # up_layer0 = self.up_conv00(layer13)
        # # up_layer0 = self.up_conv00(layer23)
        #
        # cat0 = torch.cat([up_layer0, layer12], dim=1)
        # # cat0 = torch.cat([up_layer0, layer22], dim=1)
        #
        # up_layer1 = self.up_conv11(cat0)
        # cat1 = torch.cat([up_layer1, layer11], dim=1)
        # # cat1 = torch.cat([up_layer1, layer21], dim=1)
        #
        # up_layer2 = self.up_conv22(cat1)
        # cat2 = torch.cat([up_layer2, layer10], dim=1)
        # # cat2 = torch.cat([up_layer2, layer20], dim=1)
        #
        # up_layer3 = self.up_conv33(cat2)

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
