import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=64, dropout=0.0):
        super().__init__()
        self.dropout_rate = dropout
        def block(i,o):
            layers = [nn.Conv2d(i, o, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, 1, 1), nn.ReLU(inplace=True)]
            if dropout > 0: layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)
        
        self.e1 = block(in_ch, base)
        self.e2 = block(base, base*2)
        self.p = nn.MaxPool2d(2)
        self.b = block(base*2, base*4)
        self.u = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) 
        self.d2 = block(base*6, base*2)
        self.d1 = block(base*3, base)
        self.out = nn.Conv2d(base, out_ch, 1)
        
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p(e1))
        b = self.b(self.p(e2))
        d2 = self.d2(torch.cat([self.u(b), e2], dim=1))
        d1 = self.d1(torch.cat([self.u(d2), e1], dim=1))
        return self.out(d1)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class CA_Block(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CA_Block, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, self.mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(self.mip)
        self.act = nn.Hardswish()
        self.conv_h = nn.Conv2d(self.mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(self.mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out

class AttentiveResBlock(nn.Module):
    def __init__(self, channels):
        super(AttentiveResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.attention = CA_Block(channels)
    def forward(self, x):
        residual = self.conv2(self.relu(self.conv1(x)))
        residual = self.attention(residual)
        return x + residual

def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2; x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]; x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]; x4 = x02[:, :, :, 1::2]
    ll = x1 + x2 + x3 + x4
    hl = -x1 - x2 + x3 + x4; lh = -x1 + x2 - x3 + x4; hh = x1 - x2 - x3 + x4
    return ll, torch.cat((hl, lh, hh), 1)

def iwt_init(ll, highs):
    hl, lh, hh = torch.chunk(highs, 3, dim=1)
    x1 = (ll - hl - lh + hh) / 2.0
    x2 = (ll - hl + lh - hh) / 2.0
    x3 = (ll + hl - lh - hh) / 2.0
    x4 = (ll + hl + lh + hh) / 2.0
    b,c,h,w = ll.shape
    out = torch.zeros((b,c,h*2,w*2), device=ll.device, dtype=ll.dtype)
    out[:,:,0::2,0::2] = x1; out[:,:,1::2,0::2] = x2
    out[:,:,0::2,1::2] = x3; out[:,:,1::2,1::2] = x4
    return out

class WRTPNet_Baseline(nn.Module):
    def __init__(self, in_ch=3, base=96, num_blocks=6):
        super().__init__()
        self.head_ll = nn.Conv2d(in_ch, base, 3, 1, 1)
        self.head_high = nn.Conv2d(in_ch*3, base, 3, 1, 1)
        self.body_ll = nn.Sequential(*[ResBlock(base) for _ in range(num_blocks)])
        self.body_high = nn.Sequential(*[ResBlock(base) for _ in range(num_blocks)])
        self.fusion = nn.Sequential(nn.Conv2d(base*2, base, 1), nn.ReLU(inplace=True), nn.Conv2d(base, base, 3, 1, 1))
        self.tail = nn.Conv2d(base, in_ch*4, 1)
    def forward(self, x):
        ll, highs = dwt_init(x)
        f_ll = self.head_ll(ll); f_high = self.head_high(highs)
        f_ll = self.body_ll(f_ll); f_high = self.body_high(f_high)
        f_cat = torch.cat([f_ll, f_high], dim=1)
        f_fuse = self.fusion(f_cat)
        res = self.tail(f_fuse)
        return iwt_init(ll + res[:, 0:3], highs + res[:, 3:])

class WRTPNet_Attentive(nn.Module):
    def __init__(self, in_ch=3, base=96, num_blocks=6):
        super().__init__()
        self.head_ll = nn.Conv2d(in_ch, base, 3, 1, 1)
        self.head_high = nn.Conv2d(in_ch*3, base, 3, 1, 1)
        self.body_ll = nn.Sequential(*[AttentiveResBlock(base) for _ in range(num_blocks)])
        self.body_high = nn.Sequential(*[AttentiveResBlock(base) for _ in range(num_blocks)])
        self.fusion = nn.Sequential(nn.Conv2d(base*2, base, 1), nn.ReLU(inplace=True), nn.Conv2d(base, base, 3, 1, 1))
        self.tail = nn.Conv2d(base, in_ch*4, 1)
    def forward(self, x):
        ll, highs = dwt_init(x)
        f_ll = self.head_ll(ll); f_high = self.head_high(highs)
        f_ll = self.body_ll(f_ll); f_high = self.body_high(f_high)
        f_cat = torch.cat([f_ll, f_high], dim=1)
        f_fuse = self.fusion(f_cat)
        res = self.tail(f_fuse)
        return iwt_init(ll + res[:, 0:3], highs + res[:, 3:])
class WRTPNet_AdaptiveMask(nn.Module):
    def __init__(self, in_ch=3, base=96, num_blocks=6):
        super().__init__()
        self.head_ll = nn.Conv2d(in_ch, base, 3, 1, 1)
        self.head_high = nn.Conv2d(in_ch*3, base, 3, 1, 1)
        self.body_ll = nn.Sequential(*[ResBlock(base) for _ in range(num_blocks)])
        self.body_high = nn.Sequential(*[ResBlock(base) for _ in range(num_blocks)])
        self.fusion = nn.Sequential(nn.Conv2d(base*2, base, 1), nn.ReLU(inplace=True), nn.Conv2d(base, base, 3, 1, 1))
        self.tail = nn.Conv2d(base, in_ch*4, 1)
    def forward(self, x):
        ll, highs = dwt_init(x)
        f_ll = self.head_ll(ll); f_high = self.head_high(highs)
        f_ll = self.body_ll(f_ll); f_high = self.body_high(f_high)
        f_cat = torch.cat([f_ll, f_high], dim=1)
        f_fuse = self.fusion(f_cat)
        res = self.tail(f_fuse)
        return iwt_init(ll + res[:, 0:3], highs + res[:, 3:])
