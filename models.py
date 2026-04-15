import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

# --- Wavelet Utilities (db2) ---
def _get_db2_2d_filters(device, dtype):
    """Generates 2D DB2 filters for DWT and IWT."""
    w = pywt.Wavelet('db2')
    lo_d = torch.tensor(w.dec_lo, dtype=dtype, device=device)
    hi_d = torch.tensor(w.dec_hi, dtype=dtype, device=device)
    lo_r = torch.tensor(w.rec_lo, dtype=dtype, device=device)
    hi_r = torch.tensor(w.rec_hi, dtype=dtype, device=device)

    def make_2d(f1, f2):
        return torch.outer(f1, f2).unsqueeze(0).unsqueeze(0)

    # Order: LL, HL, LH, HH 
    dec_filters = torch.cat([
        make_2d(lo_d, lo_d), make_2d(lo_d, hi_d),
        make_2d(hi_d, lo_d), make_2d(hi_d, hi_d)
    ], dim=0)

    rec_filters = torch.cat([
        make_2d(lo_r, lo_r), make_2d(lo_r, hi_r),
        make_2d(hi_r, lo_r), make_2d(hi_r, hi_r)
    ], dim=0)

    return dec_filters, rec_filters

def dwt_db2(x):
    """Single-level 2D db2 DWT."""
    b, c, h, w = x.shape
    dec_f, _ = _get_db2_2d_filters(x.device, x.dtype)
    filters = dec_f.repeat(c, 1, 1, 1)
    x_pad = F.pad(x, (1, 1, 1, 1), mode='reflect')
    out = F.conv2d(x_pad, filters, stride=2, groups=c)
    out = out.view(b, c, 4, h//2, w//2)
    ll = out[:, :, 0, :, :]
    highs = out[:, :, 1:, :, :].reshape(b, -1, h//2, w//2)
    return ll, highs

def iwt_db2(ll, highs):
    """Single-level 2D db2 IDWT."""
    b, c, h, w = ll.shape
    _, rec_f = _get_db2_2d_filters(ll.device, ll.dtype)
    filters = rec_f.repeat(c, 1, 1, 1)
    highs = highs.view(b, c, 3, h, w)
    combined = torch.cat([ll.unsqueeze(2), highs], dim=2).view(b, -1, h, w)
    return F.conv_transpose2d(combined, filters, stride=2, padding=1, groups=c)

# --- FASCANet Components ---

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class SpatialCrossAttn(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(8, channels // reduction)
        self.high_to_ll = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.ll_to_high = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 3, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.norm_ll   = nn.GroupNorm(8, channels)
        self.norm_high = nn.GroupNorm(8, channels)

    def forward(self, f_ll, f_high):
        gate_ll    = self.high_to_ll(f_high)
        f_ll_out   = self.norm_ll(f_ll + f_ll * gate_ll)
        gate_high  = self.ll_to_high(f_ll)
        f_high_out = self.norm_high(f_high + f_high * gate_high)
        return f_ll_out, f_high_out

class FASCANet(nn.Module):
    """
    Frequency-Aware Spatial Cross-Attention Network (FASCANet).
    """
    def __init__(self, in_ch=3, base=96, num_blocks=6):
        super().__init__()
        self.in_ch           = in_ch
        self.head_ll         = nn.Conv2d(in_ch,   base, 3,1,1)
        self.head_high       = nn.Conv2d(in_ch*3, base, 3,1,1)
        self.res_ll          = nn.ModuleList([ResBlock(base) for _ in range(num_blocks)])
        self.res_high        = nn.ModuleList([ResBlock(base) for _ in range(num_blocks)])
        
        self.cross_interval  = 2
        num_cross            = num_blocks // self.cross_interval
        self.cross           = nn.ModuleList([SpatialCrossAttn(base) for _ in range(num_cross)])

        self.fusion          = nn.Sequential(
            nn.Conv2d(base*2, base, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, 1, 1)
        )
        self.tail = nn.Conv2d(base, in_ch*4, 1)

    def forward(self, x):
        ll, highs = dwt_db2(x)
        f_ll   = self.head_ll(ll)
        f_high = self.head_high(highs)

        cross_idx = 0
        for i, (res_ll, res_high) in enumerate(zip(self.res_ll, self.res_high)):
            f_ll   = res_ll(f_ll)
            f_high = res_high(f_high)
            if (i + 1) % self.cross_interval == 0:
                f_ll, f_high = self.cross[cross_idx](f_ll, f_high)
                cross_idx += 1

        f_fuse = self.fusion(torch.cat([f_ll, f_high], dim=1))
        res    = self.tail(f_fuse)
        return iwt_db2(ll + res[:, :self.in_ch], highs + res[:, self.in_ch:])

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=64, dropout=0.0):
        super().__init__()
        def block(i,o):
            layers = [nn.Conv2d(i, o, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(o, o, 3, 1, 1), nn.ReLU(inplace=True)]
            if dropout > 0: layers.append(nn.Dropout2d(dropout))
            return nn.Sequential(*layers)
        self.e1 = block(in_ch, base); self.e2 = block(base, base*2)
        self.p = nn.MaxPool2d(2); self.b = block(base*2, base*4)
        self.u = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) 
        self.d2 = block(base*6, base*2); self.d1 = block(base*3, base)
        self.out = nn.Conv2d(base, out_ch, 1)
        
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.p(e1)); b = self.b(self.p(e2))
        d2 = self.d2(torch.cat([self.u(b), e2], dim=1))
        d1 = self.d1(torch.cat([self.u(d2), e1], dim=1))
        return self.out(d1)
