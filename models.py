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
    b, c, h, w = ll.shape
    _, rec_f = _get_db2_2d_filters(ll.device, ll.dtype)
    filters = rec_f.repeat(c, 1, 1, 1)
    highs = highs.view(b, c, 3, h, w)
    combined = torch.cat([ll.unsqueeze(2), highs], dim=2).view(b, -1, h, w)
    return F.conv_transpose2d(combined, filters, stride=2, padding=1, groups=c)

# --- SwinConvDenoiser Components ---

def _swin_window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C)

def _swin_window_reverse(windows, window_size, H, W):
    nW = (H // window_size) * (W // window_size)
    B = windows.shape[0] // nW
    C = windows.shape[-1]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)

class _WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        n = (2 * window_size - 1)
        self.rel_pos_bias = nn.Parameter(torch.zeros(n * n, num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing='ij'))
        coords_f = coords.flatten(1)
        rel = coords_f[:, :, None] - coords_f[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer('rel_pos_index', rel.sum(-1))
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.rel_pos_bias[self.rel_pos_index.view(-1)]
        bias = bias.view(N, N, self.num_heads).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask[None, :, None, :, :]
            attn = attn.view(B_, self.num_heads, N, N)
        attn = self.softmax(attn)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)

class _SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size, mlp_ratio=4.):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, dim))

    def _make_attn_mask(self, H, W, device):
        if self.shift_size == 0: return None
        img_mask = torch.zeros(1, H, W, 1, device=device)
        ws, ss = self.window_size, self.shift_size
        for hi, hs in enumerate([slice(0, -ws), slice(-ws, -ss), slice(-ss, None)]):
            for wi, ws_ in enumerate([slice(0, -ws), slice(-ws, -ss), slice(-ss, None)]):
                img_mask[:, hs, ws_, :] = hi * 3 + wi
        mw = _swin_window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        mask = mw.unsqueeze(1) - mw.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        x_n = self.norm1(x_nhwc)
        if self.shift_size > 0:
            x_n = torch.roll(x_n, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        wins = _swin_window_partition(x_n, self.window_size).view(-1, self.window_size ** 2, C)
        mask = self._make_attn_mask(H, W, x.device)
        attn_out = self.attn(wins, mask=mask).view(-1, self.window_size, self.window_size, C)
        attn_out = _swin_window_reverse(attn_out, self.window_size, H, W)
        if self.shift_size > 0:
            attn_out = torch.roll(attn_out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = shortcut + attn_out.permute(0, 3, 1, 2).contiguous()
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        return x + self.mlp(self.norm2(x_nhwc)).permute(0, 3, 1, 2).contiguous()

class _RSTG(nn.Module):
    def __init__(self, dim, num_heads, window_size, num_blocks=2, mlp_ratio=4.):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            shift = 0 if i % 2 == 0 else window_size // 2
            blocks.append(_SwinBlock(dim, num_heads, window_size, shift, mlp_ratio))
        self.blocks = nn.ModuleList(blocks)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
    def forward(self, x):
        res = x
        for blk in self.blocks: res = blk(res)
        return x + self.conv(res)

class SwinConvDenoiser(nn.Module):
    def __init__(self, in_ch=3, base=96, num_blocks=5):
        super().__init__()
        dim, num_heads, window_size = base, 4, 8
        self.embed = nn.Conv2d(in_ch, dim, 3, 1, 1)
        self.down1 = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        self.rstgs = nn.ModuleList([_RSTG(dim, num_heads, window_size, num_blocks=2) for _ in range(num_blocks)])
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(dim, dim, 3, 1, 1))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(dim, dim, 3, 1, 1))
        self.out = nn.Conv2d(dim, in_ch, 3, 1, 1)
        self.skip1_proj = nn.Conv2d(dim * 2, dim, 1)
        self.skip2_proj = nn.Conv2d(dim * 2, dim, 1)

    def forward(self, x):
        feat = self.embed(x)
        d1 = self.down1(feat)
        d2 = self.down2(d1)
        z = d2
        for rstg in self.rstgs: z = rstg(z)
        u1 = self.skip1_proj(torch.cat([self.up1(z), d1], dim=1))
        u2 = self.skip2_proj(torch.cat([self.up2(u1), feat], dim=1))
        return (x + self.out(u2)).clamp(0, 1)

# --- Original FASCANet ---

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
        self.fusion          = nn.Sequential(nn.Conv2d(base*2, base, 1), nn.ReLU(inplace=True), nn.Conv2d(base, base, 3, 1, 1))
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
