import os
import glob
import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from scipy.ndimage import distance_transform_edt
import gc
import platform
from PIL import Image, ImageDraw, ImageFont
import shutil
import warnings

warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
except (ImportError, AttributeError):
    def tqdm(iterable, *args, **kwargs):
        return iterable

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    INPUT_DIR = #dataset folder input path
    OUTPUT_DIR = #results saving path
    IMG_SIZE = 256
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    MASK_PERCENT = 0.03  
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0
    PIN_MEMORY = True if torch.cuda.is_available() else False
    NUM_PLOT_SAMPLES = 15
    SEEDS = [42, 100, 123]
    PIXEL_SIZE_UM = 1.0
    
    # Noise parameters
    SIGMA = 0.0980
    POISSON_PEAK = 30.0
    SIGMA_READ = 0.02
    
    AGGREGATED_DIR = os.path.join(OUTPUT_DIR, "aggregated")
    ZIP_DIR = os.path.join(OUTPUT_DIR, "results_zipped")
    
    USE_AMP = True
    AMP_DTYPE = torch.float16
    AMP_DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

    # DIP settings
    DIP_ITERS = 800
    DIP_LR = 1e-3
    DIP_INPUT_TYPE = "random"  # 'random' or 'meshgrid'

# Ensure directories exist
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.AGGREGATED_DIR, exist_ok=True)
os.makedirs(Config.ZIP_DIR, exist_ok=True)

# ==========================================
# SYSTEM HELPERS
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def zip_results():
    """
    Compresses the entire Config.OUTPUT_DIR into a single zip file.
    """
    source_dir = Config.OUTPUT_DIR
    output_filename = "ablation_study_results" 
    
    if not os.path.exists(source_dir):
        print(f"CRITICAL: Output directory '{source_dir}' does not exist. Nothing to zip.")
        return

    print(f"Starting compression of '{source_dir}'...")
    try:
        shutil.make_archive(output_filename, 'zip', source_dir)
        full_path = os.path.abspath(output_filename + '.zip')
        file_size_mb = os.path.getsize(full_path) / (1024 * 1024)
        print(f"SUCCESS: Results zipped successfully.")
        print(f"Saved to: {full_path}")
        print(f"Size: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"ERROR: Failed to zip results. Reason: {e}")

# ==========================================
# VISUALIZATION HELPERS
# ==========================================
def get_optimal_font_path():
    """Finds a font file supporting Unicode/scientific symbols."""
    try:
        from matplotlib import font_manager
        preferences = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'Bitstream Vera Sans']
        for family in preferences:
            try:
                font_path = font_manager.findfont(font_manager.FontProperties(family=family))
                if os.path.exists(font_path): return font_path
            except Exception: continue
        font_path = font_manager.findfont(font_manager.FontProperties(family='sans-serif'))
        if os.path.exists(font_path): return font_path
    except ImportError: pass

    system = platform.system()
    search_paths = []
    if system == "Windows":
        search_paths = ["C:\\Windows\\Fonts\\arial.ttf", "C:\\Windows\\Fonts\\calibri.ttf", "C:\\Windows\\Fonts\\seguiemj.ttf"]
    elif system == "Darwin":
        search_paths = ["/Library/Fonts/Arial.ttf", "/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/SFNS.ttf"]
    else:
        search_paths = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", "/usr/share/fonts/truetype/freefont/FreeSans.ttf"]
        
    for path in search_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path): return expanded_path
    return None

def add_scale_bar(img_np, pixel_size_um=Config.PIXEL_SIZE_UM):
    """Adds a scale bar to an image (robust Unicode 'µ' support)."""
    if img_np.dtype != np.uint8:
        vis = (np.clip(img_np, 0, 1) * 255).astype(np.uint8).copy()
    else:
        vis = img_np.copy()
    h, w = vis.shape[:2]
    bar_len_px = min(50, w // 3)
    bar_len_um = bar_len_px * pixel_size_um
    margin = 10 
    start_x = w - bar_len_px - margin
    start_point = (int(start_x), int(h - margin))
    end_point = (int(start_x + bar_len_px), int(h - margin))
    cv2.line(vis, start_point, end_point, (255, 255, 255), 2)
    
    pil_img = Image.fromarray(vis)
    draw = ImageDraw.Draw(pil_img)
    font_path = get_optimal_font_path()
    font_size = 11 
    font = None
    font_found = False
    
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
            font_found = True
        except OSError: font = None
    if font is None:
        font = ImageFont.load_default()
        font_found = False

    if font_found: label = f"{bar_len_um:.0f} \u00B5m"
    else: label = f"{bar_len_um:.0f} um"

    try:
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w = right - left
        text_h = bottom - top
    except AttributeError:
        text_w, text_h = draw.textsize(label, font=font)

    text_x = start_x + (bar_len_px // 2) - (text_w // 2)
    text_y = h - margin - text_h - 2
    draw.text((text_x, text_y), label, font=font, fill=(255, 255, 255))
    final_vis = np.array(pil_img)
    return final_vis.astype(np.float32) / 255.0

# ==========================================
# METRICS
# ==========================================
def calc_msssim(img1_tensor, img2_tensor, window_size=11, val_range=1.0):
    if img1_tensor.ndim == 3: img1_tensor = img1_tensor.unsqueeze(0)
    if img2_tensor.ndim == 3: img2_tensor = img2_tensor.unsqueeze(0)
    pad = window_size // 2
    def _gaussian(w_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
        return gauss/gauss.sum()
    def _create_window(w_size, channel):
        _1D_window = _gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channel, 1, w_size, w_size).to(img1_tensor.device)
    window = _create_window(window_size, img1_tensor.shape[1])
    def _ssim(img1, img2, window, val_range):
        mu1 = F.conv2d(img1, window, padding=pad, groups=img1.shape[1])
        mu2 = F.conv2d(img2, window, padding=pad, groups=img1.shape[1])
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2
        sigma1_sq = F.conv2d(img1*img1, window, padding=pad, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=pad, groups=img1.shape[1]) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=pad, groups=img1.shape[1]) - mu1_mu2
        C1 = (0.01 * val_range) ** 2
        C2 = (0.03 * val_range) ** 2
        luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        contrast = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        return luminance, contrast
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(img1_tensor.device)
    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        lum, con = _ssim(img1_tensor, img2_tensor, window, val_range)
        if i < levels - 1:
            mcs.append(torch.relu(con).mean(dim=(1, 2, 3)))
            img1_tensor = F.avg_pool2d(img1_tensor, (2, 2))
            img2_tensor = F.avg_pool2d(img2_tensor, (2, 2))
        else:
            mcs.append(torch.relu(con).mean(dim=(1, 2, 3)))
    mcs = torch.stack(mcs)
    pow1 = mcs ** weights.view(-1, 1)
    return torch.prod(pow1, dim=0).mean().item()

def calc_fsim(clean, pred):
    Y_c = 0.299 * clean[:,:,0] + 0.587 * clean[:,:,1] + 0.114 * clean[:,:,2]
    Y_p = 0.299 * pred[:,:,0] + 0.587 * pred[:,:,1] + 0.114 * pred[:,:,2]
    Y_c = Y_c.astype(np.float32); Y_p = Y_p.astype(np.float32)
    scharrx_c = cv2.Scharr(Y_c, cv2.CV_32F, 1, 0); scharry_c = cv2.Scharr(Y_c, cv2.CV_32F, 0, 1)
    GM_c = np.sqrt(scharrx_c**2 + scharry_c**2)
    scharrx_p = cv2.Scharr(Y_p, cv2.CV_32F, 1, 0); scharry_p = cv2.Scharr(Y_p, cv2.CV_32F, 0, 1)
    GM_p = np.sqrt(scharrx_p**2 + scharry_p**2)
    C1 = 0.0026; C2 = 0.0026
    S_GM = (2 * GM_c * GM_p + C1) / (GM_c**2 + GM_p**2 + C1)
    S_L = (2 * Y_c * Y_p + C2) / (Y_c**2 + Y_p**2 + C2)
    Wm = np.maximum(GM_c, GM_p)
    return np.sum(S_GM * S_L * Wm) / (np.sum(Wm) + 1e-8)

def calc_vif(clean, pred):
    clean = clean.astype(np.float32); pred = pred.astype(np.float32)
    sigma_nsq = 2.0; num = 0.0; den = 0.0
    for scale in range(1, 5):
        N = 2**(4-scale+1) + 1
        mu1 = cv2.GaussianBlur(clean, (N, N), N/3.0); mu2 = cv2.GaussianBlur(pred, (N, N), N/3.0)
        mu1_sq = mu1 * mu1; mu2_sq = mu2 * mu2; mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.GaussianBlur(clean * clean, (N, N), N/3.0) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(pred * pred, (N, N), N/3.0) - mu2_sq
        sigma12 = cv2.GaussianBlur(clean * pred, (N, N), N/3.0) - mu1_mu2
        sigma1_sq[sigma1_sq < 0] = 0; sigma2_sq[sigma2_sq < 0] = 0
        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g * sigma12
        g[sigma1_sq < 1e-10] = 0; sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
        sigma1_sq[sigma1_sq < 1e-10] = 0; g[sigma12 < 0] = 0; sv_sq[sigma12 < 0] = sigma2_sq[sigma12 < 0]
        sv_sq[sv_sq < 1e-10] = 1e-10
        vif_val = np.log10(1 + g**2 * sigma1_sq / (sv_sq + sigma_nsq))
        vif_ref = np.log10(1 + sigma1_sq / sigma_nsq)
        num += np.sum(vif_val); den += np.sum(vif_ref)
        if scale < 4:
            clean = cv2.resize(clean, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            pred = cv2.resize(pred, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    return num / (den + 1e-8)

def calc_fom(clean, pred):
    c_gray = cv2.cvtColor((clean*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    p_gray = cv2.cvtColor((pred*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges_c = cv2.Canny(c_gray, 50, 150); edges_p = cv2.Canny(p_gray, 50, 150)
    if np.sum(edges_c) == 0: return 1.0
    dist_map = distance_transform_edt(255 - edges_c)
    N_c = np.sum(edges_c > 0); N_p = np.sum(edges_p > 0)
    if max(N_c, N_p) == 0: return 0.0
    alpha = 1.0 / 9.0
    error_sum = np.sum(1.0 / (1.0 + alpha * (dist_map[edges_p > 0] ** 2)))
    return error_sum / max(N_c, N_p)

def compute_metrics(clean, pred):
    pred = np.clip(pred, 0, 1); clean = np.clip(clean, 0, 1)
    p = psnr_metric(clean, pred, data_range=1.0)
    s = ssim_metric(clean, pred, data_range=1.0, channel_axis=2)
    fsim_val = calc_fsim(clean, pred)
    vif_val = calc_vif(clean, pred)
    try:
        t_clean = torch.from_numpy(clean.transpose(2,0,1)).float().unsqueeze(0).to(Config.DEVICE)
        t_pred = torch.from_numpy(pred.transpose(2,0,1)).float().unsqueeze(0).to(Config.DEVICE)
        msssim_val = calc_msssim(t_clean, t_pred)
    except Exception:
        msssim_val = np.nan
    fom_val = calc_fom(clean, pred)
    return p, s, fsim_val, vif_val, msssim_val, fom_val

def register_result_with_noisy(results_dict, method_name, idx, clean_np, noisy_np, noisy_synthetic_np, pred_np, psnr, ssim, fsim, vif, msssim, fom, save_dir=None):
    if method_name not in results_dict:
        results_dict[method_name] = {
            'psnr': [], 'ssim': [], 'fsim': [], 'vif': [], 'msssim': [], 'fom': [],
            'psnr_vs_noisy': [], 'ssim_vs_noisy': [], 'fsim_vs_noisy': [], 'vif_vs_noisy': [], 'msssim_vs_noisy': [], 'fom_vs_noisy': [],
            'imgs': {}
        }
        
    results_dict[method_name]['psnr'].append(psnr)
    results_dict[method_name]['ssim'].append(ssim)
    results_dict[method_name]['fsim'].append(fsim)
    results_dict[method_name]['vif'].append(vif)
    results_dict[method_name]['msssim'].append(msssim)
    results_dict[method_name]['fom'].append(fom)

    p_n, s_n, f_n, v_n, m_n, fo_n = compute_metrics(noisy_np, pred_np)
    results_dict[method_name]['psnr_vs_noisy'].append(p_n)
    results_dict[method_name]['ssim_vs_noisy'].append(s_n)
    results_dict[method_name]['fsim_vs_noisy'].append(f_n)
    results_dict[method_name]['vif_vs_noisy'].append(v_n)
    results_dict[method_name]['msssim_vs_noisy'].append(m_n)
    results_dict[method_name]['fom_vs_noisy'].append(fo_n)
    
    if idx < Config.NUM_PLOT_SAMPLES:
        clean_bar = add_scale_bar(clean_np)
        noisy_bar = add_scale_bar(noisy_np)
        synth_bar = add_scale_bar(noisy_synthetic_np)
        pred_bar = add_scale_bar(pred_np)
        results_dict[method_name]['imgs'][idx] = pred_bar 
        
        try:
            target_dir = os.path.join(save_dir, method_name) if save_dir else Config.OUTPUT_DIR
            os.makedirs(target_dir, exist_ok=True)
            cv2.imwrite(os.path.join(target_dir, f"img{idx}_target_raw_noisy.png"), (noisy_bar * 255).astype(np.uint8)[:,:,::-1])
            cv2.imwrite(os.path.join(target_dir, f"img{idx}_input_synthetic_noisy.png"), (synth_bar * 255).astype(np.uint8)[:,:,::-1])
            cv2.imwrite(os.path.join(target_dir, f"img{idx}_denoised.png"), (pred_bar * 255).astype(np.uint8)[:,:,::-1])
            cv2.imwrite(os.path.join(target_dir, f"img{idx}_reference_clean.png"), (clean_bar * 255).astype(np.uint8)[:,:,::-1])
        except Exception: pass

# ==========================================
# DATA & NOISE
# ==========================================
class BenchmarkDataset(Dataset):
    def __init__(self, dir_path):
        exts = ("*.png", "*.jpg", "*.bmp", "*.tif", "*.tiff")
        files = []
        for e in exts: 
            files += glob.glob(os.path.join(dir_path, "**", e), recursive=True)
        self.files = sorted(files)
        if len(self.files) == 0: 
            print(f"WARNING: No images found in {dir_path}")
    def __len__(self): 
        return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        img = cv2.imread(path)
        if img is None: 
            return torch.zeros((3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=torch.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (Config.IMG_SIZE, Config.IMG_SIZE):
            img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        return torch.from_numpy(img.transpose(2,0,1)).float()

class NoiseInjector:
    @staticmethod
    def add_gaussian(img, sigma=Config.SIGMA):
        noise = torch.randn_like(img, device=img.device) * sigma
        return (img + noise).clamp(0,1)
    @staticmethod
    def add_poisson(img, peak=Config.POISSON_PEAK):
        vals = torch.poisson((img * peak).float()) / float(peak)
        return vals.clamp(0,1)
    @staticmethod
    def add_poisson_gaussian(img, peak=Config.POISSON_PEAK, sigma_read=Config.SIGMA_READ):
        p = NoiseInjector.add_poisson(img, peak=peak)
        g = NoiseInjector.add_gaussian(p, sigma=sigma_read)
        return g

def inject_noise_by_type(img, noise_type):
    if noise_type == "Gaussian":
        return NoiseInjector.add_gaussian(img, sigma=Config.SIGMA)
    elif noise_type == "Poisson":
        return NoiseInjector.add_poisson(img, peak=Config.POISSON_PEAK)
    elif noise_type == "PoissonGauss":
        return NoiseInjector.add_poisson_gaussian(img, peak=Config.POISSON_PEAK, sigma_read=Config.SIGMA_READ)
    return img

class BlindSpotMasker:
    @staticmethod
    def get_mask(img, mode='standard', percent=Config.MASK_PERCENT):
        b, c, h, w = img.shape
        if mode == 'adaptive':
            sobelx = torch.abs(F.conv2d(img, torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1,1,3,3).repeat(c,1,1,1).to(img.device), padding=1, groups=c))
            sobely = torch.abs(F.conv2d(img, torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1,1,3,3).repeat(c,1,1,1).to(img.device), padding=1, groups=c))
            grad = sobelx + sobely
            min_val = grad.view(b, -1).min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
            max_val = grad.view(b, -1).max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
            prob_map = (grad - min_val) / (max_val - min_val + 1e-8)
            prob_map = prob_map * (percent * 2.0) + (percent * 0.5) 
            prob_map = torch.clamp(prob_map, 0, 1)
            mask = torch.bernoulli(prob_map).bool()
        else: 
            mask = torch.rand(b, c, h, w, device=img.device) < percent
        return mask

def mask_n2v(img, mask_ratio):
    mask = BlindSpotMasker.get_mask(img, mode='standard', percent=mask_ratio)
    b,c,h,w = img.shape
    offs = [(0,1),(0,-1),(1,0),(-1,0)]
    rep = img.clone()
    for dx,dy in random.sample(offs, k=4):
        shifted = torch.roll(img, shifts=(dx, dy), dims=(2,3))
        rep = torch.where(mask, shifted, rep)
    return rep, mask

def mask_adaptive(img, mask_ratio):
    mask = BlindSpotMasker.get_mask(img, mode='adaptive', percent=mask_ratio)
    b,c,h,w = img.shape
    offs = [(0,1),(0,-1),(1,0),(-1,0)]
    rep = img.clone()
    for dx,dy in random.sample(offs, k=4):
        shifted = torch.roll(img, shifts=(dx, dy), dims=(2,3))
        rep = torch.where(mask, shifted, rep)
    return rep, mask

def neighbor2neighbor_augment(img):
    b, c, h, w = img.shape
    idx1, idx2 = np.random.choice(4, 2, replace=False)
    def get_subsample(idx):
        if idx == 0: return img[:, :, 0::2, 0::2]
        elif idx == 1: return img[:, :, 0::2, 1::2]
        elif idx == 2: return img[:, :, 1::2, 0::2]
        elif idx == 3: return img[:, :, 1::2, 1::2]
    return get_subsample(idx1), get_subsample(idx2)
