import os
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from scipy.ndimage import distance_transform_edt
import gc
import pandas as pd
import shutil

class Config:
    INPUT_DIR    = "input_images" # Update this to your local path
    OUTPUT_DIR   = "output_results"
    IMG_SIZE     = 256
    BATCH_SIZE   = 8
    EPOCHS       = 50
    LEARNING_RATE = 1e-4
    MASK_PERCENT = 0.03
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS  = 4 if torch.cuda.is_available() else 0
    SEEDS        = [42, 100, 123]
    SIGMA        = 0.1
    POISSON_PEAK = 30.0
    SIGMA_READ   = 0.02
    DIP_ITERS    = 1000
    DIP_LR       = 0.001
    USE_AMP      = True
    AMP_DTYPE    = torch.float16
    AMP_DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"
    AGGREGATED_DIR = os.path.join(OUTPUT_DIR, "aggregated")

os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.AGGREGATED_DIR, exist_ok=True)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.empty_cache()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class NoiseInjector:
    @staticmethod
    def add_gaussian(img, sigma=Config.SIGMA):
        return (img + torch.randn_like(img) * sigma).clamp(0, 1)
    @staticmethod
    def add_poisson(img, peak=Config.POISSON_PEAK):
        return (torch.poisson((img * peak).float()) / float(peak)).clamp(0, 1)
    @staticmethod
    def add_poisson_gaussian(img, peak=Config.POISSON_PEAK, sigma_read=Config.SIGMA_READ):
        p = NoiseInjector.add_poisson(img, peak=peak)
        return NoiseInjector.add_gaussian(p, sigma=sigma_read)

def inject_noise_by_type(clean, noise_type):
    if noise_type == "Gaussian":
        return NoiseInjector.add_gaussian(clean, sigma=Config.SIGMA)
    elif noise_type == "Poisson":
        return NoiseInjector.add_poisson(clean, peak=Config.POISSON_PEAK)
    else:
        return NoiseInjector.add_poisson_gaussian(clean, peak=Config.POISSON_PEAK, sigma_read=Config.SIGMA_READ)

def mask_n2v(img, prob):
    mask = torch.rand_like(img) < prob
    masked_img = img.clone()
    masked_img[mask] = torch.rand_like(img)[mask]
    return masked_img, mask

def neighbor2neighbor_augment(x):
    v1 = x[:, :, 0::2, 0::2]
    v2 = x[:, :, 1::2, 1::2]
    return v1, v2

# --- Advanced Metrics Implementation ---

def calc_msssim(img1_tensor, img2_tensor, window_size=11, val_range=1.0):
    """PyTorch implementation of MS-SSIM for robust benchmarking."""
    if img1_tensor.ndim == 3: img1_tensor = img1_tensor.unsqueeze(0)
    if img2_tensor.ndim == 3: img2_tensor = img2_tensor.unsqueeze(0)
    pad = window_size // 2
    
    def _gaussian(w_size, sigma):
        gauss = torch.Tensor([np.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
        return gauss/gauss.sum()
        
    def _create_window(w_size, channel):
        _1D = _gaussian(w_size, 1.5).unsqueeze(1)
        _2D = _1D.mm(_1D.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D.expand(channel, 1, w_size, w_size).to(img1_tensor.device)
        
    window = _create_window(window_size, img1_tensor.shape[1])
    
    def _ssim(img1, img2, window, val_range):
        mu1 = F.conv2d(img1, window, padding=pad, groups=img1.shape[1])
        mu2 = F.conv2d(img2, window, padding=pad, groups=img1.shape[1])
        mu1_sq = mu1.pow(2); mu2_sq = mu2.pow(2); mu1_mu2 = mu1*mu2
        sigma1_sq = F.conv2d(img1*img1, window, padding=pad, groups=img1.shape[1]) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=pad, groups=img1.shape[1]) - mu2_sq
        sigma12   = F.conv2d(img1*img2, window, padding=pad, groups=img1.shape[1]) - mu1_mu2
        C1 = (0.01*val_range)**2; C2 = (0.03*val_range)**2
        luminance = (2*mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
        contrast  = (2*sigma12 + C2)  / (sigma1_sq + sigma2_sq + C2)
        return luminance, contrast

    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(img1_tensor.device)
    levels = weights.shape[0]
    mcs = []
    for i in range(levels):
        lum, con = _ssim(img1_tensor, img2_tensor, window, val_range)
        mcs.append(torch.relu(con).mean(dim=(1,2,3)))
        if i < levels - 1:
            img1_tensor = F.avg_pool2d(img1_tensor, (2,2))
            img2_tensor = F.avg_pool2d(img2_tensor, (2,2))
    mcs = torch.stack(mcs)
    return torch.prod(mcs ** weights.view(-1,1), dim=0).mean().item()

def calc_fsim(clean, pred):
    Y_c = (0.299*clean[:,:,0] + 0.587*clean[:,:,1] + 0.114*clean[:,:,2]).astype(np.float32)
    Y_p = (0.299*pred[:,:,0]  + 0.587*pred[:,:,1]  + 0.114*pred[:,:,2]).astype(np.float32)
    sx_c = cv2.Scharr(Y_c, cv2.CV_32F, 1, 0); sy_c = cv2.Scharr(Y_c, cv2.CV_32F, 0, 1)
    sx_p = cv2.Scharr(Y_p, cv2.CV_32F, 1, 0); sy_p = cv2.Scharr(Y_p, cv2.CV_32F, 0, 1)
    GM_c = np.sqrt(sx_c**2 + sy_c**2); GM_p = np.sqrt(sx_p**2 + sy_p**2)
    C1=0.0026; S_GM = (2*GM_c*GM_p + C1) / (GM_c**2 + GM_p**2 + C1)
    Wm = np.maximum(GM_c, GM_p)
    return np.sum(S_GM * Wm) / (np.sum(Wm) + 1e-8)

def calc_vif(clean, pred):
    mu1 = cv2.GaussianBlur(clean, (5,5), 1.5)
    mu2 = cv2.GaussianBlur(pred, (5,5), 1.5)
    sigma1_sq = cv2.GaussianBlur(clean*clean, (5,5), 1.5) - mu1**2
    sigma12 = cv2.GaussianBlur(clean*pred, (5,5), 1.5) - mu1*mu2
    return np.mean(sigma12 / (sigma1_sq + 0.1))

def calc_fom(clean, pred):
    c_gray = cv2.cvtColor((clean*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    p_gray = cv2.cvtColor((pred*255).astype(np.uint8),  cv2.COLOR_RGB2GRAY)
    edges_c = cv2.Canny(c_gray,50,150); edges_p = cv2.Canny(p_gray,50,150)
    if np.sum(edges_c)==0: return 1.0
    dist_map = distance_transform_edt(255 - edges_c)
    N_c = np.sum(edges_c>0); N_p = np.sum(edges_p>0)
    if max(N_c,N_p)==0: return 0.0
    alpha = 1.0/9.0
    return np.sum(1.0/(1.0+alpha*(dist_map[edges_p>0]**2))) / max(N_c,N_p)

def compute_metrics(clean, pred):
    clean = np.clip(clean, 0, 1); pred = np.clip(pred, 0, 1)
    p = psnr_metric(clean, pred, data_range=1.0)
    s = ssim_metric(clean, pred, data_range=1.0, channel_axis=2)
    f = calc_fsim(clean, pred)
    v = calc_vif(clean, pred)
    
    # Use the robust MS-SSIM
    t_c = torch.from_numpy(clean.transpose(2,0,1)).float().unsqueeze(0).to(Config.DEVICE)
    t_p = torch.from_numpy(pred.transpose(2,0,1)).float().unsqueeze(0).to(Config.DEVICE)
    m = calc_msssim(t_c, t_p)
    
    fo = calc_fom(clean, pred)
    return p, s, f, v, m, fo

def register_result_with_noisy(res_dict, name, idx, c, n, ns, p_img, p, s, f, v, m, fo, save_dir):
    if name not in res_dict:
        res_dict[name] = {'psnr':[], 'ssim':[], 'fsim':[], 'vif':[], 'msssim':[], 'fom':[], 
                          'psnr_vs_noisy':[], 'ssim_vs_noisy':[], 'fsim_vs_noisy':[], 
                          'vif_vs_noisy':[], 'msssim_vs_noisy':[], 'fom_vs_noisy':[]}
    
    res_dict[name]['psnr'].append(p)
    res_dict[name]['ssim'].append(s)
    res_dict[name]['fsim'].append(f)
    res_dict[name]['vif'].append(v)
    res_dict[name]['msssim'].append(m)
    res_dict[name]['fom'].append(fo)

    # Comparison vs Raw Noisy
    p_n, s_n, f_n, v_n, m_n, fo_n = compute_metrics(n, p_img)
    res_dict[name]['psnr_vs_noisy'].append(p_n)
    res_dict[name]['ssim_vs_noisy'].append(s_n)
    res_dict[name]['fsim_vs_noisy'].append(f_n)
    res_dict[name]['vif_vs_noisy'].append(v_n)
    res_dict[name]['msssim_vs_noisy'].append(m_n)
    res_dict[name]['fom_vs_noisy'].append(fo_n)

    if idx < 5: 
        target_path = os.path.join(save_dir, name)
        os.makedirs(target_path, exist_ok=True)
        cv2.imwrite(os.path.join(target_path, f"sample_{idx}.png"), (p_img*255).astype(np.uint8)[:,:,::-1])

def geometric_ensemble_inference(model, x):
    model.eval()
    with torch.no_grad(): 
        return model(x).clamp(0,1)

def zip_results():
    shutil.make_archive("results", 'zip', Config.OUTPUT_DIR)

class BenchmarkDataset(torch.utils.data.Dataset):
    def __init__(self, dir):
        self.files = sorted([os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(('.png', '.jpg', '.tif'))])
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        img = cv2.imread(self.files[i]); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        return torch.from_numpy(img.transpose(2,0,1)).float()/255.0
