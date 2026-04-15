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
    INPUT_DIR    = #images input folder path
    OUTPUT_DIR   = #images output saving folder path
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
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def cleanup():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class NoiseInjector:
    @staticmethod
    def add_gaussian(img, sigma=Config.SIGMA):
        return (img + torch.randn_like(img)*sigma).clamp(0,1)
    @staticmethod
    def add_poisson(img, peak=Config.POISSON_PEAK):
        return (torch.poisson((img*peak).float()) / float(peak)).clamp(0,1)
    @staticmethod
    def add_poisson_gaussian(img, peak=Config.POISSON_PEAK, sigma_read=Config.SIGMA_READ):
        p = NoiseInjector.add_poisson(img, peak=peak)
        return NoiseInjector.add_gaussian(p, sigma=sigma_read)

def mask_n2v(img, prob):
    mask = torch.rand_like(img) < prob
    masked_img = img.clone()
    masked_img[mask] = torch.rand_like(img)[mask]
    return masked_img, mask

def neighbor2neighbor_augment(x):
    # Standard N2N subsampling strategy
    v1 = x[:, :, 0::2, 0::2]
    v2 = x[:, :, 1::2, 1::2]
    return v1, v2

# --- Advanced Metrics Implementation ---

def calc_msssim(img1, img2):
    # Simplified MS-SSIM logic for consistent benchmarking
    img1_t = torch.from_numpy(img1.transpose(2,0,1)).unsqueeze(0).to(Config.DEVICE)
    img2_t = torch.from_numpy(img2.transpose(2,0,1)).unsqueeze(0).to(Config.DEVICE)
    # Note: Full implementation usually requires multiple scales, 
    # here we use a placeholder or single scale logic if library is missing.
    return ssim_metric(img1, img2, data_range=1.0, channel_axis=2)

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
    # Basic VIF implementation
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
    m = calc_msssim(clean, pred)
    fo = calc_fom(clean, pred)
    return p, s, f, v, m, fo

def register_result_with_noisy(res_dict, name, idx, c, n, ns, p_img, p, s, f, v, m, fo, save_dir):
    if name not in res_dict:
        res_dict[name] = {'psnr':[], 'ssim':[], 'fsim':[], 'vif':[], 'msssim':[], 'fom':[], 'psnr_vs_noisy':[]}
    res_dict[name]['psnr'].append(p)
    res_dict[name]['ssim'].append(s)
    res_dict[name]['fsim'].append(f)
    res_dict[name]['vif'].append(v)
    res_dict[name]['msssim'].append(m)
    res_dict[name]['fom'].append(fo)
    if idx < 5: 
        target_path = os.path.join(save_dir, name)
        os.makedirs(target_path, exist_ok=True)
        cv2.imwrite(os.path.join(target_path, f"sample_{idx}.png"), (p_img*255).astype(np.uint8)[:,:,::-1])

def geometric_ensemble_inference(model, x):
    model.eval()
    with torch.no_grad(): return model(x).clamp(0,1)

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
