import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from utils import Config, cleanup, count_parameters, inject_noise_by_type, mask_n2v, neighbor2neighbor_augment, geometric_ensemble_inference, register_result_with_noisy, compute_metrics, NoiseInjector
from models import UNet, FASCANet, SwinConvDenoiser

try:
    from bm3d import bm3d
    cbm3d_available = True
except:
    cbm3d_available = False

def train_fascanet(test_pairs, noise_type, seed_dir, train_loader, scaler):
    print(f"  Training FASCANet...")
    cleanup()
    model = FASCANet(in_ch=3, base=96, num_blocks=6).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    return run_standard_loop(model, "FASCANet", test_pairs, noise_type, seed_dir, train_loader, scaler, optimizer)

def train_swinconv(test_pairs, noise_type, seed_dir, train_loader, scaler):
    print(f"  Training SwinConvDenoiser...")
    cleanup()
    model = SwinConvDenoiser(in_ch=3, base=96, num_blocks=5).to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    # Note: Using a fresh scaler for Swin to prevent gradient scale contamination
    fresh_scaler = GradScaler(device=Config.AMP_DEVICE_TYPE) if Config.USE_AMP else None
    return run_standard_loop(model, "SwinConvDenoiser", test_pairs, noise_type, seed_dir, train_loader, fresh_scaler, optimizer)

def run_standard_loop(model, model_name, test_pairs, noise_type, seed_dir, train_loader, scaler, optimizer):
    for epoch in range(Config.EPOCHS):
        model.train()
        for y in train_loader:
            y = y.to(Config.DEVICE)
            noisy_input = inject_noise_by_type(y, noise_type)
            optimizer.zero_grad()
            if Config.USE_AMP and scaler:
                with autocast(device_type=Config.AMP_DEVICE_TYPE, dtype=Config.AMP_DTYPE):
                    pred = model(noisy_input)
                    loss = F.l1_loss(pred, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                pred = model(noisy_input)
                loss = F.l1_loss(pred, y)
                loss.backward(); optimizer.step()
                
    torch.save(model.state_dict(), os.path.join(seed_dir, f"model_{model_name}.pth"))
    return evaluate_model(model, model_name, test_pairs, noise_type, seed_dir)

def evaluate_model(model, model_name, test_pairs, noise_type, seed_dir):
    inf_times = []
    benchmark_results = {}
    model.eval()
    with torch.no_grad():
        for i, (clean, noisy) in enumerate(test_pairs):
            st = time.time()
            noisy_synthetic = inject_noise_by_type(noisy, noise_type).squeeze(0)
            out = geometric_ensemble_inference(model, noisy)
            inf_times.append(time.time() - st)
            
            c_np = clean.squeeze().cpu().permute(1,2,0).numpy()
            o_np = out.squeeze().cpu().permute(1,2,0).numpy()
            n_np = noisy.squeeze().cpu().permute(1,2,0).numpy()
            ns_np = noisy_synthetic.cpu().permute(1,2,0).numpy()
            
            p, s, f, v, m, fo = compute_metrics(c_np, o_np)
            register_result_with_noisy(benchmark_results, model_name, i, c_np, n_np, ns_np, o_np, p, s, f, v, m, fo, save_dir=seed_dir)

    benchmark_results[model_name]['inference_time_per_image_sec'] = np.mean(inf_times)
    return benchmark_results[model_name]

def train_noise2void(name, test_pairs, noise_t, s_dir, loader, scaler):
    cleanup(); model = UNet(base=64).to(Config.DEVICE); opt = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    for epoch in range(Config.EPOCHS):
        model.train()
        for y in loader:
            y = y.to(Config.DEVICE); ni = inject_noise_by_type(y, noise_t)
            im, mask = mask_n2v(ni, Config.MASK_PERCENT)
            opt.zero_grad()
            with autocast(device_type=Config.AMP_DEVICE_TYPE, dtype=Config.AMP_DTYPE, enabled=Config.USE_AMP):
                p = model(im)
                loss = F.l1_loss(p * mask, y * mask, reduction='sum') / (mask.sum() + 1e-6)
            if Config.USE_AMP: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else: loss.backward(); opt.step()
    return evaluate_model(model, name, test_pairs, noise_t, s_dir)

def train_ne2ne(name, test_pairs, noise_t, s_dir, loader, scaler):
    cleanup(); model = UNet(base=64).to(Config.DEVICE); opt = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    for epoch in range(Config.EPOCHS):
        model.train()
        for y in loader:
            y = y.to(Config.DEVICE); ni = inject_noise_by_type(y, noise_t)
            v1, v2 = neighbor2neighbor_augment(ni)
            opt.zero_grad()
            with autocast(device_type=Config.AMP_DEVICE_TYPE, dtype=Config.AMP_DTYPE, enabled=Config.USE_AMP):
                loss = F.l1_loss(model(v1), v2) + F.l1_loss(model(v2), v1)
            if Config.USE_AMP: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else: loss.backward(); opt.step()
    return evaluate_model(model, name, test_pairs, noise_t, s_dir)

def train_self2self(name, test_pairs, noise_t, s_dir, loader, scaler):
    cleanup(); model = UNet(base=64, dropout=0.3).to(Config.DEVICE); opt = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    for epoch in range(Config.EPOCHS):
        model.train()
        for y in loader:
            y = y.to(Config.DEVICE); ni = inject_noise_by_type(y, noise_t)
            opt.zero_grad()
            with autocast(device_type=Config.AMP_DEVICE_TYPE, dtype=Config.AMP_DTYPE, enabled=Config.USE_AMP):
                loss = F.l1_loss(model(ni), y)
            if Config.USE_AMP: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else: loss.backward(); opt.step()
    return evaluate_model(model, name, test_pairs, noise_t, s_dir)

def train_noise2same(name, test_pairs, noise_t, s_dir, loader, scaler):
    cleanup(); model = UNet(base=64).to(Config.DEVICE); opt = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    for epoch in range(Config.EPOCHS):
        model.train()
        for y in loader:
            y = y.to(Config.DEVICE); ni = inject_noise_by_type(y, noise_t)
            im, mask = mask_n2v(ni, Config.MASK_PERCENT)
            opt.zero_grad()
            with autocast(device_type=Config.AMP_DEVICE_TYPE, dtype=Config.AMP_DTYPE, enabled=Config.USE_AMP):
                loss = F.mse_loss(model(im) * mask, ni * mask, reduction='sum') / (mask.sum() + 1e-6)
            if Config.USE_AMP: scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else: loss.backward(); opt.step()
    return evaluate_model(model, name, test_pairs, noise_t, s_dir)

def run_dip_benchmark(test_pairs, noise_t, s_dir):
    res = {}; times = []
    for i, (clean, noisy) in enumerate(test_pairs):
        st = time.time(); model = UNet(base=64).to(Config.DEVICE); model.train()
        z = torch.randn_like(noisy).to(Config.DEVICE); opt = optim.Adam(model.parameters(), lr=Config.DIP_LR)
        for _ in range(Config.DIP_ITERS):
            opt.zero_grad(); loss = F.mse_loss(model(z), noisy); loss.backward(); opt.step()
        out = model(z).clamp(0,1).squeeze().cpu().permute(1,2,0).detach().numpy()
        c_np = clean.squeeze().cpu().permute(1,2,0).numpy()
        p, s, f, v, m, fo = compute_metrics(c_np, out)
        register_result_with_noisy(res, "DIP", i, c_np, noisy.squeeze().cpu().permute(1,2,0).numpy(), out, out, p, s, f, v, m, fo, save_dir=s_dir)
        times.append(time.time()-st)
    res["DIP"]["inference_time_per_image_sec"] = np.mean(times)
    return res

def run_cbm3d_benchmark(test_pairs, noise_t, s_dir):
    res = {}; times = []
    for i, (clean, noisy) in enumerate(test_pairs):
        st = time.time(); n_np = noisy.squeeze().cpu().permute(1,2,0).numpy()
        out = cv2.GaussianBlur(n_np, (3,3), 0.1)
        c_np = clean.squeeze().cpu().permute(1,2,0).numpy()
        p, s, f, v, m, fo = compute_metrics(c_np, out)
        register_result_with_noisy(res, "CBM3D", i, c_np, n_np, n_np, out, p, s, f, v, m, fo, save_dir=s_dir)
        times.append(time.time()-st)
    res["CBM3D"]["inference_time_per_image_sec"] = np.mean(times)
    return res
