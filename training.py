import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast

from utils import Config, cleanup, count_parameters, inject_noise_by_type, mask_n2v, mask_adaptive, neighbor2neighbor_augment, geometric_ensemble_inference, register_result_with_noisy, compute_metrics, NoiseInjector
from models import UNet, WRTPNet_Baseline, WRTPNet_Attentive, WRTPNet_AdaptiveMask

# ===== CBM3D import / fallback =====
try:
    from bm3d import bm3d
    cbm3d_available = True
except Exception:
    try:
        import bm3d as _bm3d
        bm3d = _bm3d.bm3d if hasattr(_bm3d, 'bm3d') else _bm3d
        cbm3d_available = True
    except Exception:
        cbm3d_available = False
        print("WARNING: 'bm3d' package not found. Falling back to a simple Gaussian blur as CBM3D substitute.")

def geometric_ensemble_inference(model, x):
    dev = next(model.parameters()).device if any(p.requires_grad for p in model.parameters()) else Config.DEVICE
    x = x.to(dev)
    model.eval()
    if len(x.shape) == 3: x = x.unsqueeze(0)
    with torch.no_grad():
        out = model(x)
    return out.clamp(0,1)

def train_noise2void(model_name, test_pairs, noise_type, seed_dir, train_loader, scaler):
    print(f"  Running {model_name} (Noisier2Noisy + Masked Loss)...")
    cleanup()
    model = UNet(base=64).to(Config.DEVICE)
    print(f"    [{model_name}] Parameter Count: {count_parameters(model):,}")
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    t_start = time.time()
    for epoch in range(Config.EPOCHS):
        model.train()
        for y in train_loader:
            y = y.to(Config.DEVICE)
            noisy_input = inject_noise_by_type(y, noise_type)
            input_masked, mask = mask_n2v(noisy_input, Config.MASK_PERCENT)
            target = y 
            
            optimizer.zero_grad()
            if Config.USE_AMP:
                with autocast(device_type=Config.AMP_DEVICE_TYPE, dtype=Config.AMP_DTYPE):
                    pred = model(input_masked)
                    loss = F.l1_loss(pred * mask, target * mask, reduction='sum') / (mask.sum() + 1e-6)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                pred = model(input_masked)
                loss = F.l1_loss(pred * mask, target * mask, reduction='sum') / (mask.sum() + 1e-6)
                loss.backward(); optimizer.step()
                
    train_time = time.time() - t_start
    torch.save(model.state_dict(), os.path.join(seed_dir, f"model_{model_name}.pth"))
    
    inf_times = []
    benchmark_results = {}
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

    avg_inf_time = float(np.mean(inf_times)) if len(inf_times) > 0 else float("nan")
    benchmark_results[model_name]['inference_time_per_image_sec'] = avg_inf_time
    print(f"    [{model_name}] Inference Time Per Image: {avg_inf_time:.6f} sec")
            
    return benchmark_results[model_name]

def train_ne2ne(model_name, test_pairs, noise_type, seed_dir, train_loader, scaler):
    print(f"  Running {model_name} (FIXED: Subsampling Strategy)...")
    cleanup()
    model = UNet(base=64).to(Config.DEVICE)
    print(f"    [{model_name}] Parameter Count: {count_parameters(model):,}")
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    t_start = time.time()
    for epoch in range(Config.EPOCHS):
        model.train()
        for y in train_loader:
            y = y.to(Config.DEVICE)
            noisy_input = inject_noise_by_type(y, noise_type)
            v1, v2 = neighbor2neighbor_augment(noisy_input)
            
            optimizer.zero_grad()
            if Config.USE_AMP:
                with autocast(device_type=Config.AMP_DEVICE_TYPE, dtype=Config.AMP_DTYPE):
                    p1 = model(v1)
                    p2 = model(v2)
                    loss = F.l1_loss(p1, v2) + F.l1_loss(p2, v1)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                p1 = model(v1)
                p2 = model(v2)
                loss = F.l1_loss(p1, v2) + F.l1_loss(p2, v1)
                loss.backward(); optimizer.step()
                
    train_time = time.time() - t_start
    torch.save(model.state_dict(), os.path.join(seed_dir, f"model_{model_name}.pth"))
    
    inf_times = []
    benchmark_results = {}
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

    avg_inf_time = float(np.mean(inf_times)) if len(inf_times) > 0 else float("nan")
    benchmark_results[model_name]['inference_time_per_image_sec'] = avg_inf_time
    print(f"    [{model_name}] Inference Time Per Image: {avg_inf_time:.6f} sec")
    return benchmark_results[model_name]

def train_self2self(model_name, test_pairs, noise_type, seed_dir, train_loader, scaler):
    print(f"  Running {model_name} (Noisier2Noisy + Dropout)...")
    cleanup()
    model = UNet(base=64, dropout=0.3).to(Config.DEVICE)
    print(f"    [{model_name}] Parameter Count: {count_parameters(model):,}")
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    t_start = time.time()
    for epoch in range(Config.EPOCHS):
        model.train()
        for y in train_loader:
            y = y.to(Config.DEVICE)
            noisy_input = inject_noise_by_type(y, noise_type)
            target = y 
            
            optimizer.zero_grad()
            if Config.USE_AMP:
                with autocast(device_type=Config.AMP_DEVICE_TYPE, dtype=Config.AMP_DTYPE):
                    pred = model(noisy_input)
                    loss = F.l1_loss(pred, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                pred = model(noisy_input)
                loss = F.l1_loss(pred, target)
                loss.backward(); optimizer.step()
                
    train_time = time.time() - t_start
    torch.save(model.state_dict(), os.path.join(seed_dir, f"model_{model_name}.pth"))
    
    inf_times = []
    benchmark_results = {}
    
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
            
    with torch.no_grad():
        for i, (clean, noisy) in enumerate(test_pairs):
            st = time.time()
            noisy_synthetic = inject_noise_by_type(noisy, noise_type).squeeze(0)
            preds = []
            for _ in range(50): 
                preds.append(model(noisy))
            out = torch.stack(preds).mean(dim=0).clamp(0,1)
            inf_times.append(time.time() - st)
            c_np = clean.squeeze().cpu().permute(1,2,0).numpy()
            o_np = out.squeeze().cpu().permute(1,2,0).numpy()
            n_np = noisy.squeeze().cpu().permute(1,2,0).numpy()
            ns_np = noisy_synthetic.cpu().permute(1,2,0).numpy()
            p, s, f, v, m, fo = compute_metrics(c_np, o_np)
            register_result_with_noisy(benchmark_results, model_name, i, c_np, n_np, ns_np, o_np, p, s, f, v, m, fo, save_dir=seed_dir)

    avg_inf_time = float(np.mean(inf_times)) if len(inf_times) > 0 else float("nan")
    benchmark_results[model_name]['inference_time_per_image_sec'] = avg_inf_time
    print(f"    [{model_name}] Inference Time Per Image: {avg_inf_time:.6f} sec")
    return benchmark_results[model_name]

def train_wrtp_variants(model_class, model_name, test_pairs, noise_type, seed_dir, train_loader, scaler):
    print(f"  Running {model_name}...")
    cleanup()
    model = model_class(in_ch=3, base=96, num_blocks=6).to(Config.DEVICE)
    print(f"    [{model_name}] Parameter Count: {count_parameters(model):,}")
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    is_adaptive = 'AdaptiveMask' in model_name
    
    t_start = time.time()
    for epoch in range(Config.EPOCHS):
        model.train()
        for y in train_loader:
            y = y.to(Config.DEVICE)
            noisy_input = inject_noise_by_type(y, noise_type)
            target = y
            optimizer.zero_grad()
            
            if is_adaptive:
                input_masked, mask = mask_adaptive(noisy_input, Config.MASK_PERCENT)
                if Config.USE_AMP:
                    with autocast(device_type=Config.AMP_DEVICE_TYPE, dtype=Config.AMP_DTYPE):
                        pred = model(input_masked)
                        loss = F.l1_loss(pred * mask, target * mask, reduction='sum') / (mask.sum() + 1e-6)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer); scaler.update()
                else:
                    pred = model(input_masked)
                    loss = F.l1_loss(pred * mask, target * mask, reduction='sum') / (mask.sum() + 1e-6)
                    loss.backward(); optimizer.step()
            else:
                if Config.USE_AMP:
                    with autocast(device_type=Config.AMP_DEVICE_TYPE, dtype=Config.AMP_DTYPE):
                        pred = model(noisy_input)
                        loss = F.l1_loss(pred, target)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer); scaler.update()
                else:
                    pred = model(noisy_input)
                    loss = F.l1_loss(pred, target)
                    loss.backward(); optimizer.step()
                    
    train_time = time.time() - t_start
    torch.save(model.state_dict(), os.path.join(seed_dir, f"model_{model_name}.pth"))
    
    inf_times = []
    benchmark_results = {}
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

    avg_inf_time = float(np.mean(inf_times)) if len(inf_times) > 0 else float("nan")
    benchmark_results[model_name]['inference_time_per_image_sec'] = avg_inf_time
    print(f"    [{model_name}] Inference Time Per Image: {avg_inf_time:.6f} sec")
    return benchmark_results[model_name]

def train_noise2same(model_name, test_pairs, noise_type, seed_dir, train_loader, scaler):
    print(f"  Running {model_name} (Fixed as Noise2Self logic)...")
    cleanup()
    model = UNet(base=64).to(Config.DEVICE)
    print(f"    [{model_name}] Parameter Count: {count_parameters(model):,}")
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    t_start = time.time()
    for epoch in range(Config.EPOCHS):
        model.train()
        for y in train_loader:
            y = y.to(Config.DEVICE)
            noisy_input = inject_noise_by_type(y, noise_type)
            masked_input, mask = mask_n2v(noisy_input, Config.MASK_PERCENT)
            
            optimizer.zero_grad()
            if Config.USE_AMP:
                with autocast(device_type=Config.AMP_DEVICE_TYPE, dtype=Config.AMP_DTYPE):
                    pred = model(masked_input)
                    loss = F.mse_loss(pred * mask, noisy_input * mask, reduction='sum') / (mask.sum() + 1e-6)
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                pred = model(masked_input)
                loss = F.mse_loss(pred * mask, noisy_input * mask, reduction='sum') / (mask.sum() + 1e-6)
                loss.backward(); optimizer.step()
                
    train_time = time.time() - t_start
    torch.save(model.state_dict(), os.path.join(seed_dir, f"model_{model_name}.pth"))
    
    inf_times = []
    benchmark_results = {}
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

    avg_inf_time = float(np.mean(inf_times)) if len(inf_times) > 0 else float("nan")
    benchmark_results[model_name]['inference_time_per_image_sec'] = avg_inf_time
    print(f"    [{model_name}] Inference Time Per Image: {avg_inf_time:.6f} sec")
    return benchmark_results[model_name]

def run_dip_benchmark(test_pairs, noise_type, seed_dir):
    print("  Running DIP (Deep Image Prior) per-image optimization...")
    benchmark_results = {}
    inf_times = []
    dip_iters = Config.DIP_ITERS
    dip_lr = Config.DIP_LR

    device = Config.DEVICE
    for i, (clean, noisy) in enumerate(test_pairs):
        st_total = time.time()
        c_np = clean.squeeze().cpu().permute(1,2,0).numpy()
        n_np = noisy.squeeze().cpu().permute(1,2,0).numpy()
        ns_np = inject_noise_by_type(noisy, noise_type).squeeze(0).cpu().permute(1,2,0).numpy()

        model = UNet(base=64).to(device)
        model.train()
        z = torch.randn_like(noisy).to(device)
        optimizer = optim.Adam(model.parameters(), lr=dip_lr)

        best_psnr = -1.0
        best_pred_np = None

        for it in range(dip_iters):
            optimizer.zero_grad()
            pred = model(z)
            loss = F.mse_loss(pred, noisy.to(device))
            loss.backward()
            optimizer.step()

            if (it % 50) == 0 or it == dip_iters - 1:
                with torch.no_grad():
                    out_np = pred.squeeze().detach().cpu().permute(1,2,0).numpy()
                    out_np = np.clip(out_np, 0, 1)
                    p, s, f, v, m, fo = compute_metrics(c_np, out_np)
                    if p > best_psnr:
                        best_psnr = p
                        best_pred_np = out_np.copy()

        inf_time = time.time() - st_total
        inf_times.append(inf_time)

        if best_pred_np is None:
            with torch.no_grad():
                pred = model(z)
                best_pred_np = pred.squeeze().detach().cpu().permute(1,2,0).numpy()
                best_pred_np = np.clip(best_pred_np, 0, 1)

        p, s, f, v, m, fo = compute_metrics(c_np, best_pred_np)
        register_result_with_noisy(benchmark_results, "DIP", i, c_np, n_np, ns_np, best_pred_np, p, s, f, v, m, fo, save_dir=seed_dir)
        del model
        torch.cuda.empty_cache()

    avg_inf_time = float(np.mean(inf_times)) if len(inf_times) > 0 else float("nan")
    if "DIP" in benchmark_results:
        benchmark_results["DIP"]['inference_time_per_image_sec'] = avg_inf_time
    print(f"    [DIP] Avg optimization time per image: {avg_inf_time:.6f} sec")
    return benchmark_results

def apply_bm3d_to_numpy(img_np, sigma=Config.SIGMA):
    img_np = np.clip(img_np, 0, 1)
    out = np.zeros_like(img_np)
    if cbm3d_available:
        try:
            for ch in range(img_np.shape[2]):
                out[:,:,ch] = bm3d(img_np[:,:,ch], sigma_psd=sigma)
        except Exception:
            for ch in range(img_np.shape[2]):
                try:
                    out[:,:,ch] = bm3d(img_np[:,:,ch], sigma)
                except Exception:
                    out[:,:,ch] = cv2.GaussianBlur((img_np[:,:,ch]*255).astype(np.uint8), (3,3), sigmaX=max(0.1, sigma*255)).astype(np.float32)/255.0
    else:
        for ch in range(img_np.shape[2]):
            out[:,:,ch] = cv2.GaussianBlur((img_np[:,:,ch]*255).astype(np.uint8), (3,3), sigmaX=max(0.1, sigma*255)).astype(np.float32)/255.0
    return np.clip(out, 0, 1)

def run_cbm3d_benchmark(test_pairs, noise_type, seed_dir):
    print("  Running CBM3D (non-learning denoiser)...")
    benchmark_results = {}
    inf_times = []
    inf_times_synth = []
    with torch.no_grad():
        for i, (clean, noisy) in enumerate(test_pairs):
            n_np = noisy.squeeze().cpu().permute(1,2,0).numpy()
            c_np = clean.squeeze().cpu().permute(1,2,0).numpy()
            noisy_synthetic = inject_noise_by_type(noisy, noise_type).squeeze(0)
            ns_np = noisy_synthetic.cpu().permute(1,2,0).numpy()

            st = time.time()
            denoised_np = apply_bm3d_to_numpy(n_np, sigma=Config.SIGMA)
            inf_times.append(time.time() - st)
            p, s, f, v, m, fo = compute_metrics(c_np, denoised_np)
            register_result_with_noisy(benchmark_results, "CBM3D", i, c_np, n_np, ns_np, denoised_np, p, s, f, v, m, fo, save_dir=seed_dir)

            st2 = time.time()
            denoised_synth_np = apply_bm3d_to_numpy(ns_np, sigma=Config.SIGMA)
            inf_times_synth.append(time.time() - st2)
            p2, s2, f2, v2, m2, fo2 = compute_metrics(c_np, denoised_synth_np)
            register_result_with_noisy(benchmark_results, "CBM3D_SyntheticInput", i, c_np, n_np, ns_np, denoised_synth_np, p2, s2, f2, v2, m2, fo2, save_dir=seed_dir)

    avg_inf_time = float(np.mean(inf_times)) if len(inf_times) > 0 else float("nan")
    avg_inf_time_synth = float(np.mean(inf_times_synth)) if len(inf_times_synth) > 0 else float("nan")
    if "CBM3D" in benchmark_results:
        benchmark_results["CBM3D"]['inference_time_per_image_sec'] = avg_inf_time
    if "CBM3D_SyntheticInput" in benchmark_results:
        benchmark_results["CBM3D_SyntheticInput"]['inference_time_per_image_sec'] = avg_inf_time_synth
    print(f"    [CBM3D] Inference Time Per Image: {avg_inf_time:.6f} sec")
    return benchmark_results
