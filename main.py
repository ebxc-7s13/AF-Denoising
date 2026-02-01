import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.amp import GradScaler

from utils import Config, BenchmarkDataset, NoiseInjector, seed_everything, cleanup, zip_results
import engine
from models import WRTPNet_Baseline, WRTPNet_Attentive, WRTPNet_AdaptiveMask

if __name__ == "__main__":
    print("\n" + "="*70)
    print("WRTPNET ABLATION 2: THE TRUTH (ALL MODELS)")
    print("="*70)
    cleanup()
    
    dataset = BenchmarkDataset(Config.INPUT_DIR)
    n = len(dataset)
    train_n = int(0.8 * n) 
    test_n = n - train_n   
    
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_n, test_n])
    
    print(f"Dataset Split: 80% Train ({train_n} images), 20% Test/Validation ({test_n} images)")

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    scaler = GradScaler(device=Config.AMP_DEVICE_TYPE) if Config.USE_AMP else None
    
    all_results_buffer = []
    NOISE_TYPES = ["PoissonGauss", "Gaussian", "Poisson"]
    
    raw_csv_path = os.path.join(Config.OUTPUT_DIR, "raw_results_log.csv")
    
    for noise_type in NOISE_TYPES:
        print(f"\n--- {noise_type} ---")
        for seed in Config.SEEDS:
            print(f"\n[SEED START] Noise: {noise_type} | Seed: {seed}")
            seed_everything(seed)
            test_pairs = []
            for i, clean in enumerate(test_loader):
                if i >= 50: break
                clean = clean.to(Config.DEVICE)
                if noise_type == "Gaussian":
                    noisy = NoiseInjector.add_gaussian(clean, sigma=Config.SIGMA)
                elif noise_type == "Poisson":
                    noisy = NoiseInjector.add_poisson(clean, peak=Config.POISSON_PEAK)
                else:
                    noisy = NoiseInjector.add_poisson_gaussian(clean, peak=Config.POISSON_PEAK, sigma_read=Config.SIGMA_READ)
                test_pairs.append((clean, noisy))
                
            seed_dir = os.path.join(Config.AGGREGATED_DIR, noise_type, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            
            # Helper to log results
            def log_res(res, method_name):
                inf_t = res.get('inference_time_per_image_sec', float("nan"))
                for j in range(len(res['psnr'])):
                    row_dict = {
                        'seed': seed, 'noise_type': noise_type, 'method': method_name, 
                        'psnr': res['psnr'][j], 'ssim': res['ssim'][j], 'fsim': res['fsim'][j], 
                        'vif': res['vif'][j], 'ms_ssim': res['msssim'][j], 'fom': res['fom'][j],
                        'psnr_vs_noisy': res['psnr_vs_noisy'][j], 'ssim_vs_noisy': res['ssim_vs_noisy'][j], 
                        'fsim_vs_noisy': res['fsim_vs_noisy'][j], 'vif_vs_noisy': res['vif_vs_noisy'][j], 
                        'msssim_vs_noisy': res['msssim_vs_noisy'][j], 'fom_vs_noisy': res['fom_vs_noisy'][j],
                        'inference_time_per_image_sec': inf_t
                    }
                    all_results_buffer.append(row_dict)
                    pd.DataFrame([row_dict]).to_csv(raw_csv_path, mode='a', header=not os.path.exists(raw_csv_path), index=False)

            # 1. N2V
            res = engine.train_noise2void('Noise2Void', test_pairs, noise_type, seed_dir, train_loader, scaler)
            log_res(res, 'Noise2Void')
                
            # 2. Ne2Ne
            res = engine.train_ne2ne('Neighbor2Neighbor', test_pairs, noise_type, seed_dir, train_loader, scaler)
            log_res(res, 'Neighbor2Neighbor')
                
            # 3. Self2Self
            res = engine.train_self2self('Self2Self', test_pairs, noise_type, seed_dir, train_loader, scaler)
            log_res(res, 'Self2Self')
                
            # 4. WRTPNet Baseline
            res = engine.train_wrtp_variants(WRTPNet_Baseline, 'WRTPNet_Baseline', test_pairs, noise_type, seed_dir, train_loader, scaler)
            log_res(res, 'WRTPNet_Baseline')
            
            # 5. WRTPNet Attentive
            try:
                res = engine.train_wrtp_variants(WRTPNet_Attentive, 'WRTPNet_Attentive', test_pairs, noise_type, seed_dir, train_loader, scaler)
                log_res(res, 'WRTPNet_Attentive')
            except Exception: pass
            
            # 6. CBM3D
            cbm_res = engine.run_cbm3d_benchmark(test_pairs, noise_type, seed_dir)
            if "CBM3D" in cbm_res: log_res(cbm_res["CBM3D"], 'CBM3D')
            if "CBM3D_SyntheticInput" in cbm_res: log_res(cbm_res["CBM3D_SyntheticInput"], 'CBM3D_SyntheticInput')

            # 7. Noise2Same
            try:
                res = engine.train_noise2same('Noise2Same', test_pairs, noise_type, seed_dir, train_loader, scaler)
                log_res(res, 'Noise2Same')
            except Exception as e: print(f"Noise2Same run failed: {e}")

            # 8. DIP
            try:
                dip_res = engine.run_dip_benchmark(test_pairs, noise_type, seed_dir)
                if "DIP" in dip_res: log_res(dip_res["DIP"], 'DIP')
            except Exception as e: print(f"DIP run failed: {e}")

    # --- FINAL SUMMARY ---
    if all_results_buffer:
        df = pd.DataFrame(all_results_buffer)
        numeric_cols = ['psnr', 'ssim', 'fsim', 'vif', 'ms_ssim', 'fom', 'inference_time_per_image_sec']
        numeric_cols_vs_noisy = ['psnr_vs_noisy', 'ssim_vs_noisy', 'fsim_vs_noisy', 'vif_vs_noisy', 'msssim_vs_noisy', 'fom_vs_noisy']
        all_numeric_cols = numeric_cols + numeric_cols_vs_noisy
        
        grouped = df.groupby(['noise_type', 'method'])[all_numeric_cols].agg(['mean', 'std'])
        summary_data = []
        for idx in grouped.index:
            noise_t, method_n = idx
            row_dict = {'Noise Type': noise_t, 'Model': method_n}
            for metric in all_numeric_cols:
                m_val = grouped.loc[idx, (metric, 'mean')]
                s_val = grouped.loc[idx, (metric, 'std')]
                row_dict[metric.upper()] = f"{m_val:.6f} ± {s_val:.6f}"
            summary_data.append(row_dict)
        final_summary_df = pd.DataFrame(summary_data)
        print(final_summary_df.to_string(index=False))
        final_summary_df.to_csv(os.path.join(Config.OUTPUT_DIR, "final_ablation_summary_mean_sd.csv"), index=False)
        
    zip_results()
