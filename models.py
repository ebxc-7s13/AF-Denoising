import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.amp import GradScaler

from utils import Config, BenchmarkDataset, inject_noise_by_type, seed_everything, cleanup, zip_results
import training as engine 
from models import FASCANet, SwinConvDenoiser

if __name__ == "__main__":
    print("\n" + "="*70)
    print("FASCANet & SwinConv BENCHMARKING")
    print("="*70)
    cleanup()
    
    dataset = BenchmarkDataset(Config.INPUT_DIR)
    n = len(dataset)
    train_n = int(0.8 * n) 
    test_n = n - train_n   
    
    train_ds = torch.utils.data.Subset(dataset, range(train_n))
    test_ds  = torch.utils.data.Subset(dataset, range(train_n, n))
    
    print(f"Dataset Split: 80% Train ({train_n} images), 20% Test ({test_n} images)")

    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    scaler = GradScaler(device=Config.AMP_DEVICE_TYPE) if Config.USE_AMP else None
    
    all_results_buffer = []
    NOISE_TYPES = ["PoissonGauss", "Gaussian", "Poisson"]
    raw_csv_path = os.path.join(Config.OUTPUT_DIR, "raw_results_log.csv")
    
    for noise_type in NOISE_TYPES:
        print(f"\n--- Noise Model: {noise_type} ---")
        for seed in Config.SEEDS:
            print(f"\n[Seed {seed}]")
            seed_everything(seed)
            test_pairs = []
            for i, clean in enumerate(test_loader):
                if i >= 50: break
                clean = clean.to(Config.DEVICE)
                noisy = inject_noise_by_type(clean, noise_type)
                test_pairs.append((clean, noisy))
                
            seed_dir = os.path.join(Config.AGGREGATED_DIR, noise_type, f"seed_{seed}")
            os.makedirs(seed_dir, exist_ok=True)
            
            def log_res(res, method_name):
                inf_t = res.get('inference_time_per_image_sec', float("nan"))
                for j in range(len(res['psnr'])):
                    row_dict = {
                        'seed': seed, 'noise_type': noise_type, 'method': method_name, 
                        'psnr': res['psnr'][j], 'ssim': res['ssim'][j], 'fsim': res['fsim'][j], 
                        'vif': res['vif'][j], 'ms_ssim': res['msssim'][j], 'fom': res['fom'][j],
                        'psnr_vs_noisy': res['psnr_vs_noisy'][j], 
                        'ssim_vs_noisy': res['ssim_vs_noisy'][j], 
                        'fsim_vs_noisy': res['fsim_vs_noisy'][j], 
                        'vif_vs_noisy': res['vif_vs_noisy'][j], 
                        'msssim_vs_noisy': res['msssim_vs_noisy'][j], 
                        'fom_vs_noisy': res['fom_vs_noisy'][j],
                        'inference_time_per_image_sec': inf_t
                    }
                    all_results_buffer.append(row_dict)
                    pd.DataFrame([row_dict]).to_csv(raw_csv_path, mode='a', header=not os.path.exists(raw_csv_path), index=False)

            # Benchmarking list
            log_res(engine.train_noise2void('Noise2Void', test_pairs, noise_type, seed_dir, train_loader, scaler), 'Noise2Void')
            log_res(engine.train_ne2ne('Neighbor2Neighbor', test_pairs, noise_type, seed_dir, train_loader, scaler), 'Neighbor2Neighbor')
            log_res(engine.train_self2self('Self2Self', test_pairs, noise_type, seed_dir, train_loader, scaler), 'Self2Self')
            log_res(engine.train_noise2same('Noise2Same', test_pairs, noise_type, seed_dir, train_loader, scaler), 'Noise2Same')
            log_res(engine.train_fascanet(test_pairs, noise_type, seed_dir, train_loader, scaler), 'FASCANet')
            log_res(engine.train_swinconv(test_pairs, noise_type, seed_dir, train_loader, scaler), 'SwinConvDenoiser')
            
            cbm_res = engine.run_cbm3d_benchmark(test_pairs, noise_type, seed_dir)
            if "CBM3D" in cbm_res: log_res(cbm_res["CBM3D"], 'CBM3D')
            
            dip_res = engine.run_dip_benchmark(test_pairs, noise_type, seed_dir)
            if "DIP" in dip_res: log_res(dip_res["DIP"], 'DIP')

    if all_results_buffer:
        df = pd.DataFrame(all_results_buffer)
        numeric_cols = ['psnr', 'ssim', 'fsim', 'vif', 'ms_ssim', 'fom', 
                        'psnr_vs_noisy', 'ssim_vs_noisy', 'fsim_vs_noisy', 
                        'vif_vs_noisy', 'msssim_vs_noisy', 'fom_vs_noisy']
        grouped = df.groupby(['noise_type', 'method'])[numeric_cols].agg(['mean', 'std'])
        print("\nFinal Aggregated Summary:")
        print(grouped)
        grouped.to_csv(os.path.join(Config.OUTPUT_DIR, "final_benchmark_summary.csv"))
        
    zip_results()
