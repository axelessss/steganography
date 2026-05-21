import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from scipy import stats
from pathlib import Path

DATASETS = {
    'bossbase': 'data1',
    'medical': 'chest_xrays', 
    'other': 'airplane_bmp'
}
K_VALUES = [1, 2, 3]
ALPHA = 0.05
CONFIDENCE = 1 - ALPHA

def calculate_bit_plane_variance(image_array, k):
    """Расчёт дисперсии значений в k-й битовой плоскости"""
    bit_position = k - 1
    bit_plane = (image_array >> bit_position) & 1
    return np.var(bit_plane)

def embed_lsb(image_array, message_bits, k):
    """Внедрение случайных бит (имитация сообщения) в k-ю плоскость"""
    bit_position = k - 1
    mask_clear = ~(1 << bit_position) & 0xFF
    flat = image_array.flatten().copy()
    
    np.random.seed(42)
    random_bits = np.random.randint(0, 2, min(len(flat), 245760))
    
    for i, bit in enumerate(random_bits):
        flat[i] = (flat[i] & mask_clear) | (bit << bit_position)
    
    return flat.reshape(image_array.shape).astype(np.uint8)

def calculate_confidence_interval(data, confidence=0.95):
    """Расчёт доверительного интервала по t-распределению"""
    n = len(data)
    if n < 2:
        return np.mean(data), np.mean(data), np.mean(data)
    
    mean = np.mean(data)
    sem = stats.sem(data)
    margin = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    
    return mean, mean - margin, mean + margin

def analyze_dataset(dataset_name, dataset_path, k_values):
    """Анализ одного набора изображений"""
    results = {'psnr': {k: [] for k in k_values}, 
               'variance': {k: [] for k in k_values}}
    
    image_files = list(Path(dataset_path).glob('*.bmp'))[:100]
    
    for img_path in image_files:
        try:
            img = np.array(Image.open(img_path).convert('L'))
            
            for k in k_values:
                # Внедрение и расчёт PSNR
                stego = embed_lsb(img, None, k)
                psnr = peak_signal_noise_ratio(img, stego, data_range=255)
                results['psnr'][k].append(psnr)
                
                # Дисперсия битовой плоскости оригинала
                var = calculate_bit_plane_variance(img, k)
                results['variance'][k].append(var)
                
        except Exception as e:
            print(f" Ошибка {img_path.name}: {e}")
            continue
    
    return results

print(" Запуск статистического анализа...")
all_results = {}

for name, path in DATASETS.items():
    print(f" Обработка набора: {name}")
    all_results[name] = analyze_dataset(name, path, K_VALUES)

print("\n Расчёт доверительных интервалов (95%):")
print("="*70)

ci_table_psnr = []
ci_table_var = []

for dataset in DATASETS.keys():
    for k in K_VALUES:
        # PSNR
        psnr_data = all_results[dataset]['psnr'][k]
        mean_psnr, ci_low_psnr, ci_high_psnr = calculate_confidence_interval(psnr_data, CONFIDENCE)
        ci_table_psnr.append({
            'Dataset': dataset, 'k': k,
            'Mean_PSNR': mean_psnr, 'CI_Low': ci_low_psnr, 'CI_High': ci_high_psnr,
            'Std': np.std(psnr_data), 'N': len(psnr_data)
        })
        
        # Variance
        var_data = all_results[dataset]['variance'][k]
        mean_var, ci_low_var, ci_high_var = calculate_confidence_interval(var_data, CONFIDENCE)
        ci_table_var.append({
            'Dataset': dataset, 'k': k,
            'Mean_Var': mean_var, 'CI_Low': ci_low_var, 'CI_High': ci_high_var,
            'Std': np.std(var_data)
        })
        
        print(f"{dataset:12} | k={k} | PSNR: {mean_psnr:5.2f} [{ci_low_psnr:5.2f}; {ci_high_psnr:5.2f}] | Var: {mean_var:.4f}")

df_psnr = pd.DataFrame(ci_table_psnr)
df_var = pd.DataFrame(ci_table_var)

df_psnr.to_csv('results/ci_psnr.csv', index=False)
df_var.to_csv('results/ci_variance.csv', index=False)
print("\n Результаты сохранены: results/ci_psnr.csv, results/ci_variance.csv")

def plot_ci_comparison(df, metric_name, ylabel, filename):
    """Построение графика с доверительными интервалами"""
    x = np.arange(len(K_VALUES))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for i, (dataset, color) in enumerate(zip(DATASETS.keys(), colors)):
        subset = df[df['Dataset'] == dataset]
        means = [subset[subset['k']==k][metric_name].values[0] for k in K_VALUES]
        errors = [(subset[subset['k']==k]['CI_High'].values[0] - 
                   subset[subset['k']==k]['CI_Low'].values[0]) / 2 for k in K_VALUES]
        
        ax.errorbar(x + i*width, means, yerr=errors, 
                   label=dataset, fmt='o', capsize=5, 
                   color=color, ecolor=color, alpha=0.8)
    
    ax.set_xlabel('Номер битовой плоскости (k)', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f'95% доверительные интервалы: {metric_name}', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(K_VALUES)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

os.makedirs('results/plots', exist_ok=True)
plot_ci_comparison(df_psnr, 'Mean_PSNR', 'PSNR (дБ)', 'results/plots/ci_psnr.png')
plot_ci_comparison(df_var, 'Mean_Var', 'Дисперсия битовой плоскости', 'results/plots/ci_variance.png')