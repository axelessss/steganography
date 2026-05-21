import os, glob, csv, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Предотвращает ошибки GUI в VirtualBox/SSH
import matplotlib.pyplot as plt
from PIL import Image

from lab1 import load_bmp, save_bmp, extract_plane, embed_message, calc_metrics

DATASETS = {
    "bossbase": "data1",
    "medical": "chest_xrays",
    "other": "airplane_bmp"
}
TEXT_FILE = "message.txt"

def process_dataset(name, path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(path, "*.bmp")))
    if not files:
        print(f"В {path} нет BMP файлов.")
        return []

    print(f"\n[{name}] Генерация битовых плоскостей (5 изображений)...")
    for img_path in files[:5]:
        base = os.path.basename(img_path)
        arr = load_bmp(img_path)
        for k in range(1, 9):
            plane = extract_plane(arr, k)
            save_bmp(plane, os.path.join(out_dir, f"plane_{base}_k{k}.bmp"))

    print(f"🔍 [{name}] Внедрение текста (k=1,2,3) и расчёт метрик...")
    test_img = files[0]
    base = os.path.basename(test_img)
    results = []

    for k in [1, 2, 3]:
        stego_path = os.path.join(out_dir, f"stego_{base}_k{k}.bmp")
        embed_message(test_img, TEXT_FILE, k, stego_path)

        mse, psnr, ssim = calc_metrics(test_img, stego_path)
        results.append([name, base, k, f"{mse:.4f}", f"{psnr:.2f}", f"{ssim:.4f}"])
        print(f"  ✅ k={k} | MSE: {mse:.4f} | PSNR: {psnr:.2f} дБ | SSIM: {ssim:.4f}")

        orig_flat = load_bmp(test_img).flatten()
        stego_flat = load_bmp(stego_path).flatten()
        plt.figure(figsize=(8, 4))
        plt.hist(orig_flat, bins=256, range=(0, 255), alpha=0.6, label='Original', color='blue')
        plt.hist(stego_flat, bins=256, range=(0, 255), alpha=0.6, label=f'Stego k={k}', color='red')
        plt.title(f"Гистограмма яркости: {name} / {base} (k={k})")
        plt.xlabel("Интенсивность пикселя"); plt.ylabel("Частота")
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"hist_{base}_k{k}.png"), dpi=150)
        plt.close()

    return results

def main():
    if not os.path.exists(TEXT_FILE):
        print("Файл message.txt не найден в текущей папке.")
        sys.exit(1)

    all_results = []
    for name, path in DATASETS.items():
        if os.path.exists(path):
            out_dir = f"results_{name}_lab1"
            res = process_dataset(name, path, out_dir)
            all_results.extend(res)
        else:
            print(f"Папка {path} не найдена. Пропускаю.")

    if all_results:
        with open("research_metrics_lab1.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Dataset", "Image", "k", "MSE", "PSNR(dB)", "SSIM"])
            writer.writerows(all_results)
        print("\nИССЛЕДОВАНИЕ ЗАВЕРШЕНО!")
        print(" Таблица метрик: research_metrics.csv")
        print(" Результаты по наборам: results_bossbase/, results_medical/, results_other/")
    else:
        print("\nНи один набор не обработан.")

if __name__ == "__main__":
    main()