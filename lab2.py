import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from scipy import stats
import os
import math
import csv
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class StegoWatermarkLab:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Лаб. №2: Цифровые водяные знаки")
        self.root.geometry("750x600")
        self.container_path = tk.StringVar()
        self.wm_path = tk.StringVar()
        self.key_var = tk.IntVar(value=42)
        self.method_var = tk.StringVar(value="lsb")
        self.last_wm_shape = None
        self.setup_gui()
        os.makedirs("results", exist_ok=True)

    def setup_gui(self):
        frm = ttk.Frame(self.root, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Контейнер (изображение):").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(frm, textvariable=self.container_path, width=60).grid(row=0, column=1, padx=5)
        ttk.Button(frm, text="Обзор", command=self.select_container).grid(row=0, column=2)

        ttk.Label(frm, text="Водяной знак (логотип):").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(frm, textvariable=self.wm_path, width=60).grid(row=1, column=1, padx=5)
        ttk.Button(frm, text="Обзор", command=self.select_wm).grid(row=1, column=2)

        ttk.Label(frm, text="Секретный ключ (для LSB):").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(frm, textvariable=self.key_var, width=10).grid(row=2, column=1, sticky="w", padx=5)

        ttk.Label(frm, text="Метод внедрения:").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Radiobutton(frm, text="1. LSB с ключом", variable=self.method_var, value="lsb").grid(row=3, column=1, sticky="w")
        ttk.Radiobutton(frm, text="2. Адаптивный (дисперсия)", variable=self.method_var, value="adaptive").grid(row=4, column=1, sticky="w")

        btn_frame = ttk.Frame(frm)
        btn_frame.grid(row=5, column=0, columnspan=3, pady=10)
        ttk.Button(btn_frame, text="Встроить и извлечь", command=self.run_embed_extract).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Исследование + Экспорт", command=self.run_research_batch).pack(side=tk.LEFT, padx=5)

        self.txt_log = tk.Text(frm, height=18, width=90, state=tk.DISABLED)
        self.txt_log.grid(row=6, column=0, columnspan=3, sticky="nsew")
        frm.grid_rowconfigure(6, weight=1)

        self.log("Программа готова. Выберите файлы и запустите обработку.")
        self.log("Папка results/ создана автоматически для сохранения файлов.")

    def log(self, msg):
        self.txt_log.config(state=tk.NORMAL)
        self.txt_log.insert(tk.END, msg + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state=tk.DISABLED)

    def select_container(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png *.bmp *.jpg *.jpeg")])
        if p: self.container_path.set(p)

    def select_wm(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png *.bmp *.jpg *.jpeg")])
        if p: self.wm_path.set(p)

    def load_grayscale(self, path):
        return np.array(Image.open(path).convert('L'))

    def prepare_watermark(self, container, wm_path):
        container_h, container_w = container.shape
        total_capacity = container_h * container_w
        required_bits = int(total_capacity * 0.5)
        
        wm = Image.open(wm_path).convert('L')
        aspect = wm.width / wm.height
        wm_h = int(math.sqrt(required_bits / aspect))
        wm_w = int(math.sqrt(required_bits * aspect))
        wm = wm.resize((wm_w, wm_h), Image.LANCZOS)
        
        wm_arr = np.array(wm)
        wm_bits = (wm_arr > 128).astype(np.uint8).flatten()
        
        while len(wm_bits) < required_bits:
            wm_bits = np.concatenate([wm_bits, wm_bits])
        wm_bits = wm_bits[:required_bits]
        
        self.last_wm_shape = (wm_h, wm_w)
        return wm_bits

    def embed(self, container, wm_bits, key, method):
        stego = container.copy()
        h, w = container.shape
        flat_container = stego.flatten()
        n_bits = len(wm_bits)

        if method == "lsb":
            rng = np.random.default_rng(key)
            indices = rng.choice(flat_container.size, size=n_bits, replace=False)
            flat_container[indices] = (flat_container[indices] & 254) | wm_bits
        else:
            block_size = 8
            var_map = np.zeros_like(container, dtype=float)
            for i in range(0, h, block_size):
                for j in range(0, w, block_size):
                    block = container[i:i+block_size, j:j+block_size]
                    var_map[i:i+block_size, j:j+block_size] = np.var(block)

            flat_var = var_map.flatten()
            sorted_indices = np.argsort(flat_var)[::-1]
            indices = sorted_indices[:n_bits]
            flat_container[indices] = (flat_container[indices] & 254) | wm_bits

        stego = flat_container.reshape(h, w).astype(np.uint8)

        return stego

    def extract(self, stego, container_shape, method, key=None, wm_len=None):
        h, w = container_shape
        flat_stego = stego.flatten()
        n_bits = wm_len

        if method == "lsb":
            rng = np.random.default_rng(key)
            indices = rng.choice(flat_stego.size, size=n_bits, replace=False)
            return (flat_stego[indices] & 1).astype(np.uint8)
        else:
            block_size = 8
            var_map = np.zeros((h, w), dtype=float)
            for i in range(0, h, block_size):
                for j in range(0, w, block_size):
                    block = stego[i:i+block_size, j:j+block_size]
                    var_map[i:i+block_size, j:j+block_size] = np.var(block)
            
            flat_var = var_map.flatten()
            sorted_indices = np.argsort(flat_var)[::-1]
            indices = sorted_indices[:n_bits]
            return (flat_stego[indices] & 1).astype(np.uint8)

    def calc_psnr(self, orig, stego):
        mse = np.mean((orig.astype(np.float64) - stego.astype(np.float64)) ** 2)
        if mse == 0:
            return 100.0
        return 10 * np.log10((255.0 ** 2) / mse)

    def calc_ci(self, data, alpha=0.05):
        if len(data) < 2:
            return np.mean(data), (0, 0)
        mean = np.mean(data)
        ci = stats.t.interval(1-alpha, len(data)-1, loc=mean, scale=stats.sem(data))
        return mean, ci

    def run_embed_extract(self):
        c_path = self.container_path.get()
        w_path = self.wm_path.get()
        if not c_path or not w_path:
            messagebox.showerror("Ошибка", "Выберите оба файла!")
            return

        self.log("Загрузка изображений...")
        container = self.load_grayscale(c_path)
        wm_bits = self.prepare_watermark(container, w_path)
        key = self.key_var.get()
        method = self.method_var.get()

        self.log(f"Контейнер: {container.shape[1]}x{container.shape[0]} | Ёмкость: {container.size} бит")
        self.log(f"ЦВЗ подготовлен: {len(wm_bits)} бит ({len(wm_bits)/container.size*100:.1f}% ёмкости)")

        stego = self.embed(container, wm_bits, key, method)
        extracted_bits = self.extract(stego, container.shape, method, key, len(wm_bits))
        
        accuracy = np.mean(extracted_bits == wm_bits) * 100
        psnr = self.calc_psnr(container, stego)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stego_name = f"stego_{method}_{ts}.png"
        wm_name = f"extracted_wm_{method}_{ts}.png"
        
        Image.fromarray(stego).save(os.path.join("results", stego_name))
        if self.last_wm_shape:
            wm_img = (extracted_bits.reshape(self.last_wm_shape) * 255).astype(np.uint8)
            Image.fromarray(wm_img).save(os.path.join("results", wm_name))

        self.log(f"PSNR: {psnr:.2f} dB | Точность ЦВЗ: {accuracy:.2f}%")
        self.log(f"Сохранено в results/: {stego_name}, {wm_name}")
        messagebox.showinfo("Готово", f"PSNR: {psnr:.2f} dB\nТочность: {accuracy:.2f}%\nФайлы сохранены в папке results/")

    def run_research_batch(self):
        dir_path = filedialog.askdirectory(title="Выберите папку с набором изображений")
        if not dir_path: return

        images = sorted([os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                         if f.lower().endswith(('.png', '.bmp', '.tiff', '.jpg', '.jpeg'))])
        if len(images) < 5:
            messagebox.showwarning("Внимание", "Для надёжной статистики рекомендуется ≥5 изображений.")
            return

        self.log(f"📦 Пакетная обработка: {len(images)} изображений...")
        results = []
        key = self.key_var.get()

        test_wm_bits = np.random.randint(0, 2, size=1000, dtype=np.uint8)
        self.log(f"Тестовый ЦВЗ: {len(test_wm_bits)} бит")

        processed_count = 0
        skipped_count = 0

        for img_path in images:
            try:
                container = self.load_grayscale(img_path)
                self.log(f"  📄 {os.path.basename(img_path)}: {container.shape[1]}x{container.shape[0]} ({container.size} бит)")

                if container.size < len(test_wm_bits):
                    self.log(f"Пропущено: контейнер слишком мал")
                    skipped_count += 1
                    continue
                
                for method in ["lsb", "adaptive"]:
                    stego = self.embed(container, test_wm_bits, key, method)
                    psnr = self.calc_psnr(container, stego)

                    # Проверка: если PSNR = 100, значит внедрение не сработало
                    if psnr >= 99.9:
                        self.log(f"{method.upper()}: PSNR={psnr:.2f} (возможно, внедрение не сработало)")
                    else:
                        self.log(f"{method.upper()}: PSNR={psnr:.2f} dB")

                    results.append({
                        "file": os.path.basename(img_path),
                        "method": method,
                        "psnr": round(psnr, 3)
                    })
                    processed_count += 1

            except Exception as e:
                self.log(f"Ошибка {os.path.basename(img_path)}: {e}")
                skipped_count += 1

        if not results:
            self.log("Не удалось обработать изображения.")
            return

        self.log(f"\nОбработано: {processed_count}, Пропущено: {skipped_count}")

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join("results", f"psnr_results_{ts}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "method", "psnr"])
            writer.writeheader()
            writer.writerows(results)
        self.log(f"Таблица сохранена: {csv_path}")

        adaptive_psnrs = [r["psnr"] for r in results if r["method"] == "adaptive"]
        if adaptive_psnrs:
            mean, ci = self.calc_ci(adaptive_psnrs)
            self.log("\nИССЛЕДОВАТЕЛЬСКАЯ ЧАСТЬ (α=0.05, Адаптивный метод):")
            self.log(f"   Количество измерений: {len(adaptive_psnrs)}")
            self.log(f"   Среднее PSNR: {mean:.3f} dB")
            if np.isnan(ci[0]) or np.isnan(ci[1]):
                self.log(f"95% Доверительный интервал: не вычислен (все значения одинаковые)")
            else:
                self.log(f"95% Доверительный интервал: [{ci[0]:.3f}, {ci[1]:.3f}] dB")

        # Построение BoxPlot
        plt.figure(figsize=(6, 4))
        lsb_data = [r["psnr"] for r in results if r["method"] == "lsb"]
        plt.boxplot([lsb_data, adaptive_psnrs], labels=["LSB", "Adaptive (Dispersion)"])
        plt.ylabel("PSNR (dB)")
        plt.title(f"Распределение PSNR по методам\n{os.path.basename(dir_path)}")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
    
        plot_path = os.path.join("results", f"boxplot_{ts}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.log(f"График сохранён: {plot_path}")
        self.log("Исследовательская часть завершена.")
        messagebox.showinfo("Готово", "Результаты, CSV и график сохранены в папке results/")

if __name__ == "__main__":
    app = StegoWatermarkLab()
    app.root.mainloop()