# stego_lab.py
import argparse
import numpy as np
from PIL import Image
import math
import os
from skimage.metrics import mean_squared_error, structural_similarity

# ---------- Утилиты ----------
def load_bmp(path):
    img = Image.open(path).convert('L')  # Гарантируем 8-bit grayscale
    if img.size != (512, 512):
        raise ValueError(f"Изображение {path} не 512x512")
    return np.array(img, dtype=np.uint8)

def save_bmp(arr, path):
    Image.fromarray(arr, mode='L').save(path, format='BMP')

# ---------- Режим 1: Извлечение битовой плоскости ----------
def extract_plane(img, k):
    shift = k - 1
    plane = (img >> shift) & 1
    return (plane * 255).astype(np.uint8)

# ---------- Режим 2: Внедрение текста ----------
def embed_message(img_path, text_path, k, out_path):
    img = load_bmp(img_path)
    with open(text_path, 'rb') as f:
        data = f.read()
        
    length = len(data)
    length_bits = f'{length:032b}'  # 4 байта = 32 бита для длины
    data_bits = ''.join(f'{b:08b}' for b in data)
    msg_bits = length_bits + data_bits

    if len(msg_bits) > img.size:
        raise ValueError(f"Сообщение ({len(msg_bits)} бит) не помещается в контейнер ({img.size} пикселей)")

    flat = img.flatten().copy()
    mask_clear = ~(1 << (k-1)) & 0xFF
    for i, bit in enumerate(msg_bits):
        flat[i] = (flat[i] & mask_clear) | (int(bit) << (k-1))
        
    stego = flat.reshape(img.shape).astype(np.uint8)
    save_bmp(stego, out_path)
    print(f"✅ Внедрено {len(msg_bits)} бит ({length} байт) в плоскость k={k}")
    return stego

# ---------- Режим 3: Извлечение текста ----------
def extract_message(img_path, k, out_text_path):
    img = load_bmp(img_path)
    flat = img.flatten()
    bits = ((flat >> (k-1)) & 1).astype(str)
    bit_str = ''.join(bits)

    if len(bit_str) < 32:
        raise ValueError("Изображение слишком маленькое")
    length = int(bit_str[:32], 2)
    msg_bits = bit_str[32:32 + length*8]

    if len(msg_bits) % 8 != 0:
        raise ValueError("Ошибка извлечения: неполный байт")
    bytes_arr = [int(msg_bits[i:i+8], 2) for i in range(0, len(msg_bits), 8)]
    text = bytes(bytes_arr).decode('utf-8', errors='ignore')

    with open(out_text_path, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"✅ Извлечено {len(bytes_arr)} байт → {out_text_path}")

# ---------- Метрики ----------
def calc_metrics(orig_path, stego_path):
    o = load_bmp(orig_path)
    s = load_bmp(stego_path)
    mse_val = mean_squared_error(o, s)
    psnr_val = 20 * math.log10(255.0 / math.sqrt(mse_val)) if mse_val > 0 else float('inf')
    ssim_val = structural_similarity(o, s, data_range=255)
    return mse_val, psnr_val, ssim_val

# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="LSB Стеганография: 8-bit grayscale BMP 512x512")
    sub = parser.add_subparsers(dest='cmd')

    p1 = sub.add_parser('plane', help='Извлечение k-й битовой плоскости')
    p1.add_argument('input', help='Путь к BMP')
    p1.add_argument('-k', type=int, choices=range(1,9), required=True)
    p1.add_argument('-o', '--output', required=True)

    p2 = sub.add_parser('embed', help='Внедрение текста')
    p2.add_argument('input', help='Оригинал BMP')
    p2.add_argument('-k', type=int, choices=range(1,9), required=True)
    p2.add_argument('-t', '--text', required=True, help='Текстовый файл ≥30КБ')
    p2.add_argument('-o', '--output', required=True, help='Стего BMP')

    p3 = sub.add_parser('extract', help='Извлечение текста')
    p3.add_argument('input', help='Стего BMP')
    p3.add_argument('-k', type=int, choices=range(1,9), required=True)
    p3.add_argument('-o', '--output', required=True)

    p4 = sub.add_parser('metric', help='Расчёт MSE / PSNR / SSIM')
    p4.add_argument('orig', help='Оригинал BMP')
    p4.add_argument('stego', help='Стего BMP')

    args = parser.parse_args()

    if args.cmd == 'plane':
        save_bmp(extract_plane(load_bmp(args.input), args.k), args.output)
        print(f"✅ Плоскость k={args.k} сохранена: {args.output}")
    elif args.cmd == 'embed':
        embed_message(args.input, args.text, args.k, args.output)
    elif args.cmd == 'extract':
        extract_message(args.input, args.k, args.output)
    elif args.cmd == 'metric':
        mse, psnr, ssim = calc_metrics(args.orig, args.stego)
        print(f"MSE: {mse:.4f} | PSNR: {psnr:.2f} дБ | SSIM: {ssim:.4f}")

if __name__ == '__main__':
    main()