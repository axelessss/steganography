import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def plot_intensity_histogram(image_path: str):
    """
    Строит гистограмму распределения яркости для 8-битного изображения.
    """
    # Загружаем и принудительно приводим к 8-bit grayscale (L)
    img = Image.open(image_path).convert('L')
    pixels = np.array(img).flatten()
    
    plt.figure(figsize=(10, 5))
    plt.hist(pixels, bins=256, range=(0, 256), 
             color='black', alpha=0.6, edgecolor='gray')
    
    plt.title(f'Гистограмма яркости: {image_path}')
    plt.xlabel('Уровень яркости (0–255)')
    plt.ylabel('Количество пикселей')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.xlim(0, 255)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Гистограмма яркости изображения')
    parser.add_argument('image_file', type=str, help='Путь к файлу изображения (BMP, PNG и др.)')
    args = parser.parse_args()
    
    plot_intensity_histogram(args.image_file)