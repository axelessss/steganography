import os
from PIL import Image
import glob

def convert_to_bmp(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Ищем все PNG и TIF/TIFF
    patterns = ['*.png', '*.PNG', '*.tif', '*.TIF', '*.tiff', '*.TIFF']
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(input_dir, pat)))
        
    if not files:
        print(f"В папке {input_dir} не найдено PNG/TIF файлов.")
        return 0
        
    converted = 0
    for f in files:
        try:
            img = Image.open(f)
            
            # 1. Приводим к 8-битному серому (L)
            if img.mode != 'L':
                img = img.convert('L')
                
            # 2. Точный размер 512x512
            img = img.resize((512, 512), Image.LANCZOS)
            
            # 3. Сохраняем в BMP (по умолчанию без сжатия BI_RGB)
            base = os.path.splitext(os.path.basename(f))[0]
            out_path = os.path.join(output_dir, f"{base}.bmp")
            img.save(out_path, format='BMP')
            
            converted += 1
        except Exception as e:
            print(f"⚠️ Ошибка обработки {os.path.basename(f)}: {e}")
            
    print(f"✅ Успешно сконвертировано: {converted} файлов → {output_dir}/")
    return converted

convert_to_bmp("100_imgs", "chest_xrays")
convert_to_bmp("airplane", "airplane_bmp")