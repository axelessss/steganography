from PIL import Image, ImageDraw, ImageFont

# Создаём изображение с текстом
img = Image.new('L', (200, 100), color='white')
draw = ImageDraw.Draw(img)
draw.text((50, 30), "WATERMARK", fill='black')
img.save("watermark.png")
print("Водяной знак создан: watermark.png")