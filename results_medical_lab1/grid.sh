#!/usr/bin/env bash
# Скрипт копирует по 8 изображений 5 раз в отдельные директории

set -euo pipefail

# === НАСТРОЙКИ ===
SOURCE_DIR="."          # Директория с исходными изображениями
DEST_BASE="./dest_images"             # Базовая директория, где создадутся группы
IMAGES_PER_GROUP=8                    # Количество изображений в одной группе
GROUP_COUNT=5                         # Количество групп (директорий)
# =================

# Включаем опции для безопасной работы с файлами:
# nullglob: если паттерн не найден, массив останется пустым (не будет "*")
# nocaseglob: игнорировать регистр расширений (.JPG, .png, .WebP и т.д.)
shopt -s nullglob nocaseglob

# Собираем все изображения в массив (добавьте/уберите расширения при необходимости)
images=( "$SOURCE_DIR"/*.{jpg,jpeg,png,gif,bmp,webp,tiff,svg} )
shopt -u nullglob nocaseglob

# Проверка существования исходной директории
if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "❌ Ошибка: Исходная директория не найдена: $SOURCE_DIR" >&2
    exit 1
fi

total_images=${#images[@]}
required_images=$((IMAGES_PER_GROUP * GROUP_COUNT))

if (( total_images < required_images )); then
    echo "❌ Ошибка: Недостаточно изображений. Найдено: $total_images, требуется минимум: $required_images" >&2
    exit 1
fi

# Создаём базовую директорию назначения
mkdir -p "$DEST_BASE"

# Копируем по группам
for ((i=0; i<GROUP_COUNT; i++)); do
    dest_dir="$DEST_BASE/group_$((i+1))"
    mkdir -p "$dest_dir"
    
    start_index=$((i * IMAGES_PER_GROUP))
    chunk=( "${images[@]:start_index:IMAGES_PER_GROUP}" )
    
    echo "📦 Копирование ${#chunk[@]} изображений в $dest_dir..."
    cp -- "${chunk[@]}" "$dest_dir/"
done

echo "✅ Готово! Скопировано $required_images изображений в $GROUP_COUNT директорий."
