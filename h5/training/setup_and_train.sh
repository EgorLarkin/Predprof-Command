#!/bin/bash
# Установка зависимостей и запуск обучения (Mac M-серия)
set -e

echo "=== AlienSignal AI — Обучение на Mac M-серия ==="
echo ""

# Проверка Python
if ! command -v python3 &>/dev/null; then
    echo "ОШИБКА: python3 не найден. Установите с https://www.python.org"
    exit 1
fi

PYTHON=$(command -v python3)
echo "Python: $($PYTHON --version)"
echo ""

# Виртуальное окружение
if [ ! -d "venv" ]; then
    echo "Создание виртуального окружения..."
    $PYTHON -m venv venv
fi

source venv/bin/activate
echo "venv активирован: $(which python)"
echo ""

# Установка пакетов
echo "Установка зависимостей..."
pip install --upgrade pip -q
pip install tensorflow tensorflow-metal numpy -q
echo "Зависимости установлены."
echo ""

# Проверка файла данных
if [ ! -f "Data (1).npz" ]; then
    echo "ОШИБКА: файл 'Data (1).npz' не найден в этой папке!"
    echo "Скопируйте его сюда и запустите снова."
    exit 1
fi

echo "Файл данных найден: $(du -h 'Data (1).npz' | cut -f1)"
echo ""

# Запуск обучения
echo "Запуск обучения..."
echo ""
python train.py

echo ""
echo "Готово! Скопируйте содержимое папки output/ на Windows."
