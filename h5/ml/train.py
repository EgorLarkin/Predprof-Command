"""
Обучение нейросети для классификации радиосигналов инопланетных цивилизаций.
"""
import numpy as np
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import keras
from keras import layers, callbacks

from ml.preprocess import load_and_preprocess_data
from config import Config


def build_model(input_shape=(80000, 1), num_classes=20):
    """Создаёт модель 1D-CNN для классификации аудиосигналов.

    Архитектура:
    - 4 блока свёрточных слоёв с BatchNorm, MaxPooling и Dropout
    - GlobalAveragePooling
    - Dense-голова с Dropout
    """
    model = keras.Sequential([
        # Input
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv1D(32, kernel_size=64, strides=4, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=4),
        layers.Dropout(0.2),

        # Block 2
        layers.Conv1D(64, kernel_size=32, strides=2, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=4),
        layers.Dropout(0.2),

        # Block 3
        layers.Conv1D(128, kernel_size=16, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=4),
        layers.Dropout(0.3),

        # Block 4
        layers.Conv1D(256, kernel_size=8, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),

        # Classification head
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(epochs=50, batch_size=32, data_path=None):
    """Обучает модель и сохраняет результаты.

    Returns:
        tuple: (model, history_dict)
    """
    print("=" * 60)
    print("Загрузка и предобработка данных...")
    print("=" * 60)

    train_x, train_y, valid_x, valid_y, class_names = load_and_preprocess_data(data_path)

    print(f"Обучающая выборка: {train_x.shape}")
    print(f"Валидационная выборка: {valid_x.shape}")
    print(f"Количество классов: {len(class_names)}")
    print()

    print("=" * 60)
    print("Построение модели...")
    print("=" * 60)

    model = build_model(
        input_shape=train_x.shape[1:],
        num_classes=len(class_names)
    )
    model.summary()
    print()

    # Callbacks
    model_dir = os.path.dirname(Config.MODEL_PATH)
    os.makedirs(model_dir, exist_ok=True)

    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            Config.MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    print("=" * 60)
    print(f"Обучение модели ({epochs} эпох)...")
    print("=" * 60)

    history = model.fit(
        train_x, train_y,
        validation_data=(valid_x, valid_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1
    )

    # Сохранение истории обучения
    history_dict = {
        'accuracy': [float(v) for v in history.history['accuracy']],
        'val_accuracy': [float(v) for v in history.history['val_accuracy']],
        'loss': [float(v) for v in history.history['loss']],
        'val_loss': [float(v) for v in history.history['val_loss']],
    }

    history_path = os.path.join(model_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"\nИстория обучения сохранена: {history_path}")

    # Сохранение распределения классов
    from collections import Counter
    train_dist = Counter(train_y.tolist())
    valid_dist = Counter(valid_y.tolist())

    dataset_info = {
        'class_names': class_names,
        'train_distribution': {class_names[k]: v for k, v in sorted(train_dist.items())},
        'valid_distribution': {class_names[k]: v for k, v in sorted(valid_dist.items())},
        'train_total': len(train_y),
        'valid_total': len(valid_y),
    }

    info_path = os.path.join(model_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    print(f"Информация о данных сохранена: {info_path}")

    # Финальная оценка
    print("\n" + "=" * 60)
    print("Финальная оценка модели:")
    print("=" * 60)
    val_loss, val_acc = model.evaluate(valid_x, valid_y, verbose=0)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

    print(f"\nМодель сохранена: {Config.MODEL_PATH}")

    return model, history_dict


if __name__ == '__main__':
    train_model(epochs=50, batch_size=32)
