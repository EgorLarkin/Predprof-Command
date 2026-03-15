"""
Нейросеть классификации радиосигналов инопланетных цивилизаций.
Подход: log-mel спектрограмма -> 2D ResNet-CNN.
Оптимизировано для Mac Apple Silicon (Metal GPU).
"""
import numpy as np
import os
import re
import json
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    import keras
    from keras import layers, callbacks
except ImportError:
    print("ОШИБКА: TensorFlow не установлен.")
    print("Запустите: pip install tensorflow tensorflow-metal numpy")
    exit(1)

print(f"TensorFlow: {tf.__version__}")
print(f"Keras: {keras.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU (Metal): {'ДА' if gpus else 'нет, CPU'}")
print()

# ==================== НАСТРОЙКИ ====================

DATA_FILE = 'Data (1).npz'
os.makedirs('output', exist_ok=True)

MODEL_KERAS_PATH = 'output/model.keras'
MODEL_H5_PATH    = 'output/model.h5'
HISTORY_PATH     = 'output/training_history.json'
INFO_PATH        = 'output/dataset_info.json'

EPOCHS     = 120
BATCH_SIZE = 32

CLASS_NAMES = [
    '55_Cancri_Bc', 'Gliese_', 'Gliese_12_b', 'Gliese_163_c',
    'HD_20794_d', 'HD_216520_c', 'HIP_38594_b', 'K2-155d',
    'K2-288Bb', 'K2-332b', 'K2-72e', 'Kepler-155c',
    'Kepler-174d', 'Kepler-186f', 'Kepler-22b', 'Kepler-283c',
    'Kepler-296e', 'Kepler-296f', 'Kepler-62e', 'Kepler-62f'
]
NUM_CLASSES = len(CLASS_NAMES)  # 20


# ==================== ПРЕДОБРАБОТКА ====================

def extract_class_name(corrupted_label: str) -> str:
    """Восстанавливает имя цивилизации из повреждённой метки."""
    match = re.match(r'^[0-9a-f]{32}(.+)$', str(corrupted_label))
    return match.group(1) if match else str(corrupted_label)


def compute_spectrogram_batch(x_batch):
    """
    Конвертирует батч (N, 80000, 1) -> log-mel спектрограммы (N, T, 128, 1).
    """
    signals = tf.squeeze(x_batch, axis=-1)

    frame_length = 1024
    frame_step   = 512
    num_mel_bins = 128

    stfts = tf.signal.stft(signals, frame_length=frame_length,
                            frame_step=frame_step, pad_end=True)
    power = tf.abs(stfts) ** 2

    num_spectrogram_bins = stfts.shape[-1]
    lower_hz, upper_hz, sr = 20.0, 8000.0, 16000.0
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sr, lower_hz, upper_hz
    )
    mel = tf.tensordot(power, mel_weights, axes=1)
    log_mel = tf.math.log(mel + 1e-6)

    mn  = tf.reduce_mean(log_mel, axis=[1, 2], keepdims=True)
    std = tf.math.reduce_std(log_mel, axis=[1, 2], keepdims=True) + 1e-8
    log_mel = (log_mel - mn) / std

    return tf.expand_dims(log_mel, axis=-1)


def load_data():
    print("Загрузка данных из", DATA_FILE)
    data = np.load(DATA_FILE, allow_pickle=True)

    train_x = data['train_x']
    valid_x  = data['valid_x']

    train_y = np.array(
        [CLASS_NAMES.index(extract_class_name(l)) for l in data['train_y']],
        dtype=np.int32
    )
    valid_y = np.array(
        [CLASS_NAMES.index(extract_class_name(l)) for l in data['valid_y']],
        dtype=np.int32
    )

    print(f"  Сигналы train: {train_x.shape}, valid: {valid_x.shape}")
    print("  Вычисление спектрограмм train...", end=' ', flush=True)
    train_spec = compute_spectrogram_batch(tf.constant(train_x, dtype=tf.float32)).numpy()
    print(f"OK -> {train_spec.shape}")
    print("  Вычисление спектрограмм valid...", end=' ', flush=True)
    valid_spec  = compute_spectrogram_batch(tf.constant(valid_x, dtype=tf.float32)).numpy()
    print(f"OK -> {valid_spec.shape}")
    print()
    return train_spec, train_y, valid_spec, valid_y


# ==================== АУГМЕНТАЦИЯ ====================

def augment_specs(specs):
    """SpecAugment: маскирование по частоте и времени."""
    n, t, f, c = specs.shape
    out = specs.copy()
    for i in range(n):
        f0 = np.random.randint(0, max(1, f // 4))
        fw = np.random.randint(0, max(1, f // 8) + 1)
        out[i, :, f0:f0 + fw, :] = 0.0
        t0 = np.random.randint(0, max(1, t // 4))
        tw = np.random.randint(0, max(1, t // 8) + 1)
        out[i, t0:t0 + tw, :, :] = 0.0
    return out


def mixup(x, y_oh, alpha=0.4):
    """Mixup аугментация."""
    n = x.shape[0]
    lam = np.random.beta(alpha, alpha, size=(n, 1, 1, 1)).astype(np.float32)
    idx = np.random.permutation(n)
    x_mix = lam * x + (1 - lam) * x[idx]
    y_mix = lam[:, 0, 0, 0:1] * y_oh + (1 - lam[:, 0, 0, 0:1]) * y_oh[idx]
    return x_mix, y_mix


def data_generator(specs, labels, batch_size, augment=True):
    """Генератор батчей с SpecAugment + Mixup."""
    n = len(specs)
    y_oh = np.eye(NUM_CLASSES, dtype=np.float32)[labels]
    while True:
        idx = np.random.permutation(n)
        for start in range(0, n - batch_size + 1, batch_size):
            bx = specs[idx[start:start + batch_size]].copy()
            by = y_oh[idx[start:start + batch_size]].copy()
            if augment:
                bx = augment_specs(bx)
                if np.random.random() < 0.5:
                    bx, by = mixup(bx, by)
            yield bx, by


# ==================== АРХИТЕКТУРА ====================

def conv_block(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    x = layers.Conv2D(filters, kernel_size, strides=strides,
                      padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def res_block_2d(x, filters, downsample=False):
    strides = (2, 2) if downsample else (1, 1)
    shortcut = x

    x = conv_block(x, filters, strides=strides)
    x = layers.Dropout(0.1)(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=strides,
                                  padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def build_model(input_shape):
    """ResNet-style 2D CNN для log-mel спектрограмм."""
    inp = layers.Input(shape=input_shape)

    # Stem
    x = conv_block(inp, 32, kernel_size=(7, 7), strides=(2, 2))
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Stage 1
    x = res_block_2d(x, 64)
    x = res_block_2d(x, 64)

    # Stage 2
    x = res_block_2d(x, 128, downsample=True)
    x = res_block_2d(x, 128)

    # Stage 3
    x = res_block_2d(x, 256, downsample=True)
    x = res_block_2d(x, 256)

    # Stage 4
    x = res_block_2d(x, 512, downsample=True)
    x = res_block_2d(x, 512)

    # Голова
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ==================== ОБУЧЕНИЕ ====================

def train():
    train_spec, train_y, valid_spec, valid_y = load_data()

    input_shape = train_spec.shape[1:]
    print("Форма спектрограммы:", input_shape)

    print("Построение модели...")
    model = build_model(input_shape)
    model.summary()
    print()

    steps_per_epoch = len(train_spec) // BATCH_SIZE
    valid_y_oh = np.eye(NUM_CLASSES, dtype=np.float32)[valid_y]

    cb_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            MODEL_KERAS_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
    ]

    gen = data_generator(train_spec, train_y, BATCH_SIZE, augment=True)

    print(f"Начало обучения: {EPOCHS} эпох, batch={BATCH_SIZE}, steps={steps_per_epoch}")
    print("=" * 70)

    history = model.fit(
        gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=(valid_spec, valid_y_oh),
        epochs=EPOCHS,
        callbacks=cb_list,
        verbose=1
    )

    model.save(MODEL_H5_PATH)

    history_dict = {
        'accuracy':     [float(v) for v in history.history['accuracy']],
        'val_accuracy': [float(v) for v in history.history['val_accuracy']],
        'loss':         [float(v) for v in history.history['loss']],
        'val_loss':     [float(v) for v in history.history['val_loss']],
    }
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history_dict, f, indent=2)

    train_dist = Counter(train_y.tolist())
    valid_dist = Counter(valid_y.tolist())
    dataset_info = {
        'class_names': CLASS_NAMES,
        'train_distribution': {CLASS_NAMES[k]: v for k, v in sorted(train_dist.items())},
        'valid_distribution': {CLASS_NAMES[k]: v for k, v in sorted(valid_dist.items())},
        'train_total': int(len(train_y)),
        'valid_total': int(len(valid_y)),
    }
    with open(INFO_PATH, 'w') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    val_loss, val_acc = model.evaluate(valid_spec, valid_y_oh, verbose=0)
    print(f"\n{'=' * 70}")
    print(f"  Validation Accuracy : {val_acc * 100:.2f}%")
    print(f"  Validation Loss     : {val_loss:.4f}")
    print(f"{'=' * 70}")
    print()
    print("Скопируйте output/ на Windows:")
    print("  model.keras            -> d:\\directory\\ml\\model.keras")
    print("  model.h5               -> d:\\directory\\ml\\model.h5")
    print("  training_history.json  -> d:\\directory\\ml\\training_history.json")
    print("  dataset_info.json      -> d:\\directory\\ml\\dataset_info.json")


if __name__ == '__main__':
    train()
