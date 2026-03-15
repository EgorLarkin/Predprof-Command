import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import keras
from config import Config

_model = None


def compute_spectrogram_batch(x_batch: np.ndarray) -> np.ndarray:
    """Конвертирует сырые аудиосигналы (N, 80000, 1) в log-mel спектрограммы (N, T, 128, 1)."""
    x_tf = tf.constant(x_batch, dtype=tf.float32)
    signals = tf.squeeze(x_tf, axis=-1)

    frame_length = 1024
    frame_step   = 512
    num_mel_bins = 128

    stfts = tf.signal.stft(signals, frame_length=frame_length,
                            frame_step=frame_step, pad_end=True)
    power = tf.abs(stfts) ** 2

    num_spectrogram_bins = stfts.shape[-1]
    mel_weights = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, 16000.0, 20.0, 8000.0
    )
    mel = tf.tensordot(power, mel_weights, axes=1)
    log_mel = tf.math.log(mel + 1e-6)

    mn  = tf.reduce_mean(log_mel, axis=[1, 2], keepdims=True)
    std = tf.math.reduce_std(log_mel, axis=[1, 2], keepdims=True) + 1e-8
    log_mel = (log_mel - mn) / std

    return tf.expand_dims(log_mel, axis=-1).numpy()


def get_model():
    """Загружает обученную модель (singleton). Поддерживает .keras и .h5 форматы."""
    global _model
    if _model is None:
        model_path = None
        for path in [Config.MODEL_PATH, Config.MODEL_PATH_H5]:
            if os.path.exists(path):
                model_path = path
                break
        if model_path is None:
            raise FileNotFoundError(
                f"Модель не найдена. Поместите model.keras или model.h5 в папку ml/"
            )
        _model = keras.models.load_model(model_path)
    return _model


def predict(data: np.ndarray):
    """Предсказание классов для массива сигналов.

    Args:
        data: массив формы (N, 80000, 1)

    Returns:
        dict: {
            'predictions': list of int,
            'class_names': list of str,
            'probabilities': list of list of float
        }
    """
    model = get_model()
    specs = compute_spectrogram_batch(data)
    probabilities = model.predict(specs, verbose=0)
    predicted_classes = np.argmax(probabilities, axis=1).tolist()
    predicted_names = [Config.CLASS_NAMES[c] for c in predicted_classes]

    return {
        'predictions': predicted_classes,
        'class_names': predicted_names,
        'probabilities': probabilities.tolist()
    }


def evaluate(data: np.ndarray, labels: np.ndarray):
    """Оценка модели на тестовых данных.

    Args:
        data: массив формы (N, 80000, 1)
        labels: массив числовых меток (N,)

    Returns:
        dict с accuracy, per-sample результатами и вероятностями.
    """
    model = get_model()
    specs = compute_spectrogram_batch(data)

    num_classes = len(Config.CLASS_NAMES)
    labels_oh = np.eye(num_classes, dtype=np.float32)[labels]

    loss, accuracy = model.evaluate(specs, labels_oh, verbose=0)
    probabilities = model.predict(specs, verbose=0)
    predicted_classes = np.argmax(probabilities, axis=1)

    per_sample_correct = (predicted_classes == labels).astype(int).tolist()
    max_probs = np.max(probabilities, axis=1).tolist()

    return {
        'loss': float(loss),
        'accuracy': float(accuracy),
        'per_sample_accuracy': per_sample_correct,
        'per_sample_confidence': max_probs,
        'predictions': predicted_classes.tolist(),
        'true_labels': labels.tolist(),
        'predicted_names': [Config.CLASS_NAMES[c] for c in predicted_classes],
        'true_names': [Config.CLASS_NAMES[c] for c in labels],
        'probabilities': probabilities.tolist()
    }

