import numpy as np
import re
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


def extract_class_name(corrupted_label: str) -> str:
    """Извлекает название цивилизации из повреждённой метки.

    Повреждённые метки имеют формат: <32-символьный MD5-хеш><название цивилизации>
    Например: 'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4Kepler-22b' -> 'Kepler-22b'
    """
    match = re.match(r'^[0-9a-f]{32}(.+)$', str(corrupted_label))
    if match:
        return match.group(1)
    return str(corrupted_label)


def name_to_label(name: str) -> int:
    """Преобразует название цивилизации в числовую метку (0-19)."""
    return Config.CLASS_NAMES.index(name)


def label_to_name(label: int) -> str:
    """Преобразует числовую метку в название цивилизации."""
    return Config.CLASS_NAMES[label]


def load_and_preprocess_data(data_path: str = None):
    """Загружает и предобрабатывает набор данных для обучения и валидации.

    Returns:
        tuple: (train_x, train_y, valid_x, valid_y, class_names)
            train_x: массив аудиосигналов для обучения (1200, 80000, 1)
            train_y: массив числовых меток для обучения (1200,)
            valid_x: массив аудиосигналов для валидации (400, 80000, 1)
            valid_y: массив числовых меток для валидации (400,)
            class_names: список названий цивилизаций
    """
    if data_path is None:
        data_path = Config.DATA_PATH

    data = np.load(data_path, allow_pickle=True)

    train_x = data['train_x']
    valid_x = data['valid_x']

    train_names = [extract_class_name(label) for label in data['train_y']]
    valid_names = [extract_class_name(label) for label in data['valid_y']]

    train_y = np.array([name_to_label(name) for name in train_names], dtype=np.int32)
    valid_y = np.array([name_to_label(name) for name in valid_names], dtype=np.int32)

    return train_x, train_y, valid_x, valid_y, Config.CLASS_NAMES


def load_test_data(test_path: str):
    """Загружает тестовый набор данных.

    Returns:
        tuple: (test_x, test_y) или (test_x, None) если метки отсутствуют
    """
    data = np.load(test_path, allow_pickle=True)
    test_x = data['test_x']
    test_y = None
    if 'test_y' in data:
        test_labels = data['test_y']
        try:
            test_names = [extract_class_name(label) for label in test_labels]
            test_y = np.array([name_to_label(name) for name in test_names], dtype=np.int32)
        except (ValueError, IndexError):
            test_y = np.array(test_labels, dtype=np.int32)
    return test_x, test_y


if __name__ == '__main__':
    train_x, train_y, valid_x, valid_y, class_names = load_and_preprocess_data()
    print(f"Train: {train_x.shape}, labels: {train_y.shape}, unique: {np.unique(train_y)}")
    print(f"Valid: {valid_x.shape}, labels: {valid_y.shape}, unique: {np.unique(valid_y)}")
    print(f"Classes ({len(class_names)}): {class_names}")

    from collections import Counter
    train_counts = Counter(train_y.tolist())
    print("\nРаспределение классов в обучающей выборке:")
    for cls_id, count in sorted(train_counts.items()):
        print(f"  {cls_id:2d} ({class_names[cls_id]:20s}): {count}")
