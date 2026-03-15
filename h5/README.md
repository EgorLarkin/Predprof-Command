# AlienSignal AI — Классификатор радиосигналов инопланетных цивилизаций

Веб-приложение на Flask + нейронная сеть (2D ResNet-CNN) для распознавания
радиосигналов 20 инопланетных цивилизаций. Система поддерживает
двухролевой доступ (администратор / пользователь), интерактивную аналитику
и загрузку файлов для тестирования модели.

---

## Структура проекта

```
проект/
├── config.py                  # Конфигурация приложения
├── run.py                     # Точка входа Flask
├── requirements.txt           # Зависимости (Windows/Linux)
│
├── app/                       # Flask-приложение
│   ├── __init__.py
│   ├── models.py              # SQLAlchemy-модель пользователя
│   ├── routes.py              # Все маршруты и API-эндпоинты
│   ├── templates/             # HTML-шаблоны (Jinja2 + Bootstrap 5)
│   │   ├── base.html          # Базовый шаблон (navbar, стили)
│   │   ├── login.html         # Страница входа
│   │   ├── admin.html         # Панель администратора
│   │   ├── dashboard.html     # Аналитические графики
│   │   └── profile.html       # Профиль пользователя
│   └── static/
│       ├── css/style.css      # Кастомные стили
│       └── js/main.js         # Chart.js + логика загрузки файлов
│
├── ml/                        # Машинное обучение
│   ├── __init__.py
│   ├── preprocess.py          # Восстановление меток, загрузка данных
│   ├── predict.py             # Инференс: аудио → спектрограмма → класс
│   ├── train.py               # Обучение (Windows/CPU, базовый вариант)
│   ├── model.keras            # Обученная модель (основной формат)
│   ├── model.h5               # Обученная модель (резервный формат)
│   ├── training_history.json  # История обучения (accuracy/loss по эпохам)
│   └── dataset_info.json      # Распределение классов в датасете
│
├── tests/
│   └── test_app.py            # Unit-тесты (pytest)
│
└── training/                  # Скрипты обучения для Mac Apple Silicon
    ├── train_mac.py           # Обучение: log-mel + 2D ResNet (рекомендуется)
    ├── requirements_mac.txt   # tensorflow + tensorflow-metal
    └── setup_and_train.sh     # Скрипт автоустановки и запуска (bash)
```

---

## Как это работает

### 1. Нейронная сеть

**Проблема**: датасет содержит 1200 обучающих и 400 валидационных
аудиосигналов по 80 000 отсчётов (≈5 секунд, 16 кГц). Метки повреждены —
перед названием цивилизации добавлен MD5-хеш (32 символа).

**Восстановление меток** (`ml/preprocess.py`):
```
"a1b2c3...7890Kepler-22b"  →  "Kepler-22b"  →  14
```
Регулярное выражение `^[0-9a-f]{32}(.+)$` отрезает хеш.

**Пайплайн инференса** (`ml/predict.py`):
```
.npz файл  →  (N, 80000, 1) float32
           →  log-mel спектрограмма (N, 157, 128, 1)
           →  2D ResNet-CNN
           →  softmax (N, 20)
           →  название цивилизации
```

**Архитектура модели** (`training/train_mac.py`):

| Блок | Операция | Выходной размер |
|------|----------|-----------------|
| Вход | — | (157, 128, 1) |
| Stem | Conv2D(32, 7×7, stride=2) + MaxPool | (39, 32, 32) |
| Stage 1 | 2× ResBlock(64) | (39, 32, 64) |
| Stage 2 | 2× ResBlock(128, down) | (20, 16, 128) |
| Stage 3 | 2× ResBlock(256, down) | (10, 8, 256) |
| Stage 4 | 2× ResBlock(512, down) | (5, 4, 512) |
| Голова | GAP → Dense(256) → Dropout(0.5) → Dense(20) | (20,) |

**Аугментация** при обучении:
- **SpecAugment** — случайное маскирование частотных и временных полос
- **Mixup** (α=0.4) — интерполяция пар образцов с мягкими метками

**Loss**: `categorical_crossentropy` (с soft-labels от Mixup)

---

### 2. Flask-приложение

#### Роли пользователей

| Роль | Доступные страницы |
|------|--------------------|
| `admin` | Всё + создание/удаление пользователей |
| `user` | Dashboard + Profile |

Первый запуск автоматически создаёт учётную запись `admin` / `admin`.

#### Страницы

**`/login`** — форма входа  
**`/admin`** — панель управления пользователями (только для admin):
  - список всех пользователей с ролями
  - создание нового пользователя
  - удаление пользователя

**`/dashboard`** — аналитика (5 интерактивных графиков):
  1. **Accuracy по эпохам** — train vs validation
  2. **Loss по эпохам** — train vs validation
  3. **Распределение классов** — количество сигналов каждой цивилизации
  4. **Результат теста по образцам** — правильность каждого предсказания
  5. **Топ-5 вероятностей** — уверенность модели для последнего файла

**`/profile`** — профиль пользователя: имя, роль, дата создания

#### API-эндпоинты

| Метод | URL | Описание |
|-------|-----|----------|
| POST | `/api/upload_test` | Загрузить `.npz` файл, получить предсказания |
| GET | `/api/training_history` | История обучения (для графиков) |
| GET | `/api/dataset_info` | Распределение классов |
| GET | `/api/last_test_result` | Последний результат тестирования |
| GET | `/api/user_info` | Информация о текущем пользователе |

---

### 3. База данных

SQLite (`instance/users.db`), управляется через Flask-SQLAlchemy.

Модель `User`:
```
id          INTEGER  PRIMARY KEY
username    TEXT     UNIQUE NOT NULL
password    TEXT     (bcrypt hash)
first_name  TEXT
last_name   TEXT
role        TEXT     ('admin' | 'user')
created_at  DATETIME
```

---

## Установка и запуск

### Шаг 1 — Создать виртуальное окружение

```powershell
cd d:\directory\проект
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Шаг 2 — Запустить сервер

```powershell
python run.py
```

Открыть в браузере: **http://127.0.0.1:5000**

### Шаг 3 — Войти в систему

- Логин: `admin`
- Пароль: `admin`

---

## Обучение модели (Mac Apple Silicon — рекомендуется)

Обучение на сырых аудиосигналах очень медленно на CPU.  
Для нормального результата нужен Mac M-серии с Metal GPU.

### Шаг 1 — Скопировать файлы на Mac

```bash
# Скопировать на Mac:
#   training/train_mac.py
#   training/setup_and_train.sh
#   Data (1).npz       ← исходный датасет
```

### Шаг 2 — Запустить обучение

```bash
cd /путь/к/папке/с/файлами
chmod +x setup_and_train.sh
./setup_and_train.sh
```

Или вручную:
```bash
python3 -m venv venv
source venv/bin/activate
pip install tensorflow tensorflow-metal numpy
python train_mac.py
```

Время обучения на M4 Max: ~30–60 минут.  
Ожидаемая точность: **≥ 60%** на validation set.

### Шаг 3 — Скопировать результаты на Windows

После окончания обучения в папке `output/` появятся 4 файла.  
Скопировать их в `проект/ml/`:

```
output/model.keras           →  ml/model.keras
output/model.h5              →  ml/model.h5
output/training_history.json →  ml/training_history.json
output/dataset_info.json     →  ml/dataset_info.json
```

---

## Тестирование

```powershell
cd d:\directory\проект
venv\Scripts\activate
python -m pytest tests\test_app.py -v
```

### Покрытие тестов

| Класс | Тесты |
|-------|-------|
| `TestPreprocess` | Восстановление меток, преобразование имя↔метка, 20 классов |
| `TestApp` | Страница логина, неверный пароль, доступ без авторизации |
| `TestAdminPanel` | Создание пользователя, удаление, запрет для non-admin |
| `TestModel` | Наличие файла модели, форма выхода (2 образца → 2 предсказания) |

---

## Тестирование модели через веб-интерфейс

1. Войти как любой пользователь
2. Открыть **Dashboard**
3. В разделе «Загрузить файл для тестирования» выбрать `.npz` файл
4. Нажать «Классифицировать»
5. Результат появится на графиках «Результат теста» и «Топ-5 вероятностей»

Формат входного `.npz` файла:
```python
import numpy as np
# Минимальная структура для тестирования:
np.savez('test.npz',
    valid_x=signals,   # shape (N, 80000, 1), dtype float32
    valid_y=labels     # list of str или int
)
```

---

## 20 классов цивилизаций

| № | Название | № | Название |
|---|----------|---|----------|
| 0 | 55_Cancri_Bc | 10 | K2-72e |
| 1 | Gliese_ | 11 | Kepler-155c |
| 2 | Gliese_12_b | 12 | Kepler-174d |
| 3 | Gliese_163_c | 13 | Kepler-186f |
| 4 | HD_20794_d | 14 | Kepler-22b |
| 5 | HD_216520_c | 15 | Kepler-283c |
| 6 | HIP_38594_b | 16 | Kepler-296e |
| 7 | K2-155d | 17 | Kepler-296f |
| 8 | K2-288Bb | 18 | Kepler-62e |
| 9 | K2-332b | 19 | Kepler-62f |

---

## Зависимости

### Windows (`requirements.txt`)
```
flask
flask-sqlalchemy
flask-login
werkzeug
tensorflow
numpy
pytest
```

### Mac (`training/requirements_mac.txt`)
```
tensorflow
tensorflow-metal
numpy
```
