# Определение занятости парковочного места с использованием ИНС

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-purple)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kamilik/parking-space-classification/blob/main/notebook.ipynb)

## Описание проекта

Данный проект реализует систему автоматического определения занятости парковочного места по фотографии с использованием глубокого обучения. На вход системы подаётся изображение одного парковочного места, на выходе — бинарная классификация:

- **Empty** — место свободно
- **Occupied** — место занято

Вместе с результатом выводится вероятность (confidence) предсказания.

### Ключевые особенности

- Бинарная классификация изображений одного парковочного места
- Transfer learning на основе пяти различных архитектур нейронных сетей
- Датасет PKLot с Kaggle (стратифицированная подвыборка для обучения за разумное время)
- **Возобновляемое обучение**: каждая модель обучается в отдельной ячейке, готовые пропускаются
- **Чекпоинт каждой эпохи сохраняется на Google Drive** — переживает отключение среды Colab
- Автоматическое сравнение всех моделей по набору метрик
- Веб-приложение Streamlit с тёмной темой и историей проверок
- Готовый Google Colab ноутбук

---

## Pipeline (Конвейер обработки)

```
PKLot (Kaggle, kagglehub)
       ↓
Стратифицированная подвыборка (Config.MAX_PER_CLASS)
       ↓
DataLoader (train / val / test = 70 / 15 / 15)
       ↓
Обучение по моделям (5 архитектур × до 20 эпох, EarlyStopping)
   └─ чекпоинт каждой эпохи → Google Drive
       ↓
Сравнение (comparison.csv / xlsx, analysis.txt)
       ↓
Лучшая модель (выбор по F1)
       ↓
Streamlit (веб-приложение)
```

---

## Датасет

### PKLot

**PKLot** (Parking Lot dataset) — один из наиболее полных публичных датасетов для задачи мониторинга парковочных мест. Собран на трёх парковках (PUC, UFPR04, UFPR05) в разных условиях освещения и погоды. Используется версия с Kaggle.

| Характеристика | Значение |
|---|---|
| Источник | [`blanderbuss/parking-lot-dataset`](https://www.kaggle.com/datasets/blanderbuss/parking-lot-dataset) (Kaggle) |
| Загрузчик | `kagglehub` |
| Изображений в архиве | сотни тысяч вырезанных парковочных мест (PKLotSegmented) |
| Классы | Empty, Occupied |
| Формат | JPEG, RGB |
| Разбиение | Автоматическое, стратифицированное: 70% train / 15% val / 15% test |

### Загрузка и подготовка

Датасет скачивается автоматически через `kagglehub` при первом запуске подготовки данных. Для этого нужен **токен Kaggle** — задайте `KAGGLE_USERNAME` и `KAGGLE_KEY` в Colab Secrets (значок замка на левой панели; токен создаётся на kaggle.com → Account → Create New API Token).

Изображения собираются из дерева `PKLotSegmented`: **класс определяется по имени папки-родителя** (`Empty` / `Occupied`) независимо от глубины вложенности архива, поэтому подготовка устойчива к разным вариантам структуры.

### Стратифицированная подвыборка

Полный PKLot очень велик (обучение на нём заняло бы десятки часов на каждую из пяти моделей). Поэтому берётся случайная подвыборка фиксированного размера на класс:

```python
Config.MAX_PER_CLASS = 10000   # ~20 000 изображений всего (2 класса)
# Config.MAX_PER_CLASS = None  # использовать весь датасет
```

Подвыборка детерминирована (зависит только от `seed`), поэтому воспроизводима.

После подготовки данные складываются в структуру `torchvision.datasets.ImageFolder`:

```
data/
├── train/
│   ├── Empty/
│   └── Occupied/
├── val/
│   ├── Empty/
│   └── Occupied/
└── test/
    ├── Empty/
    └── Occupied/
```

Функция `download_and_prepare_dataset()` идемпотентна: если данные уже подготовлены (есть файл-сторож `.pklot_prepared`), шаг пропускается.

---

## Архитектуры

В проекте обучены пять различных архитектур с применением transfer learning. Все модели инициализируются предобученными весами ImageNet-1K (torchvision Weights API), а финальный классификационный слой заменяется на `nn.Linear(in_features, 2)`.

### 1. ResNet18

**Residual Network с 18 слоями.** Лёгкая свёрточная сеть с остаточными связями (skip connections), решающими проблему затухания градиента. Head `fc: Linear(512, 1000)` заменён на `Linear(512, 2)`.

- Параметры: ~11.2 млн
- Размер: ~42.7 МБ
- Сильные стороны: быстрая сходимость, низкая латентность, малый размер
- Применение: сильный baseline для сравнения

### 2. DenseNet121

**Densely Connected Convolutional Networks, 121 слой.** Каждый слой получает feature maps от всех предыдущих — максимальное переиспользование признаков, эффективный градиентный поток. Head `classifier: Linear(1024, 1000)` заменён на `Linear(1024, 2)`.

- Параметры: ~7.9 млн
- Размер: ~30 МБ
- Сильные стороны: высокая точность при малом числе параметров, устойчивость к переобучению

### 3. EfficientNet-B0

**Efficient Neural Architecture с compound scaling.** Балансирует глубину, ширину и разрешение сети по единому коэффициенту масштабирования. Слой `classifier[1]: Linear(1280, 1000)` заменён на `Linear(1280, 2)`.

- Параметры: ~5.3 млн
- Размер: ~20 МБ
- Сильные стороны: превосходное соотношение точность/параметры

### 4. MobileNetV3-Large

**Лёгкая сеть для мобильных устройств.** Использует инвертированные остаточные блоки (inverted residuals), squeeze-and-excitation и hard-swish активацию. Слой `classifier[3]: Linear(1280, 1000)` заменён на `Linear(1280, 2)`.

- Параметры: ~5.5 млн
- Размер: ~21 МБ
- Сильные стороны: максимальная скорость инференса, минимальный размер, пригодна для edge-устройств

### 5. Vision Transformer (ViT-B/16)

**Vision Transformer — архитектура на основе self-attention.** Изображение разбивается на патчи 16×16, каждый патч обрабатывается как токен в трансформере. Head `heads.head: Linear(768, 1000)` заменён на `Linear(768, 2)`.

- Параметры: ~86.6 млн
- Размер: ~330 МБ
- Сильные стороны: глобальный контекст изображения, высокая точность на больших данных
- Особенность: самая медленная и тяжёлая модель — обучается заметно дольше остальных

### Замена головы классификатора

```python
# ResNet18
model.fc = nn.Linear(512, 2)

# DenseNet121
model.classifier = nn.Linear(1024, 2)

# EfficientNet-B0
model.classifier[1] = nn.Linear(1280, 2)

# MobileNetV3-Large
model.classifier[3] = nn.Linear(1280, 2)

# ViT-B/16
model.heads.head = nn.Linear(768, 2)
```

---

## Структура проекта

```
parking-space-classification/
├── config.py          # Центральная конфигурация (Config)
├── dataset.py         # Загрузка PKLot с Kaggle, подвыборка, подготовка
├── models.py          # 5 архитектур нейронных сетей
├── utils.py           # Утилиты: метрики, графики, анализ, колбэки
├── train.py           # Пайплайн обучения (prepare_data / train_one / build_comparison)
├── predict.py         # Модуль инференса
├── app.py             # Streamlit-приложение
├── notebook.ipynb     # Google Colab ноутбук
├── requirements.txt   # Зависимости
├── README.md          # Документация
├── history.json       # История предсказаний (Streamlit)
├── data/              # Подготовленный датасет (ImageFolder) — локально
├── saved_models/      # Лучшие веса каждой модели (.pth)
└── results/           # Метрики, таблицы, графики, анализ
```

### Артефакты на Google Drive

При вызове `Config.enable_gdrive("/content/drive/MyDrive/pklot_project")` результаты сохраняются на Диск и переживают отключение среды Colab:

```
MyDrive/pklot_project/
├── checkpoints/
│   └── <Model>/<Model>_epochNN.pth   # чекпоинт КАЖДОЙ эпохи
├── saved_models/
│   └── <Model>_best.pth              # лучшие веса модели
└── results/
    ├── <Model>_metrics.json          # метрики каждой модели
    ├── comparison.csv / comparison.xlsx
    ├── analysis.txt
    └── *.png                         # графики (loss, accuracy, CM, ROC)
```

> Данные (`data/`) всегда остаются локальными в Colab — чтение обучающих изображений напрямую с Google Drive было бы слишком медленным.

### Описание ключевых файлов

| Файл | Назначение |
|---|---|
| `config.py` | Единый источник констант (`Config`): устройство, пути, гиперпараметры, трансформации, `MAX_PER_CLASS`, `enable_gdrive()`, флаги сохранения чекпоинтов. |
| `dataset.py` | `download_and_prepare_dataset()` (Kaggle + подвыборка + разбиение), `get_data_loaders()`. |
| `models.py` | `get_model(name, num_classes)` — модель с изменённой головой; `get_model_info()` — число параметров и размер. |
| `utils.py` | `EarlyStopping`, `ModelCheckpoint`, `compute_metrics()`, `plot_training_curves()`, `plot_confusion_matrix()`, `plot_roc_curve()`, `save_comparison_table()`, `analyze_results()`. |
| `train.py` | `prepare_data()`, `train_one()` (одна модель, возобновляемо), `build_comparison()`, `train_model()`, `run_full_pipeline()`. |
| `predict.py` | `ParkingPredictor` — класс для инференса; `predict_image()` — функциональный API; поддержка CLI. |
| `app.py` | Streamlit-приложение: загрузка изображения, классификация, история, статистика, боковая панель. |
| `notebook.ipynb` | Colab-ноутбук: окружение → модули → Google Drive → подготовка данных → обучение по ячейкам → сравнение → графики → тест. |

---

## Установка

### Требования

- Python 3.11 или выше
- pip
- CUDA-совместимый GPU (для обучения — обязательно; для инференса достаточно CPU)

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Содержимое `requirements.txt`

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
streamlit>=1.28.0
openpyxl>=3.1.0
Pillow>=10.0.0
tqdm>=4.65.0
kagglehub>=0.3.0
```

---

## Обучение

### Вариант 1: Google Colab (рекомендуется)

1. Откройте `notebook.ipynb` в Google Colab.
2. **Среда выполнения → Сменить среду выполнения → GPU (T4)**.
3. Добавьте в **Secrets** (значок замка) `KAGGLE_USERNAME` и `KAGGLE_KEY` — для скачивания датасета.
4. Выполняйте ячейки сверху вниз.

Порядок ячеек ноутбука:

| Раздел | Что делает |
|---|---|
| Подготовка окружения | установка недостающих пакетов, самопроверка/починка torch, проверка GPU |
| Исходный код модулей | `%%writefile`-ячейки записывают `.py`-файлы проекта |
| Google Drive | монтирование Диска и `Config.enable_gdrive(...)` (запросит авторизацию) |
| Подготовка данных | `prepare_data()` — скачивание, подвыборка, разбиение |
| Обучение моделей | **пять отдельных ячеек** — `train_one("ResNet18")`, …, `train_one("ViT-B/16")` |
| Результаты и сравнение | `build_comparison()` — таблицы и анализ |
| Графики / Тестирование | отображение графиков и предсказание на примере |

**Возобновляемость.** Каждая модель обучается в своей ячейке и сохраняет результаты на диск/Drive. Если Colab отключится — перезапустите только незавершённую модель: уже обученные пропускаются автоматически. Чекпоинт каждой эпохи лежит на Google Drive.

### Вариант 2: Локальный запуск

```bash
python train.py
```

Скрипт запускает полный пайплайн `run_full_pipeline()` — эквивалент последовательного прогона всех моделей (с учётом `Config.MAX_PER_CLASS`).

### Параметры обучения

| Параметр | Значение |
|---|---|
| Оптимизатор | Adam |
| Функция потерь | CrossEntropyLoss |
| Learning rate | 0.0001 |
| Batch size | 32 |
| Epochs | 20 (с ранней остановкой) |
| EarlyStopping patience | 5 |
| Image size | 224 × 224 |
| Random seed | 42 |
| MAX_PER_CLASS | 10000 (≈ 20 000 изображений всего) |
| SAVE_EVERY_EPOCH | True (чекпоинт каждой эпохи) |

Для каждой модели сохраняется чекпоинт с наилучшими весами по F1-мере на валидации, а также веса после каждой эпохи.

---

## Google Drive — сохранение чекпоинтов

Чтобы результаты пережили отключение или удаление среды выполнения Colab, они сохраняются на Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")

from config import Config
Config.enable_gdrive("/content/drive/MyDrive/pklot_project")

Config.SAVE_EVERY_EPOCH = True
Config.EPOCH_CKPT_KEEP_LAST = None   # None = все эпохи; напр. 3 = только 3 последних
```

- Сохраняются: **чекпоинт каждой эпохи**, лучшие модели, метрики, таблицы и графики.
- `EPOCH_CKPT_KEEP_LAST` ограничивает число хранимых чекпоинтов на модель (экономия места).
- **Место:** хранение всех эпох для пяти моделей ≈ 8–10 ГБ (в основном ViT-B/16). При нехватке места задайте `EPOCH_CKPT_KEEP_LAST = 3`.

---

## Запуск Streamlit

```bash
streamlit run app.py
```

Приложение откроется в браузере по адресу `http://localhost:8501`.

### Возможности интерфейса

- **Выбор модели** — из доступных обученных чекпоинтов в `saved_models/`
- **Загрузка изображения** — JPG, JPEG, PNG
- **Просмотр изображения** — предпросмотр загруженного снимка
- **Кнопка «Определить»** — запуск классификации
- **Результат** — метка (Empty / Occupied) и вероятность в процентах
- **История проверок** — таблица предыдущих предсказаний с датой, временем и уверенностью
- **Статистика в боковой панели** — общее число проверок, количество Empty и Occupied, средняя уверенность, последние проверки
- **Кнопка очистки истории** — сброс всех записей из `history.json`
- **Тёмная современная тема** — кастомный CSS, стилизованные карточки результатов

История сохраняется в `history.json`:

```json
[
  {
    "date": "2026-07-03",
    "time": "14:32:07",
    "filename": "spot_001.jpg",
    "result": "Occupied",
    "confidence": 0.9732,
    "model": "ResNet18"
  }
]
```

---

## Метрики

Для каждой из пяти архитектур вычисляются метрики на тестовой выборке:

| Метрика | Описание |
|---|---|
| **Accuracy** | Доля правильно классифицированных изображений |
| **Precision** | Доля истинно положительных среди предсказанных положительных |
| **Recall** | Доля истинно положительных среди всех реальных положительных |
| **F1-score** | Гармоническое среднее Precision и Recall |
| **ROC AUC** | Площадь под ROC-кривой — разделимость классов |
| **Confusion Matrix** | Матрица ошибок: TP, FP, FN, TN |
| **Inference Time** | Среднее время инференса одного изображения (мс) |
| **Parameters** | Общее число параметров модели |
| **Model Size** | Размер чекпоинта в МБ |
| **Epochs** | Фактическое число эпох (с учётом EarlyStopping) |
| **Training Time** | Суммарное время обучения (с) |

### Графики

Для каждой архитектуры сохраняются PNG в `results/` (или на Google Drive):

- `<Model>_loss.png` — кривые потерь (train / val) по эпохам
- `<Model>_accuracy.png` — кривые точности (train / val) по эпохам
- `<Model>_confusion_matrix.png` — матрица ошибок на тестовой выборке
- `<Model>_roc_curve.png` — ROC-кривая со значением AUC

### Сравнительная таблица

После обучения `build_comparison()` формирует таблицу со столбцами:

```
Architecture | Accuracy | Precision | Recall | F1 | ROC AUC |
Inference Time | Parameters | Model Size | Epochs | Training Time | Best
```

Лучшая модель по F1 отмечается в столбце `Best`. Файлы:

```
results/comparison.csv
results/comparison.xlsx
results/analysis.txt
```

---

## Примеры работы

### Python API

```python
from predict import ParkingPredictor

# Загрузить обученную модель (один раз)
predictor = ParkingPredictor(
    model_path="saved_models/ResNet18_best.pth",
    model_name="ResNet18",
)

# Классифицировать изображение
result = predictor.predict("path/to/parking_spot.jpg")

print(result.label)             # "Empty" или "Occupied"
print(result.confidence)        # 0.9732
print(result.probabilities)     # {"Empty": 0.0268, "Occupied": 0.9732}
print(result.inference_time_ms) # 4.7
```

> Имя файла чекпоинта строится из имени модели заменой `/` и `-` на `_`.
> Например, `ViT-B/16` → `ViT_B_16_best.pth`, `EfficientNet-B0` → `EfficientNet_B0_best.pth`.

### Командная строка (CLI)

```bash
python predict.py \
    --model-path saved_models/ResNet18_best.pth \
    --model-name ResNet18 \
    --image path/to/spot.jpg
```

Пример вывода:

```
------------------------------------------------
  Image      : path/to/spot.jpg
  Model      : ResNet18
  Label      : Occupied
  Confidence : 0.9732 (97.32 %)
  Probabilities:
    Empty        0.0268
    Occupied     0.9732
  Latency    : 4.70 ms
------------------------------------------------
```

### Использование нескольких моделей

```python
from predict import ParkingPredictor
from config import Config

image_path = "test_spot.jpg"

for model_name in Config.MODEL_NAMES:
    safe = model_name.replace("/", "_").replace("-", "_")
    model_path = Config.SAVED_MODELS_DIR / f"{safe}_best.pth"
    if not model_path.exists():
        print(f"{model_name}: чекпоинт не найден, пропускаем.")
        continue
    predictor = ParkingPredictor(model_path=model_path, model_name=model_name)
    result = predictor.predict(image_path)
    print(f"{model_name:20s} -> {result.label} ({result.confidence:.4f})")
```

### Интерфейс Streamlit

1. В боковой панели — статистика всех проверок и выбор модели.
2. На главной странице — загрузчик изображения, предпросмотр, кнопка «Определить».
3. После классификации — карточка с результатом (Empty / Occupied) и уровнем уверенности.
4. Ниже — таблица истории всех проверок.
5. Кнопка «Очистить историю» сбрасывает все записи.

---

## Аугментация данных

### Обучение (train)

```
RandomResizedCrop(224, scale=(0.8, 1.0))
RandomHorizontalFlip()
RandomRotation(degrees=15)
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

### Валидация и тест (val / test)

```
Resize(256)
CenterCrop(224)
ToTensor()
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

---

## Технологии

| Технология | Версия | Назначение |
|---|---|---|
| Python | 3.11+ | Основной язык разработки |
| PyTorch | >= 2.0.0 | Фреймворк глубокого обучения |
| torchvision | >= 0.15.0 | Предобученные модели, трансформации, ImageFolder |
| kagglehub | >= 0.3.0 | Загрузка датасета PKLot с Kaggle |
| scikit-learn | >= 1.3.0 | Метрики классификации, разбиение данных |
| Streamlit | >= 1.28.0 | Веб-интерфейс приложения |
| Pandas | >= 2.0.0 | Таблицы результатов и экспорт |
| Matplotlib | >= 3.7.0 | Построение графиков |
| Pillow | >= 10.0.0 | Загрузка и обработка изображений |
| NumPy | >= 1.24.0 | Численные вычисления |
| openpyxl | >= 3.1.0 | Экспорт таблицы сравнения в XLSX |
| tqdm | >= 4.65.0 | Прогресс-бары в цикле обучения |

---

## Результаты

После обучения артефакты сохраняются:

| Директория / файл | Содержимое |
|---|---|
| `saved_models/<Model>_best.pth` | Лучшие веса каждой из пяти моделей |
| `checkpoints/<Model>/<Model>_epochNN.pth` | Чекпоинт каждой эпохи (на Google Drive) |
| `results/<Model>_metrics.json` | Метрики каждой модели |
| `results/*.png` | Графики: кривые обучения, матрицы ошибок, ROC |
| `results/comparison.csv` | Сравнительная таблица метрик (CSV) |
| `results/comparison.xlsx` | Сравнительная таблица (Excel, с выделением лучшей) |
| `results/analysis.txt` | Текстовый анализ: сильные/слабые стороны, выбор лучшей модели |
| `history.json` | История предсказаний Streamlit-приложения |

Лучшая модель по F1 автоматически отмечается в таблице и рекомендуется для использования в приложении Streamlit.

> На PKLot задача сравнительно «лёгкая» для CNN + transfer learning: все модели, как правило, достигают 98–99%+ точности. Поэтому итоговое сравнение различает их прежде всего по **скорости инференса, размеру и времени обучения**.

---

## Воспроизводимость

Все случайные процессы фиксируются через `seed_everything(42)`:

- `random.seed(42)`
- `numpy.random.seed(42)`
- `torch.manual_seed(42)`
- `torch.cuda.manual_seed_all(42)`
- `torch.backends.cudnn.deterministic = True`
- `PYTHONHASHSEED=42`

Стратифицированная подвыборка данных также детерминирована (зависит только от `seed`).

---

## Авторы

Проект разработан в рамках производственной практики.

**Тема:** Определение занятости парковочного места с использованием искусственных нейронных сетей.
