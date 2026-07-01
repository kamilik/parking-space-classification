# Определение занятости парковочного места с использованием ИНС

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-purple)

## Описание проекта

Данный проект реализует систему автоматического определения занятости парковочного места по фотографии с использованием глубокого обучения. На вход системы подаётся изображение одного парковочного места, на выходе — бинарная классификация:

- **Empty** — место свободно
- **Occupied** — место занято

Вместе с результатом выводится вероятность (confidence) предсказания.

### Ключевые особенности

- Бинарная классификация изображений одного парковочного места
- Transfer learning на основе пяти различных архитектур нейронных сетей
- Датасет PKLot (более 695 000 изображений парковочных мест)
- Автоматическое сравнение всех моделей по набору метрик
- Веб-приложение Streamlit с тёмной темой и историей проверок
- Готовый Google Colab ноутбук, запускающийся сверху вниз без изменений

---

## Pipeline (Конвейер обработки)

```
PKLot (FiftyOne)
       ↓
  DataLoader (train / val / test)
       ↓
  Training (5 архитектур × 20 эпох)
       ↓
  Comparison (таблица метрик)
       ↓
  Best Model (выбор лучшей по F1)
       ↓
  Streamlit (веб-приложение)
```

---

## Датасет

### PKLot

**PKLot** (Parking Lot dataset) — один из наиболее полных публичных датасетов для задачи мониторинга парковочных мест. Датасет собран на трёх парковках в разных условиях освещения и погоды.

| Характеристика | Значение |
|---|---|
| Источник | [Voxel51/PKLot на HuggingFace](https://huggingface.co/datasets/Voxel51/PKLot) |
| Изображений | > 695 000 вырезанных парковочных мест |
| Классы | Empty, Occupied |
| Формат | JPEG, RGB |
| Разбиение | Автоматическое: 70% train / 15% val / 15% test |

### Загрузка датасета

Датасет загружается автоматически через библиотеку FiftyOne при первом запуске `train.py` или ноутбука:

```python
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("pklot")
```

После загрузки датасет автоматически конвертируется в структуру `torchvision.datasets.ImageFolder`:

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

Функция `download_and_prepare_dataset()` идемпотентна: если данные уже скачаны и подготовлены, шаг пропускается автоматически.

---

## Архитектуры

В проекте обучены пять различных архитектур с применением transfer learning. Все модели инициализируются предобученными весами ImageNet-1K (torchvision Weights API), а финальный классификационный слой заменяется на `nn.Linear(in_features, 2)`.

### 1. ResNet18

**Residual Network с 18 слоями.** Лёгкая сверточная сеть с остаточными связями (skip connections), решающими проблему затухания градиента. Оригинальный head `fc: Linear(512, 1000)` заменён на `Linear(512, 2)`.

- Параметры: ~11.2 млн
- Размер: ~42.6 МБ
- Сильные стороны: быстрая сходимость, низкая латентность, малый размер
- Применение: сильный baseline для сравнения

### 2. DenseNet121

**Densely Connected Convolutional Networks, 121 слоёв.** Каждый слой получает feature maps от всех предыдущих слоёв — максимальное переиспользование признаков, эффективный градиентный поток. Head `classifier: Linear(1024, 1000)` заменён на `Linear(1024, 2)`.

- Параметры: ~7.9 млн
- Размер: ~30.4 МБ
- Сильные стороны: высокая точность при малом числе параметров, устойчивость к переобучению

### 3. EfficientNet-B0

**Efficient Neural Architecture с compound scaling.** Балансирует глубину, ширину и разрешение сети по единому коэффициенту масштабирования. Последний линейный слой `classifier[1]: Linear(1280, 1000)` заменён на `Linear(1280, 2)`.

- Параметры: ~5.3 млн
- Размер: ~20.2 МБ
- Сильные стороны: превосходное соотношение точность/параметры

### 4. MobileNetV3-Large

**Лёгкая сеть для мобильных устройств.** Использует инвертированные остаточные блоки (inverted residuals), squeeze-and-excitation и hard-swish активацию. Слой `classifier[3]: Linear(1280, 1000)` заменён на `Linear(1280, 2)`.

- Параметры: ~5.5 млн
- Размер: ~21.0 МБ
- Сильные стороны: максимальная скорость инференса, минимальный размер, пригодна для edge-устройств

### 5. Vision Transformer (ViT-B/16)

**Vision Transformer — архитектура на основе self-attention.** Изображение разбивается на патчи 16×16, каждый патч обрабатывается как токен в трансформере. Head `heads.head: Linear(768, 1000)` заменён на `Linear(768, 2)`.

- Параметры: ~86.6 млн
- Размер: ~330.4 МБ
- Сильные стороны: глобальный контекст изображения, высокая точность на больших данных

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

## Сравнение архитектур

| Архитектура | Параметры | Размер (MB) | Описание |
|---|---|---|---|
| ResNet18 | ~11.2M | ~42.6 | Лёгкая CNN с residual connections |
| DenseNet121 | ~7.0M | ~26.5 | Dense connections, переиспользование признаков |
| EfficientNet-B0 | ~4.0M | ~15.3 | Compound scaling, высокая эффективность |
| MobileNetV3-Large | ~4.2M | ~16.0 | Мобильная архитектура, depthwise separable conv |
| ViT-B/16 | ~85.8M | ~327.3 | Vision Transformer, self-attention |

---

## Структура проекта

```
project/
├── config.py          # Центральная конфигурация
├── dataset.py         # Загрузка и подготовка PKLot
├── models.py          # 5 архитектур нейронных сетей
├── utils.py           # Утилиты: метрики, графики, анализ
├── train.py           # Пайплайн обучения
├── predict.py         # Модуль инференса
├── app.py             # Streamlit-приложение
├── notebook.ipynb     # Google Colab ноутбук
├── requirements.txt   # Зависимости
├── README.md          # Документация
├── history.json       # История предсказаний (Streamlit)
├── models/            # Подготовленный датасет
├── saved_models/      # Обученные модели (.pth)
├── results/           # Результаты экспериментов
│   └── YYYY-MM-DD_HH-MM-SS/  # Каждый запуск сохраняется отдельно
│       ├── comparison.csv
│       ├── comparison.xlsx
│       ├── metrics.json
│       ├── analysis.txt
│       └── *.png (графики)
└── plots/             # (устаревшая) директория графиков
```

### Описание ключевых файлов

| Файл | Назначение |
|---|---|
| `config.py` | Единый источник констант (`Config`). Устройство, пути, гиперпараметры, трансформации, имена классов. |
| `dataset.py` | `download_and_prepare_dataset()` и `get_data_loaders()`. Автоматическое разбиение 70/15/15. |
| `models.py` | `get_model(name, num_classes)` — возвращает модель с изменённой головой. `get_model_info()` — число параметров и размер. |
| `utils.py` | `EarlyStopping`, `ModelCheckpoint`, `compute_metrics()`, `plot_training_curves()`, `plot_confusion_matrix()`, `plot_roc_curve()`, `save_comparison_table()`, `analyze_results()`. |
| `train.py` | `train_one_epoch()`, `evaluate()`, `train_model()`, `run_full_pipeline()`. Запускается как скрипт. |
| `predict.py` | `ParkingPredictor` — высокоуровневый класс для инференса. `predict_image()` — функциональный API. Поддерживает CLI. |
| `app.py` | Streamlit-приложение: загрузка изображения, классификация, история, статистика, боковая панель. |
| `notebook.ipynb` | Colab-ноутбук: установка зависимостей → скачивание датасета → обучение → графики → таблицы. |

---

## Установка

### Требования

- Python 3.11 или выше
- pip
- CUDA-совместимый GPU (рекомендуется) или CPU

### Установка зависимостей

```bash
cd project
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
opencv-python>=4.8.0
streamlit>=1.28.0
fiftyone>=0.23.0
openpyxl>=3.1.0
Pillow>=10.0.0
tqdm>=4.65.0
```

---

## Обучение

### Вариант 1: Google Colab (рекомендуется)

1. Откройте файл `notebook.ipynb` в Google Colab.
2. В меню выберите **Среда выполнения → Изменить тип среды выполнения → GPU (T4)**.
3. Нажмите **Среда выполнения → Выполнить всё** (Run All).

Ноутбук автоматически:
- устанавливает все зависимости,
- загружает датасет PKLot через FiftyOne,
- конвертирует его в формат ImageFolder,
- обучает все пять моделей последовательно,
- строит и сохраняет все графики,
- сохраняет чекпоинты в `saved_models/`,
- формирует `results/comparison.csv` и `results/comparison.xlsx`,
- выполняет текстовый анализ результатов.

### Вариант 2: Локальный запуск

```bash
cd project
python train.py
```

Скрипт запускает полный пайплайн — эквивалентен всем ячейкам ноутбука.

### Параметры обучения

| Параметр | Значение |
|---|---|
| Оптимизатор | Adam |
| Функция потерь | CrossEntropyLoss |
| Learning rate | 0.0001 |
| Batch size | 32 |
| Epochs | 20 |
| EarlyStopping patience | 5 |
| Image size | 224 × 224 |
| Random seed | 42 |

Для каждой модели сохраняется чекпоинт с наилучшими весами по валидационной потере.

---

## Запуск Streamlit

```bash
cd project
streamlit run app.py
```

Приложение откроется в браузере по адресу `http://localhost:8501`.

### Возможности интерфейса

- **Выбор модели** — выпадающий список из доступных обученных чекпоинтов в `saved_models/`
- **Загрузка изображения** — поддержка JPG, JPEG, PNG
- **Просмотр изображения** — предпросмотр загруженного снимка
- **Кнопка "Определить"** — запуск классификации
- **Результат** — метка (Empty / Occupied) и вероятность в процентах
- **История проверок** — таблица всех предыдущих предсказаний с датой, временем и уверенностью
- **Статистика в боковой панели** — общее число проверок, количество Empty и Occupied, средняя уверенность, последние 10 проверок
- **Кнопка очистки истории** — сброс всех записей из `history.json`
- **Тёмная современная тема** — кастомный CSS, стилизованные карточки результатов

История сохраняется в `history.json` в формате:

```json
[
  {
    "date": "2026-06-30",
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

Для каждой из пяти архитектур вычисляются следующие метрики на тестовой выборке:

| Метрика | Описание |
|---|---|
| **Accuracy** | Доля правильно классифицированных изображений |
| **Precision** | Точность: доля истинно положительных среди предсказанных положительных |
| **Recall** | Полнота: доля истинно положительных среди всех реальных положительных |
| **F1-score** | Гармоническое среднее Precision и Recall |
| **ROC AUC** | Площадь под кривой ROC, характеризует разделимость классов |
| **Confusion Matrix** | Матрица ошибок: TP, FP, FN, TN |
| **Inference Time** | Среднее время инференса одного изображения (мс) |
| **Parameters** | Общее число параметров модели |
| **Model Size** | Размер чекпоинта в МБ |
| **Epochs** | Фактическое число эпох (с учётом EarlyStopping) |
| **Training Time** | Суммарное время обучения (минуты) |

### Графики (сохраняются в `plots/`)

Для каждой архитектуры строятся:

- `<Model>_training_curves.png` — кривые потерь и точности по эпохам (train / val)
- `<Model>_confusion_matrix.png` — матрица ошибок на тестовой выборке
- `<Model>_roc_curve.png` — ROC-кривая с значением AUC

### Сравнительная таблица

После обучения всех моделей автоматически формируется таблица со столбцами:

```
Architecture | Accuracy | Precision | Recall | F1 | ROC AUC |
Inference Time | Parameters | Model Size | Epochs | Training Time
```

Лучшая модель выделяется. Файлы сохраняются в:

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

### Полный пример с выводом всех деталей

```python
from predict import ParkingPredictor
from config import Config

predictor = ParkingPredictor(
    model_path=Config.SAVED_MODELS_DIR / "EfficientNet-B0_best.pth",
    model_name="EfficientNet-B0",
)

result = predictor.predict("spot_007.jpg")

print(f"Результат:    {result.label}")
print(f"Уверенность:  {result.confidence * 100:.2f} %")
print(f"Латентность:  {result.inference_time_ms:.2f} мс")
for cls, prob in result.probabilities.items():
    print(f"  {cls}: {prob:.4f}")
```

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
    pth_name = model_name.replace("/", "")
    model_path = Config.SAVED_MODELS_DIR / f"{pth_name}_best.pth"
    if not model_path.exists():
        print(f"{model_name}: чекпоинт не найден, пропускаем.")
        continue
    predictor = ParkingPredictor(model_path=model_path, model_name=model_name)
    result = predictor.predict(image_path)
    print(f"{model_name:20s} -> {result.label} ({result.confidence:.4f})")
```

### Описание интерфейса Streamlit

После запуска `streamlit run app.py` пользователь получает интерактивный веб-интерфейс:

1. В боковой панели (sidebar) отображается статистика всех проверок и выбор модели.
2. На главной странице — загрузчик изображения, предпросмотр, кнопка "Определить".
3. После классификации отображается карточка с результатом (Empty / Occupied) и уровнем уверенности.
4. Ниже — таблица истории всех проверок с датой, временем, именем файла, результатом и уверенностью.
5. Кнопка "Очистить историю" сбрасывает все записи.

### Скриншоты

> *Скриншоты будут добавлены после обучения моделей.*

![Streamlit интерфейс](screenshots/streamlit_interface.png)

![Таблица сравнения](screenshots/comparison_table.png)

![Confusion Matrix](screenshots/confusion_matrix.png)

![ROC Curve](screenshots/roc_curve.png)

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
| torchvision | >= 0.15.0 | Предобученные модели, трансформации |
| FiftyOne | >= 0.23.0 | Загрузка датасета PKLot из HuggingFace |
| scikit-learn | >= 1.3.0 | Метрики классификации, разбиение данных |
| Streamlit | >= 1.28.0 | Веб-интерфейс приложения |
| Pandas | >= 2.0.0 | Таблицы результатов и экспорт |
| Matplotlib | >= 3.7.0 | Построение графиков |
| Pillow | >= 10.0.0 | Загрузка и конвертация изображений |
| OpenCV | >= 4.8.0 | Обработка изображений |
| NumPy | >= 1.24.0 | Численные вычисления |
| openpyxl | >= 3.1.0 | Экспорт таблицы сравнения в XLSX |
| tqdm | >= 4.65.0 | Прогресс-бары в цикле обучения |

---

## Результаты

После обучения все артефакты сохраняются в следующих директориях:

| Директория / файл | Содержимое |
|---|---|
| `saved_models/` | Файлы `.pth` — лучшие веса каждой из пяти моделей |
| `plots/` | PNG-графики: кривые обучения, матрицы ошибок, ROC-кривые |
| `results/comparison.csv` | Сравнительная таблица метрик (CSV) |
| `results/comparison.xlsx` | Сравнительная таблица метрик (Excel, с выделением лучшей) |
| `results/analysis.txt` | Текстовый анализ: сильные/слабые стороны, выбор лучшей модели |
| `history.json` | История предсказаний Streamlit-приложения |

Лучшая модель по метрике ROC AUC автоматически выделяется в таблице и рекомендуется для использования в приложении Streamlit.

---

## Воспроизводимость

Все случайные процессы фиксируются через `seed_everything(42)`:

- `random.seed(42)`
- `numpy.random.seed(42)`
- `torch.manual_seed(42)`
- `torch.cuda.manual_seed_all(42)`
- `torch.backends.cudnn.deterministic = True`
- `PYTHONHASHSEED=42`

---

## Авторы

Проект разработан в рамках производственной практики.

**Тема:** Определение занятости парковочного места с использованием искусственных нейронных сетей.
