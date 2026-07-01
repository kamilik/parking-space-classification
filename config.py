"""
config.py — Центральная конфигурация классификатора занятости парковочных мест.

Все константы проекта, пути, гиперпараметры, имена классов и трансформации данных
определены здесь, чтобы каждый другой модуль мог импортировать из единого источника.

Использование
-------------
    from config import Config, get_train_transforms, get_val_transforms, seed_everything

    device = Config.DEVICE          # torch.device("cuda") / "mps" / "cpu"
    names  = Config.CLASS_NAMES     # ["Empty", "Occupied"]
    train_tf = get_train_transforms()
"""

import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torchvision import transforms

# ---------------------------------------------------------------------------
# Логгер модуля
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Определение устройства при импорте
# ---------------------------------------------------------------------------
def _resolve_device() -> torch.device:
    """Возвращает наилучшее доступное вычислительное устройство: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Config — все константы как атрибуты класса
# ---------------------------------------------------------------------------
class Config:
    """
    Центральная конфигурация проекта.

    Все атрибуты определены на уровне класса, чтобы каждый модуль мог
    обращаться к ним без создания экземпляра: ``Config.DEVICE``,
    ``Config.CLASS_NAMES`` и т.д.
    """

    # ------------------------------------------------------------------
    # Воспроизводимость
    # ------------------------------------------------------------------
    SEED: int = 42

    # ------------------------------------------------------------------
    # Данные
    # ------------------------------------------------------------------
    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 32
    NUM_CLASSES: int = 2
    CLASS_NAMES: tuple[str, ...] = ("Empty", "Occupied")

    # ------------------------------------------------------------------
    # Гиперпараметры обучения
    # ------------------------------------------------------------------
    NUM_EPOCHS: int = 20
    LEARNING_RATE: float = 0.0001

    #: EarlyStopping: остановка после заданного числа эпох без улучшения val-loss
    PATIENCE: int = 5

    # ------------------------------------------------------------------
    # Вычислительное устройство (определяется один раз при импорте)
    # ------------------------------------------------------------------
    DEVICE: torch.device = _resolve_device()

    # ------------------------------------------------------------------
    # Статистики нормализации ImageNet
    # ------------------------------------------------------------------
    IMAGENET_MEAN: tuple[float, ...] = (0.485, 0.456, 0.406)
    IMAGENET_STD: tuple[float, ...] = (0.229, 0.224, 0.225)

    # ------------------------------------------------------------------
    # Имена архитектур (должны совпадать с ключами в models.py)
    # ------------------------------------------------------------------
    MODEL_NAMES: tuple[str, ...] = (
        "ResNet18",
        "DenseNet121",
        "EfficientNet-B0",
        "MobileNetV3-Large",
        "ViT-B/16",
    )

    # ------------------------------------------------------------------
    # Пути файловой системы (все относительно директории этого файла)
    # ------------------------------------------------------------------
    #: Корневая директория проекта (директория, содержащая config.py)
    PROJECT_ROOT: Path = Path(__file__).resolve().parent

    #: Корневая директория датасета — сюда попадут изображения PKLot
    DATA_DIR: Path = Path(__file__).resolve().parent / "data"

    #: Структура разбиений в стиле torchvision:
    #:   models/train/{Empty,Occupied}/…
    #:   models/val/{Empty,Occupied}/…
    #:   models/test/{Empty,Occupied}/…
    MODELS_DIR: Path = Path(__file__).resolve().parent / "models"

    #: Контрольные точки лучших эпох (.pth файлы)
    SAVED_MODELS_DIR: Path = Path(__file__).resolve().parent / "saved_models"

    #: Метрики обучения и оценки: comparison.csv, comparison.xlsx, JSON
    RESULTS_DIR: Path = Path(__file__).resolve().parent / "results"

    #: Все PNG-графики (кривые потерь, матрицы ошибок, ROC-кривые)
    PLOTS_DIR: Path = Path(__file__).resolve().parent / "plots"

    #: История предсказаний Streamlit
    HISTORY_FILE: Path = Path(__file__).resolve().parent / "history.json"


# ---------------------------------------------------------------------------
# Создание необходимых директорий при импорте
# ---------------------------------------------------------------------------
for _directory in (
    Config.DATA_DIR,
    Config.MODELS_DIR,
    Config.SAVED_MODELS_DIR,
    Config.RESULTS_DIR,
    Config.PLOTS_DIR,
):
    _directory.mkdir(parents=True, exist_ok=True)

logger.info(
    "Config loaded — device: %s | classes: %s",
    Config.DEVICE,
    Config.CLASS_NAMES,
)


# ---------------------------------------------------------------------------
# Утилита: воспроизводимость
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    """
    Фиксирует все случайные зерна для полной воспроизводимости.

    Устанавливает зерна для встроенного модуля Python ``random``, NumPy,
    PyTorch (как для CPU, так и для каждого CUDA-устройства) и включает
    детерминированный режим cuDNN.

    Параметры
    ----------
    seed:
        Целочисленное значение зерна. Передавайте ``Config.SEED`` (42)
        во всём проекте для обеспечения согласованных результатов между
        запусками.

    Примечания
    ----------
    ``torch.backends.cudnn.benchmark`` отключается, чтобы на каждом
    прямом проходе выбирался один и тот же алгоритм свёртки, даже
    ценой небольшой потери производительности.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed fixed to %d", seed)


# ---------------------------------------------------------------------------
# Трансформации данных
# ---------------------------------------------------------------------------
def get_train_transforms() -> transforms.Compose:
    """
    Возвращает пайплайн аугментации, применяемый к обучающим изображениям.

    Все операции аугментации применяются последовательно до передачи
    тензора в модель.

    Пайплайн
    --------
    1. ``RandomResizedCrop(224, scale=(0.8, 1.0))``
       Случайная обрезка, сохраняющая 80–100% исходной площади, с
       последующим масштабированием до 224×224.
    2. ``RandomHorizontalFlip()``
       Горизонтальное отражение с вероятностью 0.5.
    3. ``RandomRotation(15)``
       Случайный поворот в диапазоне [−15°, +15°].
    4. ``ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)``
       Случайное изменение цветовых характеристик для фотометрической инвариантности.
    5. ``ToTensor()``
       Преобразует PIL-изображение (H×W×C, uint8) в тензор float32 (C×H×W)
       в диапазоне [0, 1].
    6. ``Normalize(IMAGENET_MEAN, IMAGENET_STD)``
       Пополосная стандартизация по статистикам ImageNet.

    Возвращает
    ----------
    transforms.Compose
        Составной трансформ, готовый для передачи в ``torch.utils.data.Dataset``.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                Config.IMAGE_SIZE,
                scale=(0.8, 1.0),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=Config.IMAGENET_MEAN,
                std=Config.IMAGENET_STD,
            ),
        ]
    )


def get_val_transforms() -> transforms.Compose:
    """
    Возвращает детерминированный пайплайн предобработки для изображений
    валидации и тестирования.

    Аугментация здесь не применяется — только масштабирование и
    нормализация, необходимые для воспроизведения входной статистики
    времени обучения.

    Пайплайн
    --------
    1. ``Resize(256)``
       Масштабирование короткой стороны до 256 пикселей с сохранением
       соотношения сторон.
    2. ``CenterCrop(224)``
       Извлечение центрального патча 224×224.
    3. ``ToTensor()``
       Преобразует PIL-изображение (H×W×C, uint8) в тензор float32 (C×H×W)
       в диапазоне [0, 1].
    4. ``Normalize(IMAGENET_MEAN, IMAGENET_STD)``
       Пополосная стандартизация по статистикам ImageNet.

    Возвращает
    ----------
    transforms.Compose
        Составной трансформ, готовый для передачи в ``torch.utils.data.Dataset``.
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=Config.IMAGENET_MEAN,
                std=Config.IMAGENET_STD,
            ),
        ]
    )
