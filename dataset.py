"""
dataset.py — Загрузка, подготовка датасета PKLot и создание DataLoader.

Этот модуль обрабатывает все этапы подготовки данных для классификатора
занятости парковочных мест:

1. Загрузка датасета PKLot с Kaggle через kagglehub.
2. Разбиение по парковкам:
   - train / val — PUCPR + UFPR04 (все погоды), стратифицированное 85 / 15.
   - test — UFPR05 (все погоды).
3. Копирование в структуру директорий ``torchvision.datasets.ImageFolder``::

       data_dir/
           train/
               Empty/
               Occupied/
           val/
               Empty/
               Occupied/
           test/
               Empty/
               Occupied/

4. Создание экземпляров PyTorch ``DataLoader``, готовых для обучения.

Публичное API
-------------
- ``DatasetInfo``                — датакласс с размерами разбиений и числом классов
- ``download_and_prepare_dataset(data_dir)`` — идемпотентная функция подготовки
- ``get_data_loaders(...)``      — возвращает (train_loader, val_loader, test_loader)

Использование
-------------
    from dataset import download_and_prepare_dataset, get_data_loaders, DatasetInfo
    from config import Config, get_train_transforms, get_val_transforms

    data_root = download_and_prepare_dataset(Config.DATA_DIR)
    train_dl, val_dl, test_dl = get_data_loaders(
        data_root, Config.BATCH_SIZE,
        get_train_transforms(), get_val_transforms(),
    )
"""

from __future__ import annotations

import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets

# ---------------------------------------------------------------------------
# Логгер модуля
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

# Идентификатор датасета на Kaggle.
_KAGGLE_DATASET: str = "blanderbuss/parking-lot-dataset"

# Парковки для обучения и тестирования.
_TRAIN_LOTS: tuple[str, ...] = ("PUCPR", "UFPR04")
_TEST_LOT: str = "UFPR05"

# Допустимые расширения изображений (строчные).
_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})

_SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
_CLASS_NAMES: tuple[str, ...] = ("Empty", "Occupied")

# Доля валидации внутри обучающих парковок (PUCPR + UFPR04).
_VAL_RATIO: float = 0.15

# Настройки DataLoader
_NUM_WORKERS: int = 2


# ---------------------------------------------------------------------------
# Датакласс DatasetInfo
# ---------------------------------------------------------------------------
@dataclass
class DatasetInfo:
    """
    Сводная статистика по подготовленному датасету PKLot.

    Атрибуты
    ----------
    num_train:
        Количество изображений в обучающем разбиении.
    num_val:
        Количество изображений в валидационном разбиении.
    num_test:
        Количество изображений в тестовом разбиении.
    class_distribution:
        Вложенное отображение ``{имя_разбиения: {имя_класса: количество}}``
        с абсолютным числом изображений по классам для каждого разбиения.
    """

    num_train: int
    num_val: int
    num_test: int
    class_distribution: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        """Общее количество изображений по всем разбиениям."""
        return self.num_train + self.num_val + self.num_test

    def log_summary(self) -> None:
        """Записывает читаемую сводку в логгер модуля."""
        logger.info("Сводка датасета — всего изображений: %d", self.total)
        logger.info(
            "  train=%d | val=%d | test=%d",
            self.num_train,
            self.num_val,
            self.num_test,
        )
        for split, counts in self.class_distribution.items():
            parts = ", ".join(f"{cls}={n}" for cls, n in sorted(counts.items()))
            logger.info("  %s: %s", split, parts)


# ---------------------------------------------------------------------------
# Внутренние вспомогательные функции
# ---------------------------------------------------------------------------

def _sentinel_path(data_dir: Path) -> Path:
    """Возвращает путь к файлу-сторожу, отмечающему завершённую подготовку."""
    return data_dir / ".pklot_prepared"


def _is_already_prepared(data_dir: Path) -> bool:
    """
    Возвращает ``True``, если структура ImageFolder уже была построена
    в предыдущем запуске.

    Функция проверяет как наличие файла-сторожа, записанного в конце
    подготовки, так и то, что все ожидаемые директории разбиений/классов
    существуют.
    """
    if not _sentinel_path(data_dir).exists():
        return False
    for split in _SPLIT_NAMES:
        for cls in _CLASS_NAMES:
            if not (data_dir / split / cls).is_dir():
                return False
    return True


def _build_empty_imagefolder_tree(data_dir: Path) -> None:
    """
    Создаёт пустое дерево директорий для структуры ImageFolder.

    Создаёт ``data_dir/{разбиение}/{класс}/`` для каждой комбинации
    имени разбиения и имени класса. Существующие директории не затрагиваются.
    """
    for split in _SPLIT_NAMES:
        for cls in _CLASS_NAMES:
            target = data_dir / split / cls
            target.mkdir(parents=True, exist_ok=True)
            logger.debug("Директория готова: %s", target)


def _download_kaggle(dataset_id: str) -> Path:
    """
    Загружает датасет PKLot с Kaggle и возвращает путь к корню
    с парковками (директория, содержащая PUCPR / UFPR04 / UFPR05).
    """
    import kagglehub  # type: ignore[import]

    logger.info("Загрузка датасета '%s' с Kaggle …", dataset_id)
    raw_path = Path(kagglehub.dataset_download(dataset_id))
    logger.info("Датасет загружен: %s", raw_path)

    root = _find_image_root(raw_path)
    logger.info("Корень изображений: %s", root)
    return root


def _find_image_root(base: Path) -> Path:
    """
    Рекурсивно ищет директорию, содержащую папки парковок
    (PUCPR, UFPR04, UFPR05). Kaggle может вложить данные на
    произвольную глубину.
    """
    lot_names = set(_TRAIN_LOTS) | {_TEST_LOT}

    if lot_names.issubset({d.name for d in base.iterdir() if d.is_dir()}):
        return base

    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        try:
            subdirs = {d.name for d in child.iterdir() if d.is_dir()}
        except PermissionError:
            continue
        if lot_names.issubset(subdirs):
            return child
        found = _find_image_root(child)
        if found != child:
            return found

    raise RuntimeError(
        f"Не удалось найти директорию с парковками {lot_names} "
        f"внутри {base}. Проверьте структуру загруженного датасета."
    )


def _collect_all_images(
    extracted_root: Path,
) -> list[tuple[Path, str, str]]:
    """
    Обходит дерево директорий PKLotSegmented и собирает все изображения.

    Ожидаемая структура::

        PKLotSegmented/
            {parking_lot}/{weather}/{date}/Empty/*.jpg
            {parking_lot}/{weather}/{date}/Occupied/*.jpg

    Параметры
    ----------
    extracted_root:
        Корень распакованного архива PKLotSegmented.

    Возвращает
    ----------
    list of (src_path, class_name, unique_filename)
        ``class_name`` — ``"Empty"`` или ``"Occupied"``.
        ``unique_filename`` — уникальное имя файла вида
        ``{parking_lot}_{weather}_{date}_{original_name}``
        для предотвращения коллизий при копировании.
    """
    records: list[tuple[Path, str, str]] = []
    # Счётчик изображений по парковкам для логирования.
    lot_counts: dict[str, int] = defaultdict(int)

    for img_path in sorted(extracted_root.rglob("*")):
        # Фильтруем только файлы с допустимыми расширениями.
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue

        # Ожидаемые части пути (относительно extracted_root):
        #   parts[-1] = filename
        #   parts[-2] = Empty | Occupied  (имя класса)
        #   parts[-3] = дата (например, 2012-09-12)
        #   parts[-4] = Cloudy | Rainy | Sunny  (погода)
        #   parts[-5] = PUCPR | UFPR04 | UFPR05  (парковка)
        try:
            rel_parts = img_path.relative_to(extracted_root).parts
        except ValueError:
            logger.warning("Не удалось вычислить относительный путь: %s", img_path)
            continue

        if len(rel_parts) < 5:
            logger.debug("Нестандартная глубина пути, пропускаем: %s", img_path)
            continue

        parking_lot = rel_parts[0]
        weather = rel_parts[1]
        date_str = rel_parts[2]
        class_folder = rel_parts[3]
        original_name = rel_parts[4]

        # Нормализуем имя класса: принимаем Empty / empty / Occupied / occupied.
        class_lower = class_folder.strip().lower()
        if class_lower == "empty":
            class_name = "Empty"
        elif class_lower == "occupied":
            class_name = "Occupied"
        else:
            logger.debug(
                "Неизвестный класс '%s', пропускаем: %s", class_folder, img_path
            )
            continue

        # Уникальное имя файла для предотвращения коллизий между парковками.
        unique_filename = f"{parking_lot}_{weather}_{date_str}_{original_name}"

        records.append((img_path, class_name, unique_filename))
        lot_counts[parking_lot] += 1

    logger.info(
        "Всего собрано изображений: %d. По парковкам: %s",
        len(records),
        dict(lot_counts),
    )
    return records


def _split_by_parking_lot(
    records: list[tuple[Path, str, str]],
    extracted_root: Path,
    seed: int,
) -> dict[str, list[tuple[Path, str, str]]]:
    """
    Разбиение по парковкам: PUCPR + UFPR04 → train/val, UFPR05 → test.

    Внутри обучающих парковок выполняется стратифицированное разбиение
    на train / val (85 / 15) для сохранения баланса классов.
    """
    train_lot_names = {lot.lower() for lot in _TRAIN_LOTS}
    test_lot_name = _TEST_LOT.lower()

    train_val_records: list[tuple[Path, str, str]] = []
    test_records: list[tuple[Path, str, str]] = []

    for rec in records:
        src_path = rec[0]
        try:
            lot_name = src_path.relative_to(extracted_root).parts[0].lower()
        except (ValueError, IndexError):
            continue

        if lot_name == test_lot_name:
            test_records.append(rec)
        elif lot_name in train_lot_names:
            train_val_records.append(rec)

    labels = [cls for _, cls, _ in train_val_records]
    train_records, val_records = train_test_split(
        train_val_records,
        test_size=_VAL_RATIO,
        stratify=labels,
        random_state=seed,
    )

    logger.info(
        "Разбиение по парковкам — train=%d (PUCPR+UFPR04 85%%) | "
        "val=%d (PUCPR+UFPR04 15%%) | test=%d (UFPR05)",
        len(train_records),
        len(val_records),
        len(test_records),
    )
    return {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }


def _log_parking_lot_distribution(
    split_records: dict[str, list[tuple[Path, str, str]]],
) -> None:
    """
    Выводит в лог количество изображений от каждой парковки в каждом разбиении.

    Параметры
    ----------
    split_records:
        Отображение имени разбиения в список кортежей
        ``(filepath, class_name, unique_filename)``.
    """
    for split, records in split_records.items():
        lot_cls_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for src_path, class_name, _ in records:
            # Имя парковки — первая часть относительного пути.
            # unique_filename имеет вид {lot}_{weather}_{date}_{orig}.
            # Восстанавливаем имя парковки из уникального имени файла.
            lot_name = src_path.parts[-5] if len(src_path.parts) >= 5 else "unknown"
            lot_cls_counts[lot_name][class_name] += 1

        for lot, cls_counts in sorted(lot_cls_counts.items()):
            parts = ", ".join(
                f"{cls}={cnt}" for cls, cnt in sorted(cls_counts.items())
            )
            logger.info("  [%s] %s: %s", split, lot, parts)


def _copy_images(
    split_records: dict[str, list[tuple[Path, str, str]]],
    data_dir: Path,
) -> DatasetInfo:
    """
    Копирует изображения в дерево ImageFolder и возвращает статистику датасета.

    Для каждого кортежа ``(src_path, class_name, unique_filename)``
    исходное изображение копируется (через ``shutil.copy2``) по пути::

        data_dir/{разбиение}/{class_name}/{unique_filename}

    Параметры
    ----------
    split_records:
        Отображение имени разбиения в список кортежей
        ``(filepath, class_name, unique_filename)``.
    data_dir:
        Корень дерева ImageFolder (уже созданного функцией
        ``_build_empty_imagefolder_tree``).

    Возвращает
    ----------
    DatasetInfo
        Счётчики по разбиениям и классам.
    """
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    distribution: dict[str, dict[str, int]] = {
        split: {cls: 0 for cls in _CLASS_NAMES}
        for split in _SPLIT_NAMES
    }

    for split in _SPLIT_NAMES:
        records = split_records.get(split, [])

        for src_path, class_name, unique_filename in records:
            dst_dir = data_dir / split / class_name
            dst_path = dst_dir / unique_filename

            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)

            counts[split] += 1
            distribution[split][class_name] += 1

        logger.info(
            "Разбиение '%s' — скопировано %d изображений (%s).",
            split,
            counts[split],
            ", ".join(
                f"{cls}={distribution[split][cls]}" for cls in _CLASS_NAMES
            ),
        )

    return DatasetInfo(
        num_train=counts["train"],
        num_val=counts["val"],
        num_test=counts["test"],
        class_distribution=distribution,
    )


def _count_existing_dataset(data_dir: Path) -> DatasetInfo:
    """
    Подсчитывает изображения в уже подготовленном дереве ImageFolder
    и возвращает экземпляр ``DatasetInfo``.

    Используется, когда ``_is_already_prepared`` сообщает о завершённой
    подготовке, чтобы вернуть точную статистику без повторного запуска
    подготовки.
    """
    counts: dict[str, int] = {}
    distribution: dict[str, dict[str, int]] = {}

    for split in _SPLIT_NAMES:
        split_total = 0
        distribution[split] = {}
        for cls in _CLASS_NAMES:
            cls_dir = data_dir / split / cls
            n = len(list(cls_dir.glob("*.*"))) if cls_dir.is_dir() else 0
            distribution[split][cls] = n
            split_total += n
        counts[split] = split_total

    return DatasetInfo(
        num_train=counts["train"],
        num_val=counts["val"],
        num_test=counts["test"],
        class_distribution=distribution,
    )


# ---------------------------------------------------------------------------
# Публичное API
# ---------------------------------------------------------------------------

def download_and_prepare_dataset(data_dir: Path, seed: int = 42) -> Path:
    """
    Загружает датасет PKLot с Kaggle и организует его в структуру ImageFolder.

    Разбиение по парковкам:
    - train / val — PUCPR + UFPR04 (все погоды), стратификация 85 / 15.
    - test — UFPR05 (все погоды).
    """
    data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Быстрый путь: подготовка уже выполнена
    # ------------------------------------------------------------------
    if _is_already_prepared(data_dir):
        logger.info(
            "Структура ImageFolder уже существует в '%s' — подготовка пропущена.",
            data_dir,
        )
        info = _count_existing_dataset(data_dir)
        info.log_summary()
        return data_dir

    logger.info("Начало подготовки датасета PKLot в '%s'.", data_dir)

    # ------------------------------------------------------------------
    # Шаг 1: Загрузка датасета с Kaggle
    # ------------------------------------------------------------------
    extracted_root = _download_kaggle(_KAGGLE_DATASET)

    # ------------------------------------------------------------------
    # Шаг 2: Сбор всех изображений со всех парковок
    # ------------------------------------------------------------------
    all_records = _collect_all_images(extracted_root)

    if not all_records:
        raise RuntimeError(
            "Не найдено ни одного изображения в загруженном датасете PKLot. "
            "Проверьте корректность загрузки и структуру архива."
        )

    # ------------------------------------------------------------------
    # Шаг 3: Разбиение по парковкам
    # ------------------------------------------------------------------
    logger.info(
        "Разбиение по парковкам: train/val=%s, test=%s (%d изображений, seed=%d).",
        _TRAIN_LOTS,
        _TEST_LOT,
        len(all_records),
        seed,
    )
    split_records = _split_by_parking_lot(all_records, extracted_root, seed=seed)

    # ------------------------------------------------------------------
    # Шаг 4: Создание дерева директорий ImageFolder
    # ------------------------------------------------------------------
    _build_empty_imagefolder_tree(data_dir)

    # ------------------------------------------------------------------
    # Шаг 5: Копирование изображений
    # ------------------------------------------------------------------
    logger.info("Копирование изображений в структуру ImageFolder …")
    info = _copy_images(split_records, data_dir)

    # ------------------------------------------------------------------
    # Шаг 6: Детальное логирование распределения по парковкам
    # ------------------------------------------------------------------
    logger.info("Распределение изображений по парковкам в каждом разбиении:")
    _log_parking_lot_distribution(split_records)
    info.log_summary()

    # ------------------------------------------------------------------
    # Шаг 7: Запись файла-сторожа
    # ------------------------------------------------------------------
    _sentinel_path(data_dir).write_text(
        f"PKLot prepared: train={info.num_train}, val={info.num_val}, test={info.num_test}\n",
        encoding="utf-8",
    )
    logger.info(
        "Подготовка завершена. Файл-сторож записан в '%s'.",
        _sentinel_path(data_dir),
    )

    return data_dir


def get_data_loaders(
    data_dir: Path,
    batch_size: int,
    train_transform: Any,
    val_transform: Any,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Создаёт и возвращает экземпляры ``DataLoader`` для обучения, валидации
    и тестирования.

    Использует ``torchvision.datasets.ImageFolder``, ожидающий структуру
    директорий, созданную ``download_and_prepare_dataset``::

        data_dir/
            train/{Empty,Occupied}/
            val/{Empty,Occupied}/
            test/{Empty,Occupied}/

    Параметры
    ----------
    data_dir:
        Корень подготовленного дерева ImageFolder (значение, возвращённое
        ``download_and_prepare_dataset``).
    batch_size:
        Количество образцов в мини-батче (обычно ``Config.BATCH_SIZE = 32``).
    train_transform:
        Пайплайн аугментации для обучающих изображений (например, из
        ``get_train_transforms()``).
    val_transform:
        Детерминированный пайплайн предобработки для изображений валидации
        и тестирования (например, из ``get_val_transforms()``).

    Возвращает
    ----------
    tuple[DataLoader, DataLoader, DataLoader]
        ``(train_loader, val_loader, test_loader)``, готовые к итерации
        во время обучения и оценки модели.

    Примечания
    ----------
    * ``pin_memory`` включается только при наличии CUDA-устройства, следуя
      рекомендации PyTorch для снижения накладных расходов на CPU-only машинах.
    * Обучающий загрузчик использует ``shuffle=True``; оба загрузчика val
      и test используют ``shuffle=False``.
    * ``drop_last=False`` используется везде, чтобы каждое изображение
      оценивалось при валидации и тестировании, даже когда размер датасета
      не делится нацело на ``batch_size``.
    """
    data_dir = Path(data_dir)
    pin_memory: bool = torch.cuda.is_available()

    # ------------------------------------------------------------------
    # Обучающий датасет
    # ------------------------------------------------------------------
    train_dir = data_dir / "train"
    train_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=train_transform,
    )
    logger.info(
        "Обучающий датасет: %d изображений | классы: %s",
        len(train_dataset),
        train_dataset.classes,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=_NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # Валидационный датасет
    # ------------------------------------------------------------------
    val_dir = data_dir / "val"
    val_dataset = datasets.ImageFolder(
        root=str(val_dir),
        transform=val_transform,
    )
    logger.info(
        "Валидационный датасет: %d изображений | классы: %s",
        len(val_dataset),
        val_dataset.classes,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # Тестовый датасет
    # ------------------------------------------------------------------
    test_dir = data_dir / "test"
    test_dataset = datasets.ImageFolder(
        root=str(test_dir),
        transform=val_transform,
    )
    logger.info(
        "Тестовый датасет: %d изображений | классы: %s",
        len(test_dataset),
        test_dataset.classes,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    logger.info(
        "DataLoader созданы — batch_size=%d | pin_memory=%s | num_workers=%d",
        batch_size,
        pin_memory,
        _NUM_WORKERS,
    )

    return train_loader, val_loader, test_loader
