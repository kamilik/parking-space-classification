"""
dataset.py — Загрузка, подготовка датасета PKLot и создание DataLoader.

Этот модуль обрабатывает все этапы подготовки данных для классификатора
занятости парковочных мест:

1. Загрузка датасета PKLot через FiftyOne Model Zoo.
2. Преобразование датасета FiftyOne в структуру директорий
   ``torchvision.datasets.ImageFolder``::

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

3. Создание экземпляров PyTorch ``DataLoader``, готовых для обучения.

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
_FIFTYONE_DATASET_NAME: str = "Voxel51/PKLot"
_SPLIT_NAMES: tuple[str, ...] = ("train", "val", "test")
_CLASS_NAMES: tuple[str, ...] = ("Empty", "Occupied")

# Соотношения разбиений при отсутствии предопределённых разбиений в датасете.
_TRAIN_RATIO: float = 0.70
_VAL_RATIO: float = 0.15
# Доля теста = 1 - _TRAIN_RATIO - _VAL_RATIO = 0.15  (вычисляется автоматически)

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
        logger.info("Dataset summary — total images: %d", self.total)
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
            logger.debug("Directory ready: %s", target)


def _label_to_class(label: str) -> str | None:
    """
    Приводит метку разметки FiftyOne к одному из двух канонических имён
    классов, используемых в проекте.

    Датасет PKLot использует ``"empty"`` / ``"occupied"`` (строчные) в
    некоторых версиях и ``"Empty"`` / ``"Occupied"`` (с заглавной) в
    других. Эта функция принимает оба варианта без учёта регистра.

    Возвращает ``None`` для любой нераспознанной метки, чтобы вызывающий
    код мог пропустить такой образец.
    """
    mapping: dict[str, str] = {
        "empty": "Empty",
        "occupied": "Occupied",
    }
    return mapping.get(label.strip().lower())


def _collect_samples_from_fiftyone(
    fo_dataset: Any,
) -> list[tuple[Path, str]]:
    """
    Итерирует FiftyOne датасет и возвращает пары ``(путь_к_изображению, имя_класса)``
    для каждого образца с допустимой меткой разметки.

    Параметры
    ----------
    fo_dataset:
        Экземпляр ``fiftyone.core.dataset.Dataset``.

    Возвращает
    ----------
    list of (filepath, class_name)
        Отфильтрованный список — образцы без распознанной метки ``ground_truth``
        пропускаются с предупреждением.
    """
    samples: list[tuple[Path, str]] = []
    skipped: int = 0

    for sample in fo_dataset:
        gt = sample.get_field("ground_truth")
        if gt is None:
            skipped += 1
            continue
        label: str | None = _label_to_class(gt.label)
        if label is None:
            logger.warning("Unrecognised label '%s' — skipping sample.", gt.label)
            skipped += 1
            continue
        samples.append((Path(sample.filepath), label))

    logger.info(
        "FiftyOne samples collected: %d valid, %d skipped.",
        len(samples),
        skipped,
    )
    return samples


def _collect_samples_by_split_from_fiftyone(
    fo_dataset: Any,
) -> dict[str, list[tuple[Path, str]]]:
    """
    Собирает образцы из датасета FiftyOne и группирует их по тегу разбиения.

    Каждый образец FiftyOne может иметь ``"train"``, ``"validation"``
    (преобразуется в ``"val"``) или ``"test"`` в списке тегов ``tags``.
    Образцы без тегов помещаются в группу ``"untagged"``, чтобы вызывающий
    код мог перейти к случайному разбиению.

    Возвращает
    ----------
    dict, отображающий имя разбиения в список (filepath, class_name)
        Ключи: ``"train"``, ``"val"``, ``"test"`` и/или ``"untagged"``.
    """
    tag_to_split: dict[str, str] = {
        "train": "train",
        "validation": "val",
        "val": "val",
        "test": "test",
    }
    buckets: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    skipped: int = 0

    for sample in fo_dataset:
        gt = sample.get_field("ground_truth")
        if gt is None:
            skipped += 1
            continue
        label: str | None = _label_to_class(gt.label)
        if label is None:
            logger.warning("Unrecognised label '%s' — skipping sample.", gt.label)
            skipped += 1
            continue

        filepath = Path(sample.filepath)
        tags: list[str] = list(sample.tags) if sample.tags else []

        assigned_split: str = "untagged"
        for tag in tags:
            if tag.lower() in tag_to_split:
                assigned_split = tag_to_split[tag.lower()]
                break

        buckets[assigned_split].append((filepath, label))

    logger.info(
        "Sample distribution by tag: %s | skipped=%d",
        {k: len(v) for k, v in buckets.items()},
        skipped,
    )
    return dict(buckets)


def _split_samples_randomly(
    samples: list[tuple[Path, str]],
    seed: int,
) -> dict[str, list[tuple[Path, str]]]:
    """
    Стратифицированное случайное разбиение на train / val / test.

    Использует ``sklearn.model_selection.train_test_split`` со ``stratify``
    для сохранения соотношения классов во всех трёх разбиениях.

    Параметры
    ----------
    samples:
        Полный список пар ``(filepath, class_name)``.
    seed:
        Случайное зерно для воспроизводимости.

    Возвращает
    ----------
    dict с ключами ``"train"``, ``"val"``, ``"test"``.
    """
    labels = [cls for _, cls in samples]

    # Первое разбиение: train vs. (val + test)
    train_samples, remaining_samples = train_test_split(
        samples,
        test_size=1.0 - _TRAIN_RATIO,
        stratify=labels,
        random_state=seed,
    )

    # Второе разбиение: val vs. test (равные половины оставшейся части)
    remaining_labels = [cls for _, cls in remaining_samples]
    val_samples, test_samples = train_test_split(
        remaining_samples,
        test_size=0.5,  # 50% от оставшихся 30% = 15% общего объёма
        stratify=remaining_labels,
        random_state=seed,
    )

    logger.info(
        "Random split — train=%d | val=%d | test=%d",
        len(train_samples),
        len(val_samples),
        len(test_samples),
    )
    return {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples,
    }


def _copy_images(
    split_data: dict[str, list[tuple[Path, str]]],
    data_dir: Path,
) -> DatasetInfo:
    """
    Копирует изображения в дерево ImageFolder и возвращает статистику датасета.

    Для каждой пары ``(src_path, class_name)`` в ``split_data`` исходное
    изображение копируется (через ``shutil.copy2``) по пути::

        data_dir/{разбиение}/{class_name}/{оригинальное_имя_файла}

    Дублирующиеся имена файлов в паре разбиение/класс различаются
    добавлением нуль-заполненного индекса.

    Параметры
    ----------
    split_data:
        Отображение имени разбиения в список пар ``(filepath, class_name)``.
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
        samples = split_data.get(split, [])
        # Отслеживаем уже встреченные имена файлов в данной директории (разбиение, класс),
        # чтобы избежать молчаливой перезаписи при совпадении имён из разных источников.
        seen: dict[tuple[str, str], int] = {}

        for src_path, class_name in samples:
            dst_dir = data_dir / split / class_name
            filename = src_path.name
            # Ключ включает class_name и filename для ограничения дубликатов в пределах класса.
            key = (class_name, filename)
            if key in seen:
                seen[key] += 1
                stem = src_path.stem
                suffix = src_path.suffix
                filename = f"{stem}_{seen[key]:05d}{suffix}"
            else:
                seen[key] = 0

            dst_path = dst_dir / filename
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)

            counts[split] += 1
            distribution[split][class_name] += 1

        logger.info(
            "Split '%s' — %d images copied (%s).",
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
    Загружает датасет PKLot через FiftyOne и организует его в структуру
    директорий, совместимую с ImageFolder.

    Эта функция является **идемпотентной**: если целевая директория уже
    содержит ожидаемую структуру разбиений/классов (определяется через
    файл-сторож), функция пропускает шаги загрузки и копирования и
    немедленно возвращает управление.

    Параметры
    ----------
    data_dir:
        Корневая директория, в которой будет создано дерево ImageFolder.
        Обычно ``Config.DATA_DIR`` (``project/data/``).
    seed:
        Случайное зерно для стратифицированных разбиений (по умолчанию 42).

    Возвращает
    ----------
    Path
        ``data_dir`` — корень подготовленного дерева ImageFolder.

    Исключения
    ----------
    RuntimeError
        Если загрузка FiftyOne прошла успешно, но в датасете не найдено
        допустимых образцов.

    Примечания
    ----------
    * Изображения **копируются** (не создаются символические ссылки) для
      совместимости с Colab и кросс-платформенного использования.
    * Датасет PKLot в FiftyOne может иметь теги ``"train"``, ``"validation"``
      и ``"test"`` на образцах. При наличии этих тегов они используются
      напрямую. При их отсутствии (или при покрытии менее двух разбиений)
      применяется стратифицированное случайное разбиение 70 / 15 / 15.
    """
    data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Быстрый путь: подготовка уже выполнена
    # ------------------------------------------------------------------
    if _is_already_prepared(data_dir):
        logger.info(
            "ImageFolder structure already exists at '%s' — skipping preparation.",
            data_dir,
        )
        info = _count_existing_dataset(data_dir)
        info.log_summary()
        return data_dir

    logger.info("Starting PKLot dataset preparation in '%s'.", data_dir)

    # ------------------------------------------------------------------
    # Шаг 1: Загрузка через FiftyOne
    # ------------------------------------------------------------------
    try:
        import fiftyone.zoo as foz  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "fiftyone is not installed.  Run: pip install fiftyone"
        ) from exc

    logger.info("Downloading '%s' from FiftyOne Zoo …", _FIFTYONE_DATASET_NAME)
    fo_dataset = foz.load_zoo_dataset(_FIFTYONE_DATASET_NAME)
    logger.info(
        "FiftyOne dataset loaded: %d total samples.", len(fo_dataset)
    )

    # ------------------------------------------------------------------
    # Шаг 2: Сбор образцов с учётом существующих тегов разбиений
    # ------------------------------------------------------------------
    buckets = _collect_samples_by_split_from_fiftyone(fo_dataset)

    # Определяем, содержит ли датасет пригодные теги разбиений.
    tagged_splits = {
        k: v for k, v in buckets.items() if k in ("train", "val", "test") and v
    }
    untagged_samples = buckets.get("untagged", [])

    has_predefined_splits = len(tagged_splits) >= 2  # минимум train + ещё одно

    if has_predefined_splits and not untagged_samples:
        # Все образцы имеют теги разбиений — используем их напрямую.
        logger.info(
            "Using pre-defined dataset splits: %s",
            {k: len(v) for k, v in tagged_splits.items()},
        )
        split_data: dict[str, list[tuple[Path, str]]] = dict(tagged_splits)

        # Гарантируем наличие каждого ожидаемого разбиения (даже пустого).
        for split in _SPLIT_NAMES:
            split_data.setdefault(split, [])

    elif has_predefined_splits and untagged_samples:
        # Часть образцов имеет теги, часть нет. Объединяем теговые разбиения
        # и применяем стратифицированное разбиение к нетеговой части.
        logger.info(
            "Mixed tagging: %d tagged samples, %d untagged — "
            "splitting untagged portion %d/%d/%d.",
            sum(len(v) for v in tagged_splits.values()),
            len(untagged_samples),
            int(_TRAIN_RATIO * 100),
            int(_VAL_RATIO * 100),
            int((1.0 - _TRAIN_RATIO - _VAL_RATIO) * 100),
        )
        extra_splits = _split_samples_randomly(untagged_samples, seed=seed)
        split_data = {
            "train": tagged_splits.get("train", []) + extra_splits["train"],
            "val": tagged_splits.get("val", []) + extra_splits["val"],
            "test": tagged_splits.get("test", []) + extra_splits["test"],
        }

    else:
        # Нет пригодных тегов разбиений — собираем всё и разбиваем случайно.
        all_samples = _collect_samples_from_fiftyone(fo_dataset)
        if not all_samples:
            raise RuntimeError(
                "No valid samples found in the PKLot FiftyOne dataset. "
                "Check that the download completed successfully."
            )
        logger.info(
            "No pre-defined splits detected — applying stratified 70/15/15 split."
        )
        split_data = _split_samples_randomly(all_samples, seed=seed)

    # ------------------------------------------------------------------
    # Шаг 3: Создание дерева директорий
    # ------------------------------------------------------------------
    _build_empty_imagefolder_tree(data_dir)

    # ------------------------------------------------------------------
    # Шаг 4: Копирование изображений
    # ------------------------------------------------------------------
    logger.info("Copying images into ImageFolder structure …")
    info = _copy_images(split_data, data_dir)
    info.log_summary()

    # ------------------------------------------------------------------
    # Шаг 5: Запись файла-сторожа для пропуска при повторных запусках
    # ------------------------------------------------------------------
    _sentinel_path(data_dir).write_text(
        f"PKLot prepared: train={info.num_train}, val={info.num_val}, test={info.num_test}\n"
    )
    logger.info("Preparation complete.  Sentinel written to '%s'.", _sentinel_path(data_dir))

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
        "Training dataset: %d images | classes: %s",
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
        "Validation dataset: %d images | classes: %s",
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
        "Test dataset: %d images | classes: %s",
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
        "DataLoaders created — batch_size=%d | pin_memory=%s | num_workers=%d",
        batch_size,
        pin_memory,
        _NUM_WORKERS,
    )

    return train_loader, val_loader, test_loader
