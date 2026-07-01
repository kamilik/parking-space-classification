"""
models.py — Фабрика для всех пяти предобученных архитектур CNN / Transformer.

Поддерживаемые архитектуры
--------------------------
1. ResNet18          — torchvision.models.resnet18
2. DenseNet121       — torchvision.models.densenet121
3. EfficientNet-B0   — torchvision.models.efficientnet_b0
4. MobileNetV3-Large — torchvision.models.mobilenet_v3_large
5. ViT-B/16          — torchvision.models.vit_b_16

Каждая модель загружается с предобученными весами ImageNet через современный
API перечисления Weights (torchvision ≥ 0.13). Финальная классификационная
голова заменяется, чтобы модель выдавала ровно ``num_classes`` логитов.

Публичное API
-------------
    from models import get_model, get_model_info

    model = get_model("ResNet18", num_classes=2)
    info  = get_model_info(model)
    # {'total_params': 11_181_634, 'trainable_params': 11_181_634,
    #  'model_size_mb': 42.6}
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    ViT_B_16_Weights,
)

# ---------------------------------------------------------------------------
# Логгер модуля
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Внутренний реестр — отображает каноническое имя → функция-строитель
# ---------------------------------------------------------------------------
_SUPPORTED_NAMES: tuple[str, ...] = (
    "ResNet18",
    "DenseNet121",
    "EfficientNet-B0",
    "MobileNetV3-Large",
    "ViT-B/16",
)


# ---------------------------------------------------------------------------
# Внутренние функции-строители (по одной на архитектуру)
# ---------------------------------------------------------------------------

def _build_resnet18(num_classes: int, pretrained: bool) -> nn.Module:
    """
    Создаёт ResNet-18 и заменяет полносвязную голову.

    Исходный слой ``model.fc`` является ``nn.Linear(512, 1000)``.
    Он заменяется на ``nn.Linear(512, num_classes)``.

    Параметры
    ----------
    num_classes:
        Количество выходных классов (2 для бинарной классификации парковки).
    pretrained:
        Если ``True``, загружает предобученные веса ImageNet-1K перед
        изменением головы.

    Возвращает
    ----------
    nn.Module
        ResNet-18 с изменённой классификационной головой.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model: nn.Module = tv_models.resnet18(weights=weights)
    in_features: int = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    logger.debug(
        "ResNet18 built — head: Linear(%d, %d), pretrained=%s",
        in_features,
        num_classes,
        pretrained,
    )
    return model


def _build_densenet121(num_classes: int, pretrained: bool) -> nn.Module:
    """
    Создаёт DenseNet-121 и заменяет классификационную голову.

    Исходный слой ``model.classifier`` является ``nn.Linear(1024, 1000)``.
    Он заменяется на ``nn.Linear(1024, num_classes)``.

    Параметры
    ----------
    num_classes:
        Количество выходных классов.
    pretrained:
        Если ``True``, загружает предобученные веса ImageNet-1K.

    Возвращает
    ----------
    nn.Module
        DenseNet-121 с изменённой классификационной головой.
    """
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model: nn.Module = tv_models.densenet121(weights=weights)
    in_features: int = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    logger.debug(
        "DenseNet121 built — head: Linear(%d, %d), pretrained=%s",
        in_features,
        num_classes,
        pretrained,
    )
    return model


def _build_efficientnet_b0(num_classes: int, pretrained: bool) -> nn.Module:
    """
    Создаёт EfficientNet-B0 и заменяет линейный слой в его классификаторе.

    Классификатор является ``nn.Sequential``:
        [0] Dropout(p=0.2)
        [1] Linear(1280, 1000)   ← заменяется

    Параметры
    ----------
    num_classes:
        Количество выходных классов.
    pretrained:
        Если ``True``, загружает предобученные веса ImageNet-1K.

    Возвращает
    ----------
    nn.Module
        EfficientNet-B0 с изменённой классификационной головой.
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model: nn.Module = tv_models.efficientnet_b0(weights=weights)
    in_features: int = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    logger.debug(
        "EfficientNet-B0 built — head: Linear(%d, %d), pretrained=%s",
        in_features,
        num_classes,
        pretrained,
    )
    return model


def _build_mobilenet_v3_large(num_classes: int, pretrained: bool) -> nn.Module:
    """
    Создаёт MobileNetV3-Large и заменяет финальный линейный слой.

    Классификатор является ``nn.Sequential``:
        [0] Linear(960, 1280)
        [1] Hardswish()
        [2] Dropout(p=0.2)
        [3] Linear(1280, 1000)   ← заменяется

    Параметры
    ----------
    num_classes:
        Количество выходных классов.
    pretrained:
        Если ``True``, загружает предобученные веса ImageNet-1K.

    Возвращает
    ----------
    nn.Module
        MobileNetV3-Large с изменённой классификационной головой.
    """
    weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model: nn.Module = tv_models.mobilenet_v3_large(weights=weights)
    in_features: int = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    logger.debug(
        "MobileNetV3-Large built — head: Linear(%d, %d), pretrained=%s",
        in_features,
        num_classes,
        pretrained,
    )
    return model


def _build_vit_b_16(num_classes: int, pretrained: bool) -> nn.Module:
    """
    Создаёт Vision Transformer ViT-B/16 и заменяет классификационную голову.

    Модель предоставляет ``model.heads.head``, который является слоем
    ``nn.Linear(768, 1000)``. Он заменяется на ``nn.Linear(768, num_classes)``.

    Параметры
    ----------
    num_classes:
        Количество выходных классов.
    pretrained:
        Если ``True``, загружает предобученные веса ImageNet-1K.

    Возвращает
    ----------
    nn.Module
        ViT-B/16 с изменённой классификационной головой.
    """
    weights = ViT_B_16_Weights.DEFAULT if pretrained else None
    model: nn.Module = tv_models.vit_b_16(weights=weights)
    in_features: int = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    logger.debug(
        "ViT-B/16 built — head: Linear(%d, %d), pretrained=%s",
        in_features,
        num_classes,
        pretrained,
    )
    return model


# ---------------------------------------------------------------------------
# Внутренняя таблица диспетчеризации
# ---------------------------------------------------------------------------
_BUILDERS: dict[str, Any] = {
    "ResNet18": _build_resnet18,
    "DenseNet121": _build_densenet121,
    "EfficientNet-B0": _build_efficientnet_b0,
    "MobileNetV3-Large": _build_mobilenet_v3_large,
    "ViT-B/16": _build_vit_b_16,
}


# ---------------------------------------------------------------------------
# Публичное API
# ---------------------------------------------------------------------------

def get_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    """
    Фабричная функция — возвращает одну из пяти поддерживаемых архитектур
    с финальной классификационной головой, адаптированной под ``num_classes``
    выходов.

    Параметры
    ----------
    model_name:
        Один из канонических идентификаторов архитектур, определённых в
        ``Config.MODEL_NAMES``:

        * ``"ResNet18"``
        * ``"DenseNet121"``
        * ``"EfficientNet-B0"``
        * ``"MobileNetV3-Large"``
        * ``"ViT-B/16"``

    num_classes:
        Количество выходных логитов. Для бинарной классификации парковки
        это 2 (``Config.NUM_CLASSES``).

    pretrained:
        Инициализировать ли базовую сеть весами ImageNet-1K.
        Устанавливается в ``False`` только для юнит-тестов или ablation-
        исследований. По умолчанию ``True``.

    Возвращает
    ----------
    nn.Module
        Экземпляр модели с изменённой головой в режиме оценки.
        Вызовите ``.train()`` перед циклом обучения при необходимости.

    Исключения
    ----------
    ValueError
        Если ``model_name`` не является одним из пяти поддерживаемых имён.

    Примеры
    --------
    >>> from models import get_model
    >>> model = get_model("ResNet18", num_classes=2)
    >>> model  # doctest: +ELLIPSIS
    ResNet(...)
    """
    if model_name not in _BUILDERS:
        supported = ", ".join(f'"{n}"' for n in _SUPPORTED_NAMES)
        raise ValueError(
            f"Unknown model name '{model_name}'. "
            f"Supported values are: {supported}."
        )

    logger.info("Building model '%s' (num_classes=%d, pretrained=%s) …",
                model_name, num_classes, pretrained)

    builder = _BUILDERS[model_name]
    model: nn.Module = builder(num_classes, pretrained)

    logger.info("Model '%s' ready.", model_name)
    return model


def get_model_info(model: nn.Module) -> dict[str, Any]:
    """
    Вычислить базовую статистику о модели PyTorch.

    Параметры
    ----------
    model:
        Любой экземпляр ``nn.Module`` (как правило, возвращённый :func:`get_model`).

    Возвращает
    -------
    dict со следующими ключами:

    ``total_params`` : int
        Общее количество скалярных параметров (обучаемых и замороженных).

    ``trainable_params`` : int
        Количество параметров, у которых ``requires_grad`` равно ``True``.

    ``model_size_mb`` : float
        Оценочный размер модели в мегабайтах при хранении в формате float32:
        ``total_params * 4 / (1024 * 1024)``.

    Примеры
    --------
    >>> from models import get_model, get_model_info
    >>> model = get_model("ResNet18", num_classes=2)
    >>> info = get_model_info(model)
    >>> info["total_params"]  # doctest: +SKIP
    11181634
    >>> isinstance(info["model_size_mb"], float)
    True
    """
    total_params: int = sum(p.numel() for p in model.parameters())
    trainable_params: int = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    # Float32 = 4 байта на параметр
    model_size_mb: float = total_params * 4 / (1024 * 1024)

    logger.debug(
        "Model info — total: %d, trainable: %d, size: %.2f MB",
        total_params,
        trainable_params,
        model_size_mb,
    )

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": model_size_mb,
    }
