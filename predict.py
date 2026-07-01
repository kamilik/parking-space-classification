"""
predict.py — Инференс на одном изображении для классификатора занятости парковочных мест.

Данный модуль предоставляет три уровня абстракции для запуска инференса:

1. ``load_model()``   — низкий уровень: строит архитектуру и загружает сохранённый
                         state-dict с диска.
2. ``predict_image()`` — средний уровень: принимает путь к изображению, загруженную модель и
                         трансформацию предобработки; возвращает
                         ``PredictionResult``.
3. ``ParkingPredictor`` — высокий уровень: класс-обёртка, владеющий моделью и
                         трансформацией, так что вызывающему коду нужно лишь обращаться к ``.predict()``.

Типичное использование
----------------------
Из Python-кода (например, app.py):

    from predict import ParkingPredictor

    predictor = ParkingPredictor(
        model_path="saved_models/ResNet18_best.pth",
        model_name="ResNet18",
    )
    result = predictor.predict("some_spot.jpg")
    print(result.label)        # "Empty" or "Occupied"
    print(result.confidence)   # e.g. 0.9732
    print(result.probabilities)# {"Empty": 0.0268, "Occupied": 0.9732}

Из командной строки:

    python predict.py --model-path saved_models/ResNet18_best.pth \\
                      --model-name ResNet18 \\
                      --image path/to/spot.jpg
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from config import Config, get_val_transforms
from models import get_model

# ---------------------------------------------------------------------------
# Логгер уровня модуля
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Датакласс PredictionResult
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """
    Контейнер для результата одного инференса на изображении.

    Attributes
    ----------
    label : str
        Предсказанное имя класса, например ``"Empty"`` или ``"Occupied"``.
    confidence : float
        Вероятность softmax для предсказанного класса в диапазоне [0.0, 1.0].
    probabilities : dict[str, float]
        Полное распределение softmax с ключами по именам классов, например
        ``{"Empty": 0.15, "Occupied": 0.85}``.
    inference_time_ms : float
        Время по настенным часам (в миллисекундах), затраченное только на прямой проход
        (без загрузки изображения и предобработки).

    Examples
    --------
    >>> result = PredictionResult(
    ...     label="Occupied",
    ...     confidence=0.9732,
    ...     probabilities={"Empty": 0.0268, "Occupied": 0.9732},
    ...     inference_time_ms=4.7,
    ... )
    >>> result.label
    'Occupied'
    >>> result.confidence
    0.9732
    """

    label: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    inference_time_ms: float = 0.0

    def __str__(self) -> str:
        prob_str = ", ".join(
            f"{name}: {prob:.4f}" for name, prob in self.probabilities.items()
        )
        return (
            f"PredictionResult("
            f"label={self.label!r}, "
            f"confidence={self.confidence:.4f}, "
            f"probabilities={{{prob_str}}}, "
            f"inference_time_ms={self.inference_time_ms:.2f})"
        )


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------

def load_model(
    model_path: Path,
    model_name: str,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """
    Строит архитектуру и восстанавливает веса из сохранённого файла state-dict.

    Функция использует :func:`models.get_model` для построения базовой модели
    (с ``pretrained=False``, чтобы веса ImageNet не скачивались), затем
    загружает дообученные веса из ``model_path`` через
    ``torch.load`` / ``model.load_state_dict``. Модель размещается на
    ``device`` и переводится в режим оценки перед возвратом.

    Parameters
    ----------
    model_path : Path
        Путь к сохранённому файлу ``.pth``, содержащему ``state_dict`` модели.
        Ожидается, что файл был создан через
        ``torch.save(model.state_dict(), path)`` (или аналогичную логику
        ``ModelCheckpoint`` в ``train.py``).
    model_name : str
        Один из пяти канонических идентификаторов архитектур, принимаемых
        :func:`models.get_model`:
        ``"ResNet18"``, ``"DenseNet121"``, ``"EfficientNet-B0"``,
        ``"MobileNetV3-Large"`` или ``"ViT-B/16"``.
    num_classes : int
        Количество выходных классов, для которых обучена модель.
        Используйте ``Config.NUM_CLASSES`` (2), если не переобучали на другом разбиении.
    device : torch.device
        Вычислительное устройство, на которое перемещается модель после загрузки.
        Используйте ``Config.DEVICE`` для автоматического выбора наилучшего устройства.

    Returns
    -------
    nn.Module
        Загруженная модель в режиме оценки на запрошенном устройстве.

    Raises
    ------
    FileNotFoundError
        Если ``model_path`` не существует.
    RuntimeError
        Если ключи state-dict не совпадают с архитектурой модели (например,
        файл ``.pth`` сохранён из другой архитектуры).

    Examples
    --------
    >>> from pathlib import Path
    >>> import torch
    >>> model = load_model(
    ...     model_path=Path("saved_models/ResNet18_best.pth"),
    ...     model_name="ResNet18",
    ...     num_classes=2,
    ...     device=torch.device("cpu"),
    ... )  # doctest: +SKIP
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            "Train the model first (train.py) or provide a valid path."
        )

    logger.info(
        "Loading model '%s' from '%s' onto device '%s' …",
        model_name,
        model_path,
        device,
    )

    # Строим архитектуру без предобученных весов; загрузим собственные.
    model: nn.Module = get_model(model_name, num_classes=num_classes, pretrained=False)

    # Загружаем чекпоинт — маппинг на целевое устройство позволяет загружать
    # GPU-чекпоинты на машине только с CPU.
    state_dict = torch.load(model_path, map_location=device)

    # Некоторые конвейеры обучения оборачивают state-dict в словарь с дополнительными ключами,
    # например {"model_state_dict": ..., "epoch": ..., "val_loss": ...}.
    # Прозрачно обрабатываем оба формата: плоский и вложенный.
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        logger.debug("Checkpoint contains a nested dict; extracting 'model_state_dict'.")
        state_dict = state_dict["model_state_dict"]

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(
        "Model '%s' loaded successfully (%d parameters, eval mode).",
        model_name,
        sum(p.numel() for p in model.parameters()),
    )
    return model


# ---------------------------------------------------------------------------
# predict_image
# ---------------------------------------------------------------------------

def predict_image(
    image_path: str,
    model: nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    class_names: Sequence[str],
) -> PredictionResult:
    """
    Выполняет один прямой проход и возвращает структурированный результат предсказания.

    Функция обрабатывает загрузку изображения, предобработку, формирование батча, инференс и
    преобразование через softmax. Модель должна уже находиться в режиме оценки; эта
    функция никогда не вызывает ``.train()`` или ``.eval()``.

    Parameters
    ----------
    image_path : str
        Путь в файловой системе к файлу изображения. Принимается любой формат,
        поддерживаемый Pillow (JPEG, PNG, BMP, TIFF, …).
    model : nn.Module
        Загруженная модель в режиме оценки, как возвращает :func:`load_model`.
    transform : transforms.Compose
        Трансформация предобработки, применяемая перед прямым проходом. В
        продакшене передавайте результат ``get_val_transforms()`` из
        ``config.py``.
    device : torch.device
        Устройство для перемещения тензора изображения (должно совпадать с устройством модели).
    class_names : Sequence[str]
        Упорядоченный список меток классов, соответствующих выходным логитам модели.
        Для данного проекта передавайте ``Config.CLASS_NAMES``
        (``["Empty", "Occupied"]``).

    Returns
    -------
    PredictionResult
        Экземпляр датакласса, содержащий предсказанную метку, уверенность,
        полное распределение вероятностей и время прямого прохода.

    Raises
    ------
    FileNotFoundError
        Если ``image_path`` не указывает на существующий файл.
    OSError
        Если Pillow не может открыть файл (повреждённый или неподдерживаемый формат).

    Examples
    --------
    >>> from config import Config, get_val_transforms
    >>> from predict import load_model, predict_image
    >>> import torch
    >>> model = load_model(
    ...     Path("saved_models/ResNet18_best.pth"),
    ...     "ResNet18", 2, torch.device("cpu")
    ... )  # doctest: +SKIP
    >>> result = predict_image(
    ...     "test_spot.jpg", model, get_val_transforms(),
    ...     torch.device("cpu"), Config.CLASS_NAMES,
    ... )  # doctest: +SKIP
    """
    image_path_obj = Path(image_path)
    if not image_path_obj.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    logger.debug("Opening image '%s'.", image_path)

    # Загружаем и конвертируем в RGB, чтобы grayscale и RGBA изображения обрабатывались
    # единообразно для всех пяти архитектур.
    pil_image: Image.Image = Image.open(image_path_obj).convert("RGB")

    # Применяем предобработку на время валидации (resize → centre-crop → normalise).
    tensor: torch.Tensor = transform(pil_image)  # shape: (C, H, W)

    # Добавляем размерность батча, необходимую для nn.Module: (1, C, H, W).
    tensor = tensor.unsqueeze(0).to(device)

    # Прямой проход — отключаем вычисление градиентов для ускорения инференса.
    t_start = time.perf_counter()
    with torch.no_grad():
        logits: torch.Tensor = model(tensor)  # shape: (1, num_classes)
    t_end = time.perf_counter()
    inference_ms: float = (t_end - t_start) * 1_000.0

    # Преобразуем логиты в вероятности через softmax по размерности классов.
    probs: torch.Tensor = F.softmax(logits, dim=1).squeeze(0)  # shape: (num_classes,)

    # Определяем победивший класс.
    predicted_idx: int = int(probs.argmax().item())
    confidence: float = float(probs[predicted_idx].item())
    label: str = class_names[predicted_idx]

    # Формируем полный маппинг вероятностей.
    probabilities: dict[str, float] = {
        name: float(probs[i].item()) for i, name in enumerate(class_names)
    }

    result = PredictionResult(
        label=label,
        confidence=confidence,
        probabilities=probabilities,
        inference_time_ms=inference_ms,
    )

    logger.info(
        "Prediction for '%s': label=%s, confidence=%.4f, time=%.2f ms.",
        image_path,
        label,
        confidence,
        inference_ms,
    )
    return result


# ---------------------------------------------------------------------------
# ParkingPredictor — высокоуровневый класс-обёртка
# ---------------------------------------------------------------------------

class ParkingPredictor:
    """
    Высокоуровневая, stateful обёртка для инференса классификации парковочных мест.

    ``ParkingPredictor`` загружает модель один раз в ``__init__`` и предоставляет
    простой метод ``.predict()``, так что вызывающему коду не нужно управлять моделью,
    устройством или трансформацией самостоятельно. Этот класс используется в ``app.py``
    (интерфейс Streamlit).

    Parameters
    ----------
    model_path : str | Path
        Путь к сохранённому чекпоинту ``.pth``.
    model_name : str
        Идентификатор архитектуры, один из ``Config.MODEL_NAMES``.
    num_classes : int, optional
        Количество выходных классов. По умолчанию ``Config.NUM_CLASSES`` (2).
    device : torch.device | None, optional
        Целевое вычислительное устройство. Если ``None`` (по умолчанию), используется
        ``Config.DEVICE`` (CUDA > MPS > CPU).

    Attributes
    ----------
    model_name : str
        Имя загруженной архитектуры.
    model_path : Path
        Разрешённый путь к файлу чекпоинта.
    device : torch.device
        Устройство, на которое загружена модель.
    class_names : tuple[str, ...]
        Метки классов из ``Config.CLASS_NAMES``.

    Examples
    --------
    >>> predictor = ParkingPredictor(
    ...     model_path="saved_models/ResNet18_best.pth",
    ...     model_name="ResNet18",
    ... )  # doctest: +SKIP
    >>> result = predictor.predict("parking_spot.jpg")  # doctest: +SKIP
    >>> print(result.label, result.confidence)          # doctest: +SKIP
    Occupied 0.9912
    """

    def __init__(
        self,
        model_path: str | Path,
        model_name: str,
        num_classes: int = Config.NUM_CLASSES,
        device: torch.device | None = None,
    ) -> None:
        """
        Загружает модель с диска и подготавливает трансформацию предобработки.

        Parameters
        ----------
        model_path : str | Path
            Путь к сохранённому чекпоинту (файл ``.pth``).
        model_name : str
            Имя архитектуры для построения правильного скелета модели.
        num_classes : int, optional
            Количество классов. По умолчанию ``Config.NUM_CLASSES`` (2).
        device : torch.device | None, optional
            Устройство для инференса. По умолчанию ``Config.DEVICE``.
        """
        self.model_path: Path = Path(model_path)
        self.model_name: str = model_name
        self.device: torch.device = device if device is not None else Config.DEVICE
        self.class_names: tuple[str, ...] = Config.CLASS_NAMES
        self._num_classes: int = num_classes

        logger.info(
            "Initialising ParkingPredictor — arch=%s, device=%s.",
            model_name,
            self.device,
        )

        # Загружаем модель один раз; она будет использоваться повторно при каждом последующем вызове.
        self._model: nn.Module = load_model(
            model_path=self.model_path,
            model_name=self.model_name,
            num_classes=self._num_classes,
            device=self.device,
        )

        # Трансформация на время валидации (без аугментации, детерминированная).
        self._transform: transforms.Compose = get_val_transforms()

        logger.info("ParkingPredictor ready.")

    def predict(self, image_path: str | Path) -> PredictionResult:
        """
        Классифицирует одно изображение парковочного места как ``"Empty"`` или ``"Occupied"``.

        Parameters
        ----------
        image_path : str | Path
            Путь в файловой системе к классифицируемому изображению.

        Returns
        -------
        PredictionResult
            Датакласс с полями ``label``, ``confidence``, ``probabilities`` и
            ``inference_time_ms``.

        Raises
        ------
        FileNotFoundError
            Если ``image_path`` не существует.
        OSError
            Если файл изображения не может быть открыт Pillow.

        Examples
        --------
        >>> result = predictor.predict("lot_A_007.jpg")  # doctest: +SKIP
        >>> result.label
        'Empty'
        >>> result.confidence > 0.5
        True
        """
        return predict_image(
            image_path=str(image_path),
            model=self._model,
            transform=self._transform,
            device=self.device,
            class_names=self.class_names,
        )

    def __repr__(self) -> str:
        return (
            f"ParkingPredictor("
            f"model_name={self.model_name!r}, "
            f"model_path={self.model_path!r}, "
            f"device={self.device!r})"
        )


# ---------------------------------------------------------------------------
# Точка входа командной строки
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    """Возвращает парсер аргументов для точки входа CLI."""
    parser = argparse.ArgumentParser(
        prog="predict",
        description=(
            "Classify a single parking-spot image as Empty or Occupied "
            "using one of the five trained models."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        required=True,
        metavar="PATH",
        help="Path to the saved .pth checkpoint file.",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        choices=list(Config.MODEL_NAMES),
        metavar="NAME",
        help=(
            "Architecture name.  One of: "
            + ", ".join(f'"{n}"' for n in Config.MODEL_NAMES)
            + "."
        ),
    )
    parser.add_argument(
        "--image",
        required=True,
        metavar="PATH",
        help="Path to the parking-spot image to classify.",
    )
    parser.add_argument(
        "--device",
        default=str(Config.DEVICE),
        metavar="DEVICE",
        help="Torch device string, e.g. 'cpu', 'cuda', 'mps'.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """
    Точка входа CLI для инференса на одном изображении.

    Парсит аргументы командной строки, загружает указанную модель, классифицирует
    переданное изображение и выводит результат в stdout.

    Parameters
    ----------
    argv : list[str] | None, optional
        Список аргументов для тестирования. По умолчанию ``sys.argv[1:]`` при
        значении ``None``.

    Examples
    --------
    .. code-block:: bash

        python predict.py \\
            --model-path saved_models/ResNet18_best.pth \\
            --model-name ResNet18 \\
            --image test_spot.jpg
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    device = torch.device(args.device)

    predictor = ParkingPredictor(
        model_path=args.model_path,
        model_name=args.model_name,
        device=device,
    )

    result = predictor.predict(args.image)

    print("-" * 48)
    print(f"  Image      : {args.image}")
    print(f"  Model      : {args.model_name}")
    print(f"  Label      : {result.label}")
    print(f"  Confidence : {result.confidence:.4f} ({result.confidence * 100:.2f} %)")
    print("  Probabilities:")
    for class_name, prob in result.probabilities.items():
        print(f"    {class_name:<12} {prob:.4f}")
    print(f"  Latency    : {result.inference_time_ms:.2f} ms")
    print("-" * 48)


if __name__ == "__main__":
    main()
