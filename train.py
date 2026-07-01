"""
train.py — Полный конвейер обучения классификатора занятости парковочных мест.

Данный модуль организует сквозное обучение и оценку для всех пяти
поддерживаемых архитектур (ResNet18, DenseNet121, EfficientNet-B0,
MobileNetV3-Large, ViT-B/16).

Публичный API
-------------
- ``train_one_epoch(model, dataloader, criterion, optimizer, device)``
      Выполняет один прямой/обратный проход по всем батчам; возвращает (loss, accuracy).
- ``evaluate(model, dataloader, criterion, device)``
      Оценивает модель на DataLoader; возвращает loss, accuracy и сырые массивы.
- ``train_model(model_name, train_loader, val_loader, test_loader, config)``
      Полный конвейер обучения для одной архитектуры; возвращает словарь результатов.
- ``run_full_pipeline(config)``
      Последовательно обучает все пять моделей, сохраняет таблицы и текст анализа.

Использование
-------------
    python train.py                   # обучает все модели с конфигурацией по умолчанию
    from train import run_full_pipeline
    from config import Config
    results = run_full_pipeline(Config)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config, get_train_transforms, get_val_transforms, seed_everything
from dataset import download_and_prepare_dataset, get_data_loaders
from models import get_model, get_model_info
from utils import (
    EarlyStopping,
    ModelCheckpoint,
    analyze_results,
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_curves,
    save_comparison_table,
)

# ---------------------------------------------------------------------------
# Логгер уровня модуля
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# train_one_epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Выполняет одну полную эпоху обучения по всем батчам в ``dataloader``.

    Перед началом итерации модель переводится в режим обучения. Для каждого
    мини-батча прямой проход вычисляет функцию потерь, обратный проход вычисляет
    градиенты, и оптимизатор обновляет веса.

    Во время итерации отображается прогресс-бар ``tqdm``, показывающий
    скользящее среднее потерь и точности за текущую эпоху.

    Parameters
    ----------
    model:
        Модель PyTorch для обучения. Должна быть уже перемещена на ``device``.
    dataloader:
        DataLoader, возвращающий мини-батчи ``(images, labels)`` из
        обучающей выборки.
    criterion:
        Функция потерь (``nn.CrossEntropyLoss``).
    optimizer:
        Градиентный оптимизатор (``torch.optim.Adam``).
    device:
        Вычислительное устройство (``torch.device("cuda")``, ``"mps"`` или ``"cpu"``).

    Returns
    -------
    tuple[float, float]
        ``(avg_loss, accuracy)`` — среднее значение кросс-энтропийных потерь и доля
        правильно классифицированных образцов за всю эпоху.
    """
    model.train()

    running_loss: float = 0.0
    correct: int = 0
    total: int = 0

    progress_bar = tqdm(
        dataloader,
        desc="  Train",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )

    for images, labels in progress_bar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Обнуляем градиенты перед прямым проходом.
        optimizer.zero_grad()

        # Прямой проход.
        logits: torch.Tensor = model(images)
        loss: torch.Tensor = criterion(logits, labels)

        # Обратный проход и обновление весов.
        loss.backward()
        optimizer.step()

        # Накапливаем статистику.
        batch_size: int = labels.size(0)
        running_loss += loss.item() * batch_size
        predicted: torch.Tensor = logits.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += batch_size

        # Обновляем постфикс прогресс-бара скользящими средними.
        current_loss = running_loss / total
        current_acc = correct / total
        progress_bar.set_postfix(
            loss=f"{current_loss:.4f}",
            acc=f"{current_acc:.4f}",
        )

    progress_bar.close()

    avg_loss: float = running_loss / total if total > 0 else 0.0
    accuracy: float = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Оценивает модель на всех батчах из ``dataloader`` без обновления весов.

    Модель переводится в режим оценки, а все вычисления градиентов отключаются
    через ``torch.no_grad()``. Предсказания, истинные метки и вероятности
    класса 1 собираются по всем батчам.

    Parameters
    ----------
    model:
        Модель PyTorch для оценки. Должна быть уже перемещена на ``device``.
    dataloader:
        DataLoader, возвращающий мини-батчи ``(images, labels)``. Обычно
        DataLoader валидационной или тестовой выборки.
    criterion:
        Функция потерь (``nn.CrossEntropyLoss``), используемая для вычисления
        средних потерь по всем образцам.
    device:
        Вычислительное устройство.

    Returns
    -------
    tuple[float, float, np.ndarray, np.ndarray, np.ndarray]
        - ``avg_loss``  : float — среднее значение кросс-энтропийных потерь.
        - ``accuracy``  : float — доля правильно классифицированных образцов.
        - ``y_true``    : одномерный массив numpy int32 с истинными индексами классов.
        - ``y_pred``    : одномерный массив numpy int32 с предсказанными индексами классов.
        - ``y_proba``   : одномерный массив numpy float32 с вероятностью для класса 1
                          (Occupied), полученной через ``F.softmax``.
    """
    model.eval()

    running_loss: float = 0.0
    correct: int = 0
    total: int = 0

    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_proba: list[np.ndarray] = []

    with torch.no_grad():
        progress_bar = tqdm(
            dataloader,
            desc="  Eval ",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

        for images, labels in progress_bar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits: torch.Tensor = model(images)
            loss: torch.Tensor = criterion(logits, labels)

            batch_size: int = labels.size(0)
            running_loss += loss.item() * batch_size
            predicted: torch.Tensor = logits.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += batch_size

            # Вычисляем вероятности softmax; оставляем только вероятность класса 1 (Occupied).
            probabilities: torch.Tensor = F.softmax(logits, dim=1)
            class1_proba: torch.Tensor = probabilities[:, 1]

            all_true.append(labels.cpu().numpy().astype(np.int32))
            all_pred.append(predicted.cpu().numpy().astype(np.int32))
            all_proba.append(class1_proba.cpu().numpy().astype(np.float32))

        progress_bar.close()

    avg_loss: float = running_loss / total if total > 0 else 0.0
    accuracy: float = correct / total if total > 0 else 0.0

    y_true: np.ndarray = np.concatenate(all_true, axis=0)
    y_pred: np.ndarray = np.concatenate(all_pred, axis=0)
    y_proba: np.ndarray = np.concatenate(all_proba, axis=0)

    return avg_loss, accuracy, y_true, y_pred, y_proba


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

def train_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config: type[Config],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Полный конвейер обучения для одной архитектуры.

    Выполняемые шаги
    ----------------
    1. Строит модель через ``get_model()`` и перемещает её на ``config.DEVICE``.
    2. Создаёт оптимизатор ``Adam`` и функцию потерь ``CrossEntropyLoss``.
    3. Подключает колбэки ``EarlyStopping`` и ``ModelCheckpoint``.
    4. Цикл обучения: для каждой эпохи вызывает ``train_one_epoch()``, затем
       ``evaluate()`` на валидационной выборке; передаёт val loss в оба
       колбэка; досрочно завершает, если срабатывает ``EarlyStopping``.
    5. Загружает лучший чекпоинт, сохранённый ``ModelCheckpoint``.
    6. Запускает ``evaluate()`` на тестовой выборке.
    7. Вычисляет все метрики классификации через ``compute_metrics()``.
    8. Измеряет время инференса на одно изображение, усреднённое по 100 одиночным
       прямым проходам на вычислительном устройстве.
    9. Получает размер модели и количество параметров через ``get_model_info()``.
    10. Строит графики кривых обучения, матрицы ошибок и ROC-кривой.
    11. Возвращает словарь результатов со всеми ключами, необходимыми для
        ``save_comparison_table()``.

    Parameters
    ----------
    model_name:
        Одно из канонических имён из ``Config.MODEL_NAMES``
        (например, ``"ResNet18"``).
    train_loader:
        DataLoader для обучающей выборки.
    val_loader:
        DataLoader для валидационной выборки.
    test_loader:
        DataLoader для тестовой выборки.
    config:
        Класс ``Config`` (не экземпляр), предоставляющий все гиперпараметры
        и пути к файловой системе.

    Returns
    -------
    dict[str, Any]
        Словарь со следующими ключами, совместимый с
        ``save_comparison_table()`` и ``analyze_results()``:

        - ``Architecture``   : str
        - ``Accuracy``       : float
        - ``Precision``      : float
        - ``Recall``         : float
        - ``F1``             : float
        - ``ROC_AUC``        : float
        - ``Inference_Time`` : float — среднее время в секундах на одно изображение
        - ``Parameters``     : int
        - ``Model_Size``     : float — МБ
        - ``Epochs``         : int — фактическое количество выполненных эпох
        - ``Training_Time``  : float — общее время обучения по настенным часам в секундах
    """
    device: torch.device = config.DEVICE
    safe_name: str = model_name.replace("/", "_").replace("-", "_")

    logger.info("=" * 64)
    logger.info("Training model: %s", model_name)
    logger.info("=" * 64)

    # ------------------------------------------------------------------
    # 1. Построение модели
    # ------------------------------------------------------------------
    model: nn.Module = get_model(
        model_name=model_name,
        num_classes=config.NUM_CLASSES,
        pretrained=True,
    )
    model = model.to(device)
    logger.info("Model '%s' moved to device: %s", model_name, device)

    # ------------------------------------------------------------------
    # 2. Оптимизатор и функция потерь
    # ------------------------------------------------------------------
    optimizer: torch.optim.Adam = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
    )
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # 3. Колбэки (оба отслеживают F1-меру на валидации — mode="max")
    # ------------------------------------------------------------------
    checkpoint_path: Path = config.SAVED_MODELS_DIR / f"{safe_name}_best.pth"
    early_stopping: EarlyStopping = EarlyStopping(patience=config.PATIENCE, mode="max")
    model_checkpoint: ModelCheckpoint = ModelCheckpoint(save_path=checkpoint_path, mode="max")

    # ------------------------------------------------------------------
    # 4. Цикл обучения
    # ------------------------------------------------------------------
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    training_start: float = time.perf_counter()
    epochs_trained: int = 0

    for epoch in range(1, config.NUM_EPOCHS + 1):
        logger.info(
            "Epoch %d/%d — model: %s",
            epoch,
            config.NUM_EPOCHS,
            model_name,
        )

        # Обучаем одну эпоху.
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        # Оцениваем на валидационной выборке.
        val_loss, val_acc, val_true, val_pred, val_proba = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Вычисляем F1-меру на валидации для колбэков.
        from sklearn.metrics import f1_score as sklearn_f1_score
        val_f1: float = float(sklearn_f1_score(val_true, val_pred, average="weighted", zero_division=0))

        # Записываем историю.
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        epochs_trained = epoch

        logger.info(
            "  train_loss=%.4f | train_acc=%.4f | val_loss=%.4f | val_acc=%.4f | val_f1=%.4f",
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_f1,
        )

        # ModelCheckpoint — сохраняем, если F1-мера на валидации улучшилась.
        model_checkpoint(val_f1, model)

        # EarlyStopping — останавливаем, если F1 не улучшается в течение PATIENCE эпох.
        if early_stopping(val_f1):
            logger.info(
                "Early stopping triggered at epoch %d for model '%s'.",
                epoch,
                model_name,
            )
            break

    training_end: float = time.perf_counter()
    training_time: float = training_end - training_start

    logger.info(
        "Training complete — %d epochs in %.1f s (%.1f min)",
        epochs_trained,
        training_time,
        training_time / 60.0,
    )

    # ------------------------------------------------------------------
    # 5. Загружаем лучший чекпоинт
    # ------------------------------------------------------------------
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logger.info("Best checkpoint loaded from '%s'.", checkpoint_path)
    else:
        logger.warning(
            "Checkpoint file '%s' not found — using weights from last epoch.",
            checkpoint_path,
        )

    # ------------------------------------------------------------------
    # 6. Оцениваем на тестовой выборке
    # ------------------------------------------------------------------
    logger.info("Evaluating best model on test set …")
    test_loss, test_acc, y_true, y_pred, y_proba = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )
    logger.info(
        "Test — loss=%.4f | accuracy=%.4f", test_loss, test_acc
    )

    # ------------------------------------------------------------------
    # 7. Вычисляем все метрики классификации
    # ------------------------------------------------------------------
    metrics: dict[str, float] = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
    )

    # ------------------------------------------------------------------
    # 8. Измеряем время инференса (среднее по 100 одиночным прямым проходам)
    # ------------------------------------------------------------------
    model.eval()
    # Берём одно изображение из тестового загрузчика для использования в качестве зонда.
    probe_images: torch.Tensor
    probe_images, _ = next(iter(test_loader))
    single_image: torch.Tensor = probe_images[0:1].to(device)

    # Прогреваем устройство перед замером времени.
    _WARMUP_RUNS: int = 10
    with torch.no_grad():
        for _ in range(_WARMUP_RUNS):
            _ = model(single_image)

    if device.type == "cuda":
        torch.cuda.synchronize()

    _TIMING_RUNS: int = 100
    inference_start: float = time.perf_counter()
    with torch.no_grad():
        for _ in range(_TIMING_RUNS):
            _ = model(single_image)
            if device.type == "cuda":
                torch.cuda.synchronize()
    inference_end: float = time.perf_counter()

    avg_inference_time: float = (inference_end - inference_start) / _TIMING_RUNS
    logger.info(
        "Inference time (avg over %d runs): %.4f ms",
        _TIMING_RUNS,
        avg_inference_time * 1000.0,
    )

    # ------------------------------------------------------------------
    # 9. Информация о модели (количество параметров и размер)
    # ------------------------------------------------------------------
    model_info: dict[str, Any] = get_model_info(model)
    total_params: int = model_info["total_params"]
    model_size_mb: float = model_info["model_size_mb"]

    # Используем размер чекпоинта на диске, если он доступен, для более точного значения.
    if checkpoint_path.exists():
        disk_size_mb: float = checkpoint_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Checkpoint on-disk size: %.2f MB (parameter estimate: %.2f MB)",
            disk_size_mb,
            model_size_mb,
        )
        model_size_mb = disk_size_mb

    # ------------------------------------------------------------------
    # 10. Строим графики
    # ------------------------------------------------------------------
    plots_dir: Path = output_dir if output_dir is not None else config.PLOTS_DIR

    plot_training_curves(
        history=history,
        model_name=safe_name,
        save_dir=plots_dir,
    )
    logger.info("Training curves saved for '%s'.", model_name)

    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=list(config.CLASS_NAMES),
        model_name=safe_name,
        save_dir=plots_dir,
    )
    logger.info("Confusion matrix saved for '%s'.", model_name)

    plot_roc_curve(
        y_true=y_true,
        y_proba=y_proba,
        model_name=safe_name,
        save_dir=plots_dir,
    )
    logger.info("ROC curve saved for '%s'.", model_name)

    # ------------------------------------------------------------------
    # 11. Формируем и возвращаем словарь результатов
    # ------------------------------------------------------------------
    inference_time_ms: float = avg_inference_time * 1000.0
    fps: float = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0

    results: dict[str, Any] = {
        "Architecture": model_name,
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1"],
        "ROC AUC": metrics["roc_auc"],
        "Training Time": training_time,
        "Inference Time (ms/image)": inference_time_ms,
        "FPS": fps,
        "Model Size (MB)": model_size_mb,
        "Number of Parameters": total_params,
        "Epochs": epochs_trained,
    }

    logger.info(
        "Results for '%s': acc=%.4f | f1=%.4f | auc=%.4f | "
        "params=%d | size=%.2f MB | epochs=%d | train_time=%.1f s | fps=%.1f",
        model_name,
        results["Accuracy"],
        results["F1"],
        results["ROC AUC"],
        results["Number of Parameters"],
        results["Model Size (MB)"],
        results["Epochs"],
        results["Training Time"],
        results["FPS"],
    )

    return results


# ---------------------------------------------------------------------------
# run_full_pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline(config: type[Config]) -> list[dict[str, Any]]:
    """
    Последовательно обучает все пять архитектур и создаёт все выходные артефакты.

    Шаги
    ----
    1. Фиксирует случайные зерна через ``seed_everything(config.SEED)``.
    2. Скачивает и подготавливает датасет PKLot через
       ``download_and_prepare_dataset(config.DATA_DIR)``.
    3. Создаёт экземпляры ``DataLoader`` с трансформациями для обучения и валидации.
    4. Перебирает все пять имён моделей из ``config.MODEL_NAMES`` и вызывает
       ``train_model()`` для каждой; собирает словари результатов.
    5. Сохраняет сравнительную таблицу по архитектурам (CSV + XLSX) через
       ``save_comparison_table()``.
    6. Генерирует и выводит текстовый анализ через ``analyze_results()``.
    7. Записывает текст анализа в ``config.RESULTS_DIR / "analysis.txt"``.
    8. Возвращает полный список словарей результатов.

    Parameters
    ----------
    config:
        Класс ``Config`` (не экземпляр), содержащий все настройки проекта.

    Returns
    -------
    list[dict[str, Any]]
        Один словарь результатов на каждую обученную модель в порядке,
        определённом ``config.MODEL_NAMES``.
    """
    # ------------------------------------------------------------------
    # 1. Воспроизводимость
    # ------------------------------------------------------------------
    seed_everything(config.SEED)

    # ------------------------------------------------------------------
    # 2. Подготовка датасета
    # ------------------------------------------------------------------
    logger.info("Preparing PKLot dataset …")
    data_root: Path = download_and_prepare_dataset(
        data_dir=config.DATA_DIR,
        seed=config.SEED,
    )

    # ------------------------------------------------------------------
    # 3. DataLoaders
    # ------------------------------------------------------------------
    logger.info("Building DataLoaders …")
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=data_root,
        batch_size=config.BATCH_SIZE,
        train_transform=get_train_transforms(),
        val_transform=get_val_transforms(),
    )
    logger.info(
        "DataLoaders ready — train batches: %d | val batches: %d | test batches: %d",
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )

    # ------------------------------------------------------------------
    # 4. Создаём директорию вывода с меткой времени
    # ------------------------------------------------------------------
    timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir: Path = config.RESULTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Results will be saved to '%s'.", run_dir)

    # ------------------------------------------------------------------
    # 5. Последовательно обучаем все модели
    # ------------------------------------------------------------------
    all_results: list[dict[str, Any]] = []

    for model_name in config.MODEL_NAMES:
        logger.info("")
        logger.info("Starting training pipeline for: %s", model_name)

        try:
            result: dict[str, Any] = train_model(
                model_name=model_name,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config=config,
                output_dir=run_dir,
            )
            all_results.append(result)
            logger.info("Finished training: %s", model_name)
        except Exception:
            logger.exception(
                "Training failed for model '%s'. Skipping to next model.",
                model_name,
            )

    if not all_results:
        logger.error("No models were successfully trained.")
        return all_results

    # ------------------------------------------------------------------
    # 6. Сохраняем сравнительную таблицу в директорию с меткой времени
    # ------------------------------------------------------------------
    logger.info("Saving comparison table …")
    save_comparison_table(
        results=all_results,
        save_dir=run_dir,
    )
    logger.info(
        "Comparison table saved to '%s'.", run_dir
    )

    # ------------------------------------------------------------------
    # 7. Сохраняем metrics.json
    # ------------------------------------------------------------------
    metrics_path: Path = run_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Metrics JSON saved to '%s'.", metrics_path)

    # ------------------------------------------------------------------
    # 8. Генерируем текстовый анализ
    # ------------------------------------------------------------------
    logger.info("Generating results analysis …")
    analysis_text: str = analyze_results(all_results)
    print(analysis_text)

    # ------------------------------------------------------------------
    # 9. Сохраняем анализ в директорию с меткой времени
    # ------------------------------------------------------------------
    analysis_path: Path = run_dir / "analysis.txt"
    analysis_path.write_text(analysis_text, encoding="utf-8")
    logger.info("Analysis text saved to '%s'.", analysis_path)

    logger.info("Full pipeline complete.  %d models trained.  Output: %s", len(all_results), run_dir)

    return all_results


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipeline_results: list[dict[str, Any]] = run_full_pipeline(Config)

    logger.info("")
    logger.info("Summary:")
    logger.info("%-20s  %8s  %8s  %8s", "Architecture", "Accuracy", "F1", "ROC AUC")
    logger.info("-" * 52)
    for res in sorted(pipeline_results, key=lambda r: r["F1"], reverse=True):
        logger.info(
            "%-20s  %8.4f  %8.4f  %8.4f",
            res["Architecture"],
            res["Accuracy"],
            res["F1"],
            res["ROC AUC"],
        )
