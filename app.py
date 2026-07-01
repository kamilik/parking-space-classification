"""
app.py — Веб-приложение Streamlit для классификации занятости парковочных мест.

Данный модуль реализует полнофункциональный веб-интерфейс для классификатора
парковочных мест. Пользователи могут загрузить изображение одного парковочного места и получить
предсказание (Empty / Occupied) вместе с оценкой уверенности.

Возможности
-----------
- Выбор модели из доступных чекпоинтов ``.pth``.
- Тёмная современная тема через инжекцию CSS.
- История предсказаний, хранящаяся в JSON (``history.json``).
- Боковая панель статистики: общее число проверок, количество Empty, количество Occupied, средняя
  уверенность, последние проверки.
- Полная таблица истории, отображаемая через ``st.dataframe``.

Запуск
------
    cd project
    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Логгер уровня модуля — настраивается до импорта Streamlit, чтобы ранние сообщения
# перехватывались, если пользователь запускает модуль самостоятельно.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

import streamlit as st

from config import Config
from predict import ParkingPredictor, PredictionResult

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

#: Абсолютный путь к директории с обученными чекпоинтами ``.pth``.
SAVED_MODELS_DIR: Path = Config.SAVED_MODELS_DIR

#: Абсолютный путь к JSON-файлу истории.
HISTORY_FILE: Path = Config.HISTORY_FILE

#: Поддерживаемые MIME-типы / расширения изображений для загрузчика файлов.
ALLOWED_EXTENSIONS: list[str] = ["jpg", "jpeg", "png"]

#: Количество последних проверок, отображаемых в боковой панели.
RECENT_CHECKS_LIMIT: int = 10

# ---------------------------------------------------------------------------
# CSS тёмной темы
# ---------------------------------------------------------------------------

_DARK_THEME_CSS: str = """
<style>
/* ── Базовый фон ─────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background-color: #0e1117 !important;
    color: #e0e0e0 !important;
}

[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #30363d !important;
}

/* ── Заголовки ───────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
    font-family: "Inter", "Segoe UI", sans-serif;
}

h1 {
    font-size: 2.4rem !important;
    letter-spacing: -0.5px;
}

/* ── Основной текст ──────────────────────────────────────────────── */
p, label, span, div {
    color: #c9d1d9 !important;
    font-family: "Inter", "Segoe UI", sans-serif;
}

/* ── Кнопки ──────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: opacity 0.2s ease !important;
}

[data-testid="stButton"] > button:hover {
    opacity: 0.85 !important;
}

/* Кнопка очистки истории (вторичный / деструктивный стиль) */
button[kind="secondary"] {
    background: linear-gradient(135deg, #b91c1c, #dc2626) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
}

/* ── Загрузчик файлов ────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background-color: #161b22 !important;
    border: 2px dashed #30363d !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}

/* ── Выпадающий список ───────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background-color: #21262d !important;
    color: #e0e0e0 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}

/* ── Карточки метрик ─────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 0.8rem 1rem !important;
}

[data-testid="stMetricValue"] {
    color: #58a6ff !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: #8b949e !important;
}

/* ── DataFrame ───────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* ── Блоки info / success / error ────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
}

/* ── Разделитель ─────────────────────────────────────────────────── */
hr {
    border-color: #30363d !important;
}

/* ── Полоса прокрутки ────────────────────────────────────────────── */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0e1117;
}
::-webkit-scrollbar-thumb {
    background: #30363d;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #484f58;
}
</style>
"""

# ---------------------------------------------------------------------------
# Вспомогательные функции для истории
# ---------------------------------------------------------------------------


def load_history(path: Path) -> list[dict[str, Any]]:
    """
    Загружает историю предсказаний из JSON-файла.

    Если файл не существует или содержит некорректный JSON, возвращается
    пустой список и записывается предупреждение в лог.

    Parameters
    ----------
    path:
        Путь в файловой системе к JSON-файлу истории.

    Returns
    -------
    list[dict[str, Any]]
        Список словарей записей истории. Каждая запись имеет вид::

            {
                "date":       "2026-06-30",
                "time":       "14:30:00",
                "filename":   "spot_001.jpg",
                "result":     "Empty",
                "confidence": 0.9512,
            }
    """
    if not path.exists():
        logger.info("History file not found at '%s'; returning empty history.", path)
        return []

    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            logger.warning(
                "History file '%s' does not contain a JSON array; resetting.", path
            )
            return []
        logger.debug("Loaded %d history entries from '%s'.", len(data), path)
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Failed to read history file '%s': %s; returning empty history.", path, exc
        )
        return []


def save_history(history: list[dict[str, Any]], path: Path) -> None:
    """
    Сохраняет историю предсказаний в JSON-файл.

    Файл записывается атомарно (во временный файл рядом, затем переименовывается),
    чтобы сбой во время записи не повредил существующую историю.

    Parameters
    ----------
    history:
        Список словарей записей истории для сериализации.
    path:
        Путь в файловой системе к целевому JSON-файлу.

    Raises
    ------
    OSError
        Если в директорию невозможно выполнить запись.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Записываем во временный файл в той же директории, затем заменяем.
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
        ) as tmp_fh:
            json.dump(history, tmp_fh, ensure_ascii=False, indent=2)
            tmp_path = Path(tmp_fh.name)
        tmp_path.replace(path)
        logger.debug("History saved (%d entries) to '%s'.", len(history), path)
    except OSError as exc:
        logger.error("Failed to save history to '%s': %s", path, exc)
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _build_history_entry(
    filename: str,
    result: PredictionResult,
) -> dict[str, Any]:
    """
    Формирует словарь одной записи истории из результата предсказания.

    Parameters
    ----------
    filename:
        Оригинальное имя загруженного изображения.
    result:
        ``PredictionResult``, возвращённый ``ParkingPredictor.predict()``.

    Returns
    -------
    dict[str, Any]
        Одна запись истории в формате, ожидаемом ``save_history``.
    """
    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "filename": filename,
        "result": result.label,
        "confidence": round(result.confidence, 4),
    }


# ---------------------------------------------------------------------------
# Обнаружение моделей
# ---------------------------------------------------------------------------


def _discover_models() -> list[tuple[str, Path]]:
    """
    Сканирует директорию сохранённых моделей на наличие файлов чекпоинтов ``.pth``.

    Возвращает список пар ``(display_name, path)``, отсортированных в алфавитном порядке.
    Отображаемое имя формируется из имени файла без расширения, например
    ``"ResNet18_best.pth"`` → ``"ResNet18"``.

    Returns
    -------
    list[tuple[str, Path]]
        Отсортированный список пар ``(model_name, model_path)``. Пустой, если
        директория не существует или не содержит файлов ``.pth``.
    """
    if not SAVED_MODELS_DIR.exists():
        return []

    pairs: list[tuple[str, Path]] = []
    for pth_file in sorted(SAVED_MODELS_DIR.glob("*.pth")):
        # Преобразуем имя файла в отображаемое имя, убирая суффиксы обучения.
        stem = pth_file.stem  # например "ResNet18_best" или "EfficientNet-B0_best"
        display_name = stem.replace("_best", "").replace("_checkpoint", "")
        pairs.append((display_name, pth_file))
    return pairs


@st.cache_resource(show_spinner="Loading model …")
def _load_predictor(model_name: str, model_path_str: str) -> ParkingPredictor:
    """
    Загружает ``ParkingPredictor`` и кэширует его между перезапусками Streamlit.

    Декоратор ``@st.cache_resource`` гарантирует, что модель загружается
    с диска только один раз для каждой комбинации (model_name, model_path), независимо
    от того, сколько раз страница перезапускается.

    Parameters
    ----------
    model_name:
        Идентификатор архитектуры, например ``"ResNet18"``.
    model_path_str:
        Строковое представление пути к чекпоинту (строки хешируются кэшем
        Streamlit, в отличие от ``pathlib.Path``).

    Returns
    -------
    ParkingPredictor
        Готовый к использованию экземпляр предиктора.
    """
    logger.info(
        "Cache miss — initialising ParkingPredictor: arch=%s, path=%s",
        model_name,
        model_path_str,
    )
    return ParkingPredictor(
        model_path=model_path_str,
        model_name=model_name,
    )


# ---------------------------------------------------------------------------
# Вспомогательные функции для статистики
# ---------------------------------------------------------------------------


def _compute_statistics(
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Вычисляет агрегированную статистику из истории предсказаний.

    Parameters
    ----------
    history:
        Полный список словарей записей истории.

    Returns
    -------
    dict[str, Any]
        Словарь с ключами:
        - ``"total"``   — общее количество проверок (int)
        - ``"empty"``   — количество предсказаний Empty (int)
        - ``"occupied"``— количество предсказаний Occupied (int)
        - ``"avg_conf"``— средняя уверенность по всем записям (float), или 0.0
          если история пуста.
    """
    total = len(history)
    if total == 0:
        return {"total": 0, "empty": 0, "occupied": 0, "avg_conf": 0.0}

    empty_count = sum(1 for e in history if e.get("result") == "Empty")
    occupied_count = sum(1 for e in history if e.get("result") == "Occupied")
    confidences = [
        e["confidence"] for e in history if isinstance(e.get("confidence"), (int, float))
    ]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "total": total,
        "empty": empty_count,
        "occupied": occupied_count,
        "avg_conf": avg_conf,
    }


# ---------------------------------------------------------------------------
# Вспомогательные функции разметки страницы Streamlit
# ---------------------------------------------------------------------------


def _render_sidebar(
    history: list[dict[str, Any]],
    available_models: list[tuple[str, Path]],
) -> tuple[str | None, Path | None]:
    """
    Отрисовывает боковую панель: выбор модели, статистика, последние проверки, кнопка очистки.

    Parameters
    ----------
    history:
        Текущий список истории предсказаний.
    available_models:
        Список пар ``(display_name, path)``, возвращённых
        ``_discover_models()``.

    Returns
    -------
    tuple[str | None, Path | None]
        ``(selected_model_name, selected_model_path)`` — оба ``None``, если нет
        доступных моделей.
    """
    with st.sidebar:
        st.markdown("## Parking Spot Classifier")
        st.markdown("---")

        # ── Выбор модели ───────────────────────────────────────────────────
        st.markdown("### Model Selection")

        selected_name: str | None = None
        selected_path: Path | None = None

        if not available_models:
            st.warning(
                "No trained models found in `saved_models/`.\n\n"
                "Train the models first by running `train.py`."
            )
        else:
            model_labels = [name for name, _ in available_models]
            model_index = st.selectbox(
                label="Select architecture",
                options=range(len(model_labels)),
                format_func=lambda i: model_labels[i],
                index=0,
                key="model_selector",
            )
            selected_name = model_labels[model_index]
            selected_path = available_models[model_index][1]
            st.caption(f"Checkpoint: `{selected_path.name}`")

        st.markdown("---")

        # ── Статистика ─────────────────────────────────────────────────────
        st.markdown("### Statistics")
        stats = _compute_statistics(history)

        col_left, col_right = st.columns(2)
        with col_left:
            st.metric(label="Total Checks", value=stats["total"])
            st.metric(label="Empty", value=stats["empty"])
        with col_right:
            st.metric(label="Occupied", value=stats["occupied"])
            avg_pct = f'{stats["avg_conf"] * 100:.1f}%' if stats["total"] > 0 else "—"
            st.metric(label="Avg Confidence", value=avg_pct)

        st.markdown("---")

        # ── Последние проверки ─────────────────────────────────────────────
        st.markdown("### Recent Checks")
        recent = history[-RECENT_CHECKS_LIMIT:][::-1]  # сначала новейшие, не более 10

        if not recent:
            st.caption("No predictions yet.")
        else:
            for entry in recent:
                result_label = entry.get("result", "?")
                conf_pct = entry.get("confidence", 0.0) * 100
                color = "#3fb950" if result_label == "Empty" else "#f85149"
                fname = entry.get("filename", "unknown")
                time_str = entry.get("time", "")

                st.markdown(
                    f'<div style="border-left: 3px solid {color}; '
                    f'padding: 4px 8px; margin-bottom: 6px; '
                    f'background: #161b22; border-radius: 4px;">'
                    f'<span style="color:{color}; font-weight:700;">{result_label}</span> '
                    f'<span style="color:#8b949e; font-size:0.85rem;">{conf_pct:.1f}% — {fname}</span><br/>'
                    f'<span style="color:#484f58; font-size:0.75rem;">{time_str}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("---")

        # ── Кнопка очистки истории ─────────────────────────────────────────
        if st.button("Clear History", type="secondary", use_container_width=True):
            save_history([], HISTORY_FILE)
            logger.info("History cleared by user.")
            st.session_state["history"] = []
            st.rerun()

    return selected_name, selected_path


def _render_result_card(result: PredictionResult) -> None:
    """
    Отрисовывает стилизованную карточку результата с предсказанной меткой и уверенностью.

    Parameters
    ----------
    result:
        ``PredictionResult`` для отображения.
    """
    is_empty = result.label == "Empty"
    color = "#3fb950" if is_empty else "#f85149"
    icon = "🅿" if is_empty else "🚗"
    label_text = result.label.upper()
    conf_pct = result.confidence * 100

    st.markdown(
        f"""
        <div style="
            background: #161b22;
            border: 2px solid {color};
            border-radius: 12px;
            padding: 2rem 2.5rem;
            text-align: center;
            margin: 1rem 0;
        ">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">{icon}</div>
            <div style="
                font-size: 2.5rem;
                font-weight: 800;
                color: {color};
                letter-spacing: 2px;
                margin-bottom: 0.5rem;
            ">{label_text}</div>
            <div style="
                font-size: 1.4rem;
                color: #c9d1d9;
                font-weight: 500;
            ">Confidence: <strong style="color:{color};">{conf_pct:.2f}%</strong></div>
            <div style="margin-top: 1rem;">
                <div style="
                    display: inline-block;
                    background: #21262d;
                    border-radius: 20px;
                    padding: 0.3rem 1rem;
                    font-size: 0.9rem;
                    color: #8b949e;
                ">
                    Empty: {result.probabilities.get("Empty", 0.0)*100:.1f}% &nbsp;|&nbsp;
                    Occupied: {result.probabilities.get("Occupied", 0.0)*100:.1f}%
                </div>
            </div>
            <div style="
                margin-top: 0.75rem;
                font-size: 0.8rem;
                color: #484f58;
            ">
                Inference time: {result.inference_time_ms:.2f} ms
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_history_table(history: list[dict[str, Any]]) -> None:
    """
    Отрисовывает полную историю предсказаний в виде интерактивного датафрейма с
    русскими заголовками столбцов и кнопкой скачивания CSV.

    Самые последние записи отображаются первыми.

    Parameters
    ----------
    history:
        Полный список словарей записей истории.
    """
    if not history:
        st.info("No predictions in history yet.  Upload an image and click Определить.")
        return

    import pandas as pd  # imported locally to avoid mandatory top-level dep

    # Переворачиваем, чтобы новейшие записи были вверху.
    reversed_history = list(reversed(history))

    df = pd.DataFrame(reversed_history, columns=["date", "time", "filename", "result", "confidence"])
    df.columns = ["Дата", "Время", "Имя файла", "Результат", "Confidence"]
    df["Confidence"] = df["Confidence"].apply(lambda x: f"{x * 100:.2f}%")
    df.index = range(1, len(df) + 1)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=False,
    )

    # Кнопка скачивания CSV.
    csv_data: str = df.to_csv(index=False, encoding="utf-8")
    st.download_button(
        label="Скачать историю CSV",
        data=csv_data,
        file_name="history.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Основное приложение Streamlit
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Точка входа приложения Streamlit.

    Данная функция настраивает страницу, инжектирует CSS тёмной темы, отрисовывает
    боковую панель и основную область контента, обрабатывает загрузку файлов, запускает предсказания и
    ведёт историю предсказаний.
    """
    # ── Конфигурация страницы ──────────────────────────────────────────────
    st.set_page_config(
        page_title="Parking Spot Classifier",
        page_icon="🅿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Инжекция CSS тёмной темы ───────────────────────────────────────────
    st.markdown(_DARK_THEME_CSS, unsafe_allow_html=True)

    # ── Инициализация состояния сессии ─────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state["history"] = load_history(HISTORY_FILE)

    history: list[dict[str, Any]] = st.session_state["history"]

    # ── Обнаружение доступных моделей ─────────────────────────────────────
    available_models = _discover_models()

    # ── Боковая панель ─────────────────────────────────────────────────────
    selected_model_name, selected_model_path = _render_sidebar(
        history=history,
        available_models=available_models,
    )

    # ── Основная область контента ──────────────────────────────────────────
    st.markdown(
        "<h1>Parking Spot Occupancy Classifier</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:1.1rem; color:#8b949e; margin-top:-0.5rem;'>"
        "Upload an image of a single parking space to determine whether it is "
        "<strong style='color:#3fb950;'>Empty</strong> or "
        "<strong style='color:#f85149;'>Occupied</strong>."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Двухколоночная разметка: загрузчик слева, результат справа ────────
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown("### Upload Image")
        uploaded_file = st.file_uploader(
            label="Choose a parking spot image",
            type=ALLOWED_EXTENSIONS,
            accept_multiple_files=False,
            key="file_uploader",
        )

        if uploaded_file is not None:
            st.image(
                uploaded_file,
                caption=f"Uploaded: {uploaded_file.name}",
                use_container_width=True,
            )

    with col_result:
        st.markdown("### Result")

        if uploaded_file is None:
            st.markdown(
                "<div style='color:#484f58; padding: 2rem; text-align:center;'>"
                "No image uploaded yet."
                "</div>",
                unsafe_allow_html=True,
            )

        else:
            # Показываем кнопку «Определить» только при наличии изображения.
            determine_clicked = st.button(
                "Определить",
                type="primary",
                use_container_width=True,
                key="determine_button",
            )

            if determine_clicked:
                if selected_model_name is None or selected_model_path is None:
                    st.error(
                        "No model selected.  Train the models first and place "
                        "`*.pth` files inside the `saved_models/` directory."
                    )
                else:
                    # Сохраняем загруженные байты во временный файл, чтобы
                    # ParkingPredictor (ожидающий путь в файловой системе) мог
                    # его прочитать. Временный файл автоматически удаляется
                    # после выхода из блока `with`.
                    suffix = Path(uploaded_file.name).suffix or ".jpg"
                    with tempfile.NamedTemporaryFile(
                        suffix=suffix, delete=False
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_image_path = Path(tmp.name)

                    try:
                        with st.spinner(
                            f"Running {selected_model_name} inference …"
                        ):
                            predictor = _load_predictor(
                                model_name=selected_model_name,
                                model_path_str=str(selected_model_path),
                            )
                            prediction: PredictionResult = predictor.predict(
                                tmp_image_path
                            )

                        # Отрисовываем карточку результата.
                        _render_result_card(prediction)

                        # Добавляем в историю и сохраняем.
                        entry = _build_history_entry(
                            filename=uploaded_file.name,
                            result=prediction,
                        )
                        history.append(entry)
                        st.session_state["history"] = history
                        save_history(history, HISTORY_FILE)

                        logger.info(
                            "Prediction recorded: file=%s, label=%s, confidence=%.4f",
                            uploaded_file.name,
                            prediction.label,
                            prediction.confidence,
                        )

                    except FileNotFoundError as exc:
                        st.error(f"Model file not found: {exc}")
                        logger.error("Model not found: %s", exc)
                    except RuntimeError as exc:
                        st.error(f"Inference error: {exc}")
                        logger.error("Inference error: %s", exc)
                    except Exception as exc:  # pylint: disable=broad-except
                        st.error(f"Unexpected error during prediction: {exc}")
                        logger.exception("Unexpected prediction error.")
                    finally:
                        # Всегда удаляем временный файл изображения.
                        if tmp_image_path.exists():
                            tmp_image_path.unlink(missing_ok=True)

    # ── Таблица истории ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Prediction History")
    _render_history_table(history)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
