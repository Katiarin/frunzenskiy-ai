#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Сборка одной HTML-страницы с интерактивной картой и дашбордом.

Скрипт читает результаты, которые сохранил ноутбук
``examples/frunzensky_blocksnet.ipynb`` (папка ``outputs/``), и собирает
самодостаточный файл ``outputs/frunzensky_dashboard.html``:

* интерактивная карта кварталов района (folium) с переключателем показателей;
* карточки с ключевыми числами (KPI);
* графики (matplotlib, встроены в страницу как картинки base64);
* таблицы (обеспеченность по услугам, кварталы с худшей обеспеченностью);
* автоматические краткие выводы.

Запуск (Windows), из корня проекта::

    .venv\\Scripts\\python.exe scripts\\build_dashboard.py

Необязательные параметры::

    --input-dir  outputs                       папка с результатами
    --output     outputs\\frunzensky_dashboard.html   куда сохранить страницу

Страница не требует интернета (карта — из локальных данных, графики встроены).
"""

from __future__ import annotations

import argparse
import base64
import html as html_lib
import io
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без графического окна — сохраняем картинки в память
import matplotlib.pyplot as plt  # noqa: E402
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import folium  # noqa: E402
import branca.colormap as cm  # noqa: E402
from shapely.geometry import mapping  # noqa: E402

# --- Пути по умолчанию (относительно самого скрипта) -----------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_DIR / "outputs"
DEFAULT_OUTPUT = PROJECT_DIR / "outputs" / "frunzensky_dashboard.html"

INDICATORS_FILE = "frunzensky_indicators.geojson"
MATRIX_PARQUET = "accessibility_matrix_drive.parquet"
MATRIX_CSV = "accessibility_matrix_drive.csv"

# --- Описание числовых слоёв карты: (колонка, подпись, палитра) -------------------
# Палитра "YlOrRd_09": жёлтый -> красный (больше = насыщеннее).
# Палитра "YlGn_09":  жёлтый -> зелёный (больше = лучше).
NUMERIC_LAYERS = [
    ("population", "Население, чел.", "YlOrRd_09"),
    ("fsi", "Плотность застройки (FSI)", "YlOrRd_09"),
    ("gsi", "Доля застройки (GSI)", "YlOrRd_09"),
    ("mxi", "Доля жилья (MXI)", "YlGn_09"),
    ("l", "Средняя этажность (L)", "YlOrRd_09"),
    ("osr", "Открытые пространства (OSR)", "YlGn_09"),
    ("share_living", "Доля жилых площадей", "YlGn_09"),
    ("share_non_living", "Доля нежилых площадей", "YlOrRd_09"),
    ("area_accessibility", "Доступность (средневзвеш.), мин", "YlOrRd_09"),
    ("mean_accessibility", "Среднее время до кварталов, мин", "YlOrRd_09"),
    ("median_accessibility", "Медианное время, мин", "YlOrRd_09"),
]

# --- Категориальные слои карты ---------------------------------------------------
CATEGORICAL_LAYERS = [
    ("land_use", "Использование территории (land use)"),
    ("morphotype_spacematrix", "Морфотипы (Spacematrix)"),
    ("morphotype_strelka", "Морфотипы (Strelka)"),
]

# --- Палитра для категориальных слоёв ---------------------------------------------
CATEGORY_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]

# --- Читаемые подписи для land_use (в данных это имена enum) ----------------------
LAND_USE_LABELS = {
    "LandUse.RESIDENTIAL": "Жилая",
    "LandUse.TRANSPORT": "Транспорт",
    "LandUse.INDUSTRIAL": "Промышленная",
    "LandUse.RECREATION": "Рекреация",
    "LandUse.SPECIAL": "Специальная",
    "LandUse.BUSINESS": "Деловая",
}

MISSING_COLOR = "#d9d9d9"

# --- Бесплатные подложки карты (без API-ключа) ------------------------------------
ESRI_GRAY_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
)
ESRI_GRAY_ATTR = "Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ"
ESRI_IMAGERY_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
ESRI_IMAGERY_ATTR = "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics"
OSM_TILES = "OpenStreetMap"
OSM_ATTR = "&copy; OpenStreetMap contributors"


# =================================================================================
# Вспомогательные функции
# =================================================================================
def _log(message: str) -> None:
    print(f"[dashboard] {message}")


def _sanitize(value):
    """Заменяет NaN/inf/несериализуемые значения на None (для GeoJSON)."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (str, int)):
        return value
    return str(value)


def _gdf_to_features(gdf: gpd.GeoDataFrame, columns: list[str]) -> dict:
    """Превращает GeoDataFrame в FeatureCollection из обычного Python (без NaN внутри)."""
    features = []
    for _, row in gdf.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        properties = {"block_id": _sanitize(row.get("block_id"))}
        for column in columns:
            properties[column] = _sanitize(row.get(column))
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": properties,
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _fig_to_base64(fig) -> str:
    """Сохраняет фигуру matplotlib в base64-PNG для вставки в HTML."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _read_matrix(input_dir: Path) -> pd.DataFrame | None:
    """Загружает матрицу времени в пути, если она есть (parquet быстрее, csv — запасной)."""
    parquet_path = input_dir / MATRIX_PARQUET
    csv_path = input_dir / MATRIX_CSV
    try:
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        if csv_path.exists():
            return pd.read_csv(csv_path, index_col=0)
    except Exception as error:  # noqa: BLE001
        _log(f"не удалось прочитать матрицу доступности: {error}")
    return None


def _service_meta(service: str):
    """Норматив доступности и потребности для услуги из конфига BlocksNet."""
    try:
        from blocksnet.config import service_types_config

        info = service_types_config[service]
        return info.get("name_ru"), int(info["accessibility"]), int(info["demand"])
    except Exception:  # noqa: BLE001
        return None, None, None


# =================================================================================
# 1. Загрузка данных
# =================================================================================
def load_indicators(input_dir: Path) -> gpd.GeoDataFrame:
    path = input_dir / INDICATORS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл {path}.\n"
            "Сначала запустите ноутбук examples/frunzensky_blocksnet.ipynb до последней ячейки."
        )
    _log(f"читаем {path.name}")
    gdf = gpd.read_file(path)

    # добавляем идентификатор квартала, если его нет
    if "block_id" not in gdf.columns:
        gdf = gdf.reset_index().rename(columns={"index": "block_id"})

    # приводим геометрию к градусам (WGS84) — это требование карт
    if gdf.crs is None:
        _log("предупреждение: у слоя не задана система координат, считаем что это WGS84")
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    # land_use в данных хранится как имя enum (LandUse.RESIDENTIAL и т.п.)
    if "land_use" in gdf.columns:
        gdf["land_use"] = gdf["land_use"].map(LAND_USE_LABELS).fillna(gdf["land_use"])

    _log(f"кварталов: {len(gdf)}, столбцов: {gdf.shape[1]}")
    return gdf


def collect_provision_columns(gdf: gpd.GeoDataFrame) -> list[str]:
    # provision_<service> — строгая обеспеченность (provision_strong).
    # provision_weak_<service> — слабая; в таблице показывается отдельной колонкой.
    return sorted(
        c
        for c in gdf.columns
        if c.startswith("provision_") and not c.startswith("provision_weak_")
    )


# =================================================================================
# 2. Интерактивная карта
# =================================================================================
def _numeric_style(column: str, colormap):
    def style(feature):
        value = feature["properties"].get(column)
        if value is None:
            color = MISSING_COLOR
        else:
            color = colormap(value)[:7]  # убираем альфа-канал из hex (#RRGGBBAA -> #RRGGBB)
        return {"fillColor": color, "color": "#666666", "weight": 0.3, "fillOpacity": 0.75}

    return style


def _categorical_style(column: str, color_map: dict):
    def style(feature):
        value = feature["properties"].get(column)
        color = color_map.get(value, MISSING_COLOR)
        return {"fillColor": color, "color": "#666666", "weight": 0.3, "fillOpacity": 0.8}

    return style


def build_map(gdf: gpd.GeoDataFrame, provision_columns: list[str], tiles: bool = True) -> folium.Map:
    minx, miny, maxx, maxy = gdf.total_bounds
    center = [(miny + maxy) / 2, (minx + maxx) / 2]

    # tiles=None: базовых подложек по умолчанию нет — добавляем их сами (иначе CartoDB
    # из folium требует API-ключ и вместо карты показывается "API KEY REQUIRED").
    fmap = folium.Map(location=center, zoom_start=11, tiles=None, control_scale=True)
    fmap.fit_bounds([[miny, minx], [maxy, maxx]])

    if tiles:
        # Три бесплатные подложки без ключа; пользователь переключает их в правом верхнем углу.
        folium.TileLayer(
            tiles=ESRI_GRAY_TILES, attr=ESRI_GRAY_ATTR,
            name="Подложка: светло-серая (Esri)", overlay=False, control=True, show=True,
        ).add_to(fmap)
        folium.TileLayer(
            tiles=OSM_TILES, attr=OSM_ATTR,
            name="Подложка: улицы (OpenStreetMap)", overlay=False, control=True, show=False,
        ).add_to(fmap)
        folium.TileLayer(
            tiles=ESRI_IMAGERY_TILES, attr=ESRI_IMAGERY_ATTR,
            name="Подложка: спутник (Esri)", overlay=False, control=True, show=False,
        ).add_to(fmap)

    all_numeric = list(NUMERIC_LAYERS)
    for column in provision_columns:
        service = column.replace("provision_", "")
        name_ru, _, _ = _service_meta(service)
        title = f"Обеспеченность: {name_ru or service}"
        all_numeric.append((column, title, "YlGn_09"))

    for column, title, palette in all_numeric:
        if column not in gdf.columns:
            continue

        values = pd.to_numeric(gdf[column], errors="coerce")
        finite = values[np.isfinite(values)]
        vmin = float(finite.min()) if len(finite) else 0.0
        vmax = float(finite.max()) if len(finite) else 1.0
        if vmin == vmax:
            vmax = vmin + 1.0

        colormap = getattr(cm.linear, palette).scale(vmin, vmax)
        layer = folium.FeatureGroup(name=title, show=False)
        folium.GeoJson(
            _gdf_to_features(gdf, ["block_id", column]),
            name=title,
            style_function=_numeric_style(column, colormap),
            highlight_function=lambda _: {"weight": 2, "color": "#000000"},
            tooltip=folium.GeoJsonTooltip(
                fields=["block_id", column],
                aliases=["Квартал", title],
                localize=True,
                sticky=False,
            ),
        ).add_to(layer)
        layer.add_to(fmap)

    for column, title in CATEGORICAL_LAYERS:
        if column not in gdf.columns:
            continue
        categories = sorted({str(v) for v in gdf[column].dropna().unique()})
        color_map = {
            category: CATEGORY_COLORS[i % len(CATEGORY_COLORS)]
            for i, category in enumerate(categories)
        }
        layer = folium.FeatureGroup(name=title, show=False)
        folium.GeoJson(
            _gdf_to_features(gdf, ["block_id", column]),
            name=title,
            style_function=_categorical_style(column, color_map),
            highlight_function=lambda _: {"weight": 2, "color": "#000000"},
            tooltip=folium.GeoJsonTooltip(
                fields=["block_id", column],
                aliases=[
                    "Квартал",
                    "Использование территории" if column == "land_use" else "Тип застройки",
                ],
                localize=True,
                sticky=False,
            ),
        ).add_to(layer)
        layer.add_to(fmap)

    # Слой-подсказка со значениями по умолчанию: население, если есть
    folium.LayerControl(collapsed=True, position="topright").add_to(fmap)

    # Единственная общая легенда-градиент (чтобы не загромождать карту)
    legend_html = (
        '<div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999; '
        'background: white; padding: 8px 12px; border-radius: 6px; '
        'box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-size: 12px;">'
        '<b>Цвет квартала</b><br>'
        '<span style="display:inline-block;width:120px;height:10px;'
        'background:linear-gradient(90deg,#ffffcc,#fd8d3c,#800026);'
        'vertical-align:middle;margin-right:6px;"></span>'
        'меньше &rarr; больше<br>'
        '<span style="color:#666;">в пределах выбранного показателя</span>'
        "</div>"
    )
    fmap.get_root().html.add_child(folium.Element(legend_html))
    return fmap


# =================================================================================
# 3. Графики
# =================================================================================
def build_charts(gdf: gpd.GeoDataFrame, provision_columns: list[str], matrix: pd.DataFrame | None):
    charts = []  # список словарей {"title", "image"}

    def add_chart(title: str, fig):
        charts.append({"title": title, "image": _fig_to_base64(fig)})

    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    # 3.1. Плотность застройки (FSI)
    if "fsi" in gdf.columns and pd.to_numeric(gdf["fsi"], errors="coerce").notna().any():
        fig, ax = plt.subplots(figsize=(6, 3.6))
        pd.to_numeric(gdf["fsi"], errors="coerce").dropna().clip(upper=6).plot.hist(
            bins=30, ax=ax, color="#3182bd", edgecolor="white"
        )
        ax.set_title("Распределение плотности застройки (FSI)")
        ax.set_xlabel("FSI")
        add_chart("Плотность застройки (FSI)", fig)

    # 3.2. Транспортная доступность
    if "mean_accessibility" in gdf.columns:
        values = pd.to_numeric(gdf["mean_accessibility"], errors="coerce").dropna()
        if len(values):
            fig, ax = plt.subplots(figsize=(6, 3.6))
            values.plot.hist(bins=30, ax=ax, color="#e6550d", edgecolor="white")
            ax.set_title("Среднее время до остальных кварталов")
            ax.set_xlabel("минуты")
            add_chart("Транспортная доступность", fig)

    # 3.3. Обеспеченность услугами
    if provision_columns:
        means = {}
        for column in provision_columns:
            service = column.replace("provision_", "")
            name_ru, _, _ = _service_meta(service)
            values = pd.to_numeric(gdf[column], errors="coerce")
            means[name_ru or service] = float(values.mean(skipna=True))
        series = pd.Series(means).sort_values()
        fig, ax = plt.subplots(figsize=(6.4, max(3.0, 0.3 * len(series) + 1.2)))
        series.plot.barh(ax=ax, color="#31a354")
        ax.set_xlim(0, 1)
        ax.set_xlabel("средняя обеспеченность (доля закрытого спроса)")
        ax.set_title("Обеспеченность по услугам")
        add_chart("Обеспеченность по услугам", fig)

    # 3.4. Морфотипы (Strelka)
    if "morphotype_strelka" in gdf.columns:
        counts = gdf["morphotype_strelka"].dropna().value_counts()
        if len(counts):
            fig, ax = plt.subplots(figsize=(6, max(3.0, 0.3 * len(counts) + 1.2)))
            counts.sort_values().plot.barh(ax=ax, color="#756bb1")
            ax.set_xlabel("число кварталов")
            ax.set_title("Морфотипы застройки (Strelka)")
            add_chart("Морфотипы застройки", fig)

    # 3.5. Распределение времени в пути (по матрице)
    if matrix is not None and matrix.size:
        values = matrix.to_numpy(dtype="float64").ravel()
        values = values[np.isfinite(values)]
        if values.size:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.hist(values, bins=50, color="#6baed6", edgecolor="white")
            ax.set_title("Время в пути между кварталами")
            ax.set_xlabel("минуты")
            add_chart("Время в пути (матрица)", fig)

    return charts


def build_heatmap(matrix: pd.DataFrame | None):
    if matrix is None or not matrix.size:
        return None
    values = matrix.to_numpy(dtype="float64")
    finite = values[np.isfinite(values)]
    if not finite.size:
        return None
    vmax = float(np.percentile(finite, 95))
    values = np.where(np.isfinite(values), values, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(values, cmap="viridis", vmax=vmax, aspect="auto")
    ax.set_title("Матрица времени в пути (минуты)")
    ax.set_xlabel("квартал")
    ax.set_ylabel("квартал")
    fig.colorbar(image, ax=ax, label="минуты")
    return _fig_to_base64(fig)


# =================================================================================
# 4. Таблицы
# =================================================================================
def build_provision_table(gdf: gpd.GeoDataFrame, provision_columns: list[str]) -> str:
    if not provision_columns:
        return "<p>Нет данных об обеспеченности.</p>"
    rows = []
    for column in provision_columns:
        service = column.replace("provision_", "")
        name_ru, accessibility, demand = _service_meta(service)
        values = pd.to_numeric(gdf[column], errors="coerce")

        weak_column = f"provision_weak_{service}"
        weak_mean = None
        if weak_column in gdf.columns:
            weak_mean = round(
                float(pd.to_numeric(gdf[weak_column], errors="coerce").mean(skipna=True)), 3
            )

        rows.append(
            {
                "Услуга": name_ru or service,
                "Норматив доступности, мин": accessibility,
                "Норматив, на 1000 чел.": demand,
                "Средняя обеспеченность": round(float(values.mean(skipna=True)), 3),
                "Средняя (слабая)": weak_mean,
                "Доля кварталов с обеспеченностью ≥ 0.8": round(
                    float((values >= 0.8).mean(skipna=True)), 3
                ),
            }
        )
    table = pd.DataFrame(rows).sort_values("Средняя обеспеченность")
    return table.fillna("—").to_html(index=False, classes="data-table", border=0, justify="left")


def build_worst_blocks_table(gdf: gpd.GeoDataFrame, provision_columns: list[str]) -> str:
    if not provision_columns:
        return "<p>Нет данных.</p>"
    work = gdf.copy()
    work["Средняя обеспеченность"] = work[provision_columns].apply(
        pd.to_numeric, errors="coerce"
    ).mean(axis=1)

    columns = {
        "block_id": "Квартал",
        "population": "Население, чел.",
        "fsi": "FSI",
        "mean_accessibility": "Среднее время, мин",
        "Средняя обеспеченность": "Средняя обеспеченность",
        "morphotype_strelka": "Морфотип (Strelka)",
    }
    present = [c for c in columns if c in work.columns]
    table = (
        work[present]
        .sort_values("Средняя обеспеченность", ascending=True)
        .head(10)
        .rename(columns=columns)
        .round(2)
    )
    return table.fillna("—").to_html(index=False, classes="data-table", border=0, justify="left")


# =================================================================================
# 5. Ключевые числа и выводы
# =================================================================================
def build_kpis(gdf: gpd.GeoDataFrame, provision_columns: list[str]) -> list[dict]:
    kpis = [{"label": "Кварталов в районе", "value": f"{len(gdf):,}".replace(",", " ")}]

    try:
        projected = gdf.to_crs(gdf.estimate_utm_crs())
        area_km2 = float(projected.geometry.area.sum()) / 1e6
        kpis.append({"label": "Площадь кварталов, км²", "value": f"{area_km2:,.1f}".replace(",", " ")})
    except Exception:  # noqa: BLE001
        pass

    for column, label, fmt in [
        ("population", "Население, чел.", "{:,.0f}"),
        ("fsi", "Средняя плотность (FSI)", "{:.2f}"),
        ("l", "Средняя этажность", "{:.1f}"),
        ("mean_accessibility", "Среднее время, мин", "{:.1f}"),
    ]:
        if column in gdf.columns:
            value = float(pd.to_numeric(gdf[column], errors="coerce").mean(skipna=True))
            if not math.isnan(value):
                kpis.append({"label": label, "value": fmt.format(value).replace(",", " ")})

    if provision_columns:
        value = float(
            gdf[provision_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1).mean(skipna=True)
        )
        kpis.append({"label": "Средняя обеспеченность", "value": f"{value:.2f}"})

    return kpis


def build_conclusions(gdf: gpd.GeoDataFrame, provision_columns: list[str]) -> list[str]:
    conclusions = []

    if provision_columns:
        means = {}
        for column in provision_columns:
            service = column.replace("provision_", "")
            name_ru, _, _ = _service_meta(service)
            values = pd.to_numeric(gdf[column], errors="coerce")
            means[name_ru or service] = float(values.mean(skipna=True))
        series = pd.Series(means).sort_values()
        overall = float(series.mean())
        worst = series.head(3)

        conclusions.append(
            f"Средняя обеспеченность по набору услуг — <b>{overall:.2f}</b> "
            "(доля спроса, закрытого в пределах норматива доступности)."
        )
        conclusions.append(
            "Хуже всего обеспечены: "
            + ", ".join(f"<b>{name}</b> ({value:.2f})" for name, value in worst.items())
            + "."
        )
        low_share = float(
            (gdf[provision_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1) < 0.5).mean()
        )
        conclusions.append(
            f"В <b>{low_share * 100:.0f}%</b> кварталов средняя обеспеченность ниже 0.5 — "
            "там стоит усилить сервисную сеть."
        )

    if "mean_accessibility" in gdf.columns:
        value = float(pd.to_numeric(gdf["mean_accessibility"], errors="coerce").mean(skipna=True))
        conclusions.append(
            f"Среднее время поездки от квартала до остальных кварталов района — <b>{value:.1f} мин</b> "
            "(личный автотранспорт)."
        )

    if "fsi" in gdf.columns:
        value = float(pd.to_numeric(gdf["fsi"], errors="coerce").mean(skipna=True))
        conclusions.append(f"Средняя плотность застройки (FSI) — <b>{value:.2f}</b>.")

    if "morphotype_strelka" in gdf.columns:
        counts = gdf["morphotype_strelka"].dropna().value_counts()
        if len(counts):
            conclusions.append(
                f"Самый частый тип застройки (Strelka) — <b>{counts.index[0]}</b> "
                f"({int(counts.iloc[0])} кварталов)."
            )

    return conclusions


# =================================================================================
# 6. Сборка HTML-страницы
# =================================================================================
CSS = """
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f4f6f8; color: #222; }
header { background: #1b3a5c; color: #fff; padding: 22px 28px; }
header h1 { margin: 0 0 6px 0; font-size: 22px; }
header p { margin: 0; opacity: 0.85; font-size: 13px; }
.container { max-width: 1200px; margin: 0 auto; padding: 20px 28px 60px 28px; }
.section-title { font-size: 17px; margin: 28px 0 12px 0; color: #1b3a5c; border-left: 4px solid #3182bd; padding-left: 10px; }
@media (min-width: 1240px) { .container { max-width: 1200px; } }
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }
.kpi { background: #fff; border-radius: 8px; padding: 14px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.kpi .value { font-size: 24px; font-weight: 600; color: #1b3a5c; }
.kpi .label { font-size: 12px; color: #667; margin-top: 4px; }
.map-frame { width: 100%; height: 720px; border: 1px solid #ccd; border-radius: 8px; background: #fff; }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }
.chart { background: #fff; border-radius: 8px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.chart img { width: 100%; height: auto; display: block; }
table.data-table { border-collapse: collapse; width: 100%; background: #fff; font-size: 13px; }
table.data-table th, table.data-table td { border: 1px solid #e2e6ea; padding: 7px 10px; text-align: left; }
table.data-table th { background: #eef3f7; }
table.data-table tr:nth-child(even) td { background: #fafcff; }
.conclusions { background: #fff; border-radius: 8px; padding: 16px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.conclusions li { margin: 6px 0; }
.legend-chips { background: #fff; border-radius: 8px; padding: 12px 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-size: 13px; }
.chip { display: inline-block; margin: 3px 10px 3px 0; }
.chip .dot { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 5px; vertical-align: middle; }
footer { text-align: center; color: #889; font-size: 12px; padding: 20px; }
"""


def render_page(
    gdf: gpd.GeoDataFrame,
    provision_columns: list[str],
    map_html: str,
    charts: list[dict],
    heatmap_b64: str | None,
    provision_table: str,
    worst_table: str,
    kpis: list[dict],
    conclusions: list[str],
) -> str:
    kpi_html = "".join(
        f'<div class="kpi"><div class="value">{kpi["value"]}</div>'
        f'<div class="label">{kpi["label"]}</div></div>'
        for kpi in kpis
    )

    chart_html = "".join(
        f'<div class="chart"><img src="data:image/png;base64,{chart["image"]}" '
        f'alt="{html_lib.escape(chart["title"])}"></div>'
        for chart in charts
    )
    if heatmap_b64:
        chart_html += f'<div class="chart"><img src="data:image/png;base64,{heatmap_b64}" alt="Матрица доступности"></div>'

    conclusion_html = "".join(f"<li>{item}</li>" for item in conclusions)

    # Легенда категориальных слоёв
    chips_html = ""
    for column, title in CATEGORICAL_LAYERS:
        if column not in gdf.columns:
            continue
        categories = sorted({str(v) for v in gdf[column].dropna().unique()})
        chips = "".join(
            f'<span class="chip"><span class="dot" style="background:{CATEGORY_COLORS[i % len(CATEGORY_COLORS)]}">'
            f"</span>{html_lib.escape(category)}</span>"
            for i, category in enumerate(categories)
        )
        chips_html += f"<div style='margin-bottom:6px;'><b>{html_lib.escape(title)}</b><br>{chips}</div>"

    map_iframe = (
        f'<iframe class="map-frame" srcdoc="{html_lib.escape(map_html, quote=True)}"></iframe>'
    )
    generated = datetime.now().strftime("%d.%m.%Y %H:%M")

    html = "".join(
        [
            "<!DOCTYPE html><html lang='ru'><head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            "<title>Фрунзенский район — дашборд</title>",
            f"<style>{CSS}</style>",
            "</head><body>",
            "<header>",
            "<h1>Фрунзенский район Санкт-Петербурга — дашборд</h1>",
            "<p>Показатели BlocksNet: доступность, плотность, морфотипы, обеспеченность услугами</p>",
            "</header>",
            "<div class='container'>",
            f"<div class='section-title'>Ключевые показатели</div><div class='kpis'>{kpi_html}</div>",
            "<div class='section-title'>Карта кварталов</div>",
            "<p style='font-size:13px;color:#667;margin:0 0 8px 0;'>"
            "Справа сверху — переключатель показателей. Наведите курсор на квартал, чтобы увидеть значение.</p>",
            map_iframe,
            "<div class='section-title'>Графики</div>",
            f"<div class='charts'>{chart_html}</div>",
            "<div class='section-title'>Легенда типов застройки</div>",
            f"<div class='legend-chips'>{chips_html}</div>",
            "<div class='section-title'>Обеспеченность услугами</div>",
            provision_table,
            "<div class='section-title'>Кварталы с наименьшей обеспеченностью (10)</div>",
            worst_table,
            "<div class='section-title'>Краткие выводы</div>",
            f"<div class='conclusions'><ul>{conclusion_html}</ul></div>",
            "</div>",
            f"<footer>Страница сгенерирована {generated}. Источник данных: "
            "BlocksNet + открытые данные OpenStreetMap (проект IDUclub/blocksnet-data).</footer>",
            "</body></html>",
        ]
    )
    return html


# =================================================================================
# Основной сценарий
# =================================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Собирает HTML-дашборд по результатам ноутбука.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="папка с результатами")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="куда сохранить HTML")
    parser.add_argument(
        "--no-tiles",
        action="store_true",
        help="не подключать карты-подложки (полностью офлайн: белый фон + контуры кварталов)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        gdf = load_indicators(input_dir)
    except FileNotFoundError as error:
        print("\nОШИБКА:")
        print(error)
        raise SystemExit(1) from None

    provision_columns = collect_provision_columns(gdf)
    matrix = _read_matrix(input_dir)

    _log("строим карту...")
    fmap = build_map(gdf, provision_columns, tiles=not args.no_tiles)
    map_html = fmap.get_root().render()

    _log("строим графики...")
    charts = build_charts(gdf, provision_columns, matrix)
    heatmap_b64 = build_heatmap(matrix)

    _log("строим таблицы и выводы...")
    provision_table = build_provision_table(gdf, provision_columns)
    worst_table = build_worst_blocks_table(gdf, provision_columns)
    kpis = build_kpis(gdf, provision_columns)
    conclusions = build_conclusions(gdf, provision_columns)

    page = render_page(
        gdf,
        provision_columns,
        map_html,
        charts,
        heatmap_b64,
        provision_table,
        worst_table,
        kpis,
        conclusions,
    )
    output_path.write_text(page, encoding="utf-8")
    size_mb = output_path.stat().st_size / 1e6
    _log(f"готово: {output_path} ({size_mb:.1f} МБ)")
    print("\nОткройте файл в браузере (двойной клик) или выполните:")
    print(f'  start "" "{output_path}"')


if __name__ == "__main__":
    main()
