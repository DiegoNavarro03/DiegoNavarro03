from __future__ import annotations

import re
import tempfile
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    import geopandas as gpd
    from shapely.geometry import Point
except ImportError as exc:
    raise SystemExit(
        "Falta geopandas o una dependencia geoespacial. Instale con:\n"
        "  py -m pip install geopandas shapely pyproj fiona"
    ) from exc


BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_LIMPIOS_DIR = BASE_DIR / "datos_limpios"
FIGURAS_DIR = BASE_DIR / "figuras"
CUENCA_DIR = BASE_DIR / "cuenca"
HYDROBASINS_RAW_DIR = CUENCA_DIR / "hydrobasins_raw"
DATOS_PREC_DIR = BASE_DIR / "DATOS_PREC"
ESTACION_DIR = next(DATOS_PREC_DIR.glob("ESTACI*"), DATOS_PREC_DIR / "ESTACION")
SATELITE_DIR = next(DATOS_PREC_DIR.glob("SAT*LITE"), DATOS_PREC_DIR / "SATELITE")
IMERG_DIR = SATELITE_DIR / "IMERG"
CMORPH_DIR = SATELITE_DIR / "CMORPH"

CSV_DIARIO = DATOS_LIMPIOS_DIR / "precipitacion_diaria_estacion_imerg_cmorph.csv"
CSV_CMORPH = CMORPH_DIR / "cmorph_champaign_1998_2025.csv"
CSV_MAXIMOS_ESTACION = DATOS_LIMPIOS_DIR / "idf_maximos_anuales_estacion.csv"
CSV_COMPARACION_IDF = DATOS_LIMPIOS_DIR / "idf_comparacion_30min_o_mas.csv"

SALIDA_GPKG = CUENCA_DIR / "cuenca_estacion_champaign.gpkg"
SALIDA_ATRIBUTOS = CUENCA_DIR / "atributos_cuenca_champaign.csv"
SALIDA_MAPA = CUENCA_DIR / "mapa_cuenca_estacion_champaign.png"
SALIDA_CAUDAL = CUENCA_DIR / "caudal_maximo_teorico_champaign.csv"
FIG19_PNG = FIGURAS_DIR / "fig19_caudal_maximo_teorico_cuenca.png"
FIG19_HTML = FIGURAS_DIR / "fig19_caudal_maximo_teorico_cuenca.html"

LAT_ESTACION = 40.05
LON_ESTACION = -88.37
CRS_GEOGRAFICO = "EPSG:4326"
CRS_AREA_NA = "EPSG:5070"
INICIO = pd.Timestamp("2015-01-01")
FIN = pd.Timestamp("2024-12-31")
FIN_EXCLUSIVO = FIN + pd.Timedelta(days=1)
ANIOS = range(INICIO.year, FIN.year + 1)
SEGUNDOS_ANIO = 365.25 * 24 * 3600
DURACIONES_OBJETIVO = [30, 60, 360, 1440]

COLUMNAS_CRNS = [
    "wban",
    "utc_date",
    "utc_time",
    "lst_date",
    "lst_time",
    "crx_vn",
    "longitude",
    "latitude",
    "t_calc_c",
    "p_calc_mm",
    "solarad_w_m2",
    "solarad_flag",
    "solarad_max_w_m2",
    "solarad_max_flag",
    "solarad_max_time",
    "precip_type",
    "precip_type_flag",
    "rh_pct",
    "rh_flag",
    "soil_moisture_5_cm",
    "soil_temp_5_cm",
    "wetness",
    "wetness_flag",
]
VALORES_FALTANTES = [-99, -99.0, -9999, -9999.0, -9999.9]
ETIQUETAS = {
    "estacion": "Estacion CRNS",
    "imerg": "IMERG",
    "cmorph": "CMORPH",
}
COLORES_FUENTE = {
    "estacion": "#1f77b4",
    "imerg": "#d95f02",
    "cmorph": "#2ca25f",
}


def configurar_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def listar_archivos_hydrobasins() -> list[Path]:
    if not HYDROBASINS_RAW_DIR.exists():
        return []
    return sorted(ruta for ruta in HYDROBASINS_RAW_DIR.rglob("*") if ruta.is_file())


def stem_desde_shp(nombre: str) -> str:
    return nombre[:-4] if nombre.lower().endswith(".shp") else Path(nombre).stem


def extraer_shapefile_de_zip(zip_path: Path, shp_interno: str) -> tuple[tempfile.TemporaryDirectory, Path]:
    temp_dir = tempfile.TemporaryDirectory(prefix="hydrobasins_")
    destino = Path(temp_dir.name)
    stem = stem_desde_shp(shp_interno)
    with ZipFile(zip_path) as zip_file:
        miembros = [
            nombre
            for nombre in zip_file.namelist()
            if not nombre.endswith("/") and stem_desde_shp(nombre) == stem
        ]
        for nombre in miembros:
            zip_file.extract(nombre, destino)

    shp_extraido = next(destino.rglob(Path(shp_interno).name), None)
    if shp_extraido is None:
        temp_dir.cleanup()
        raise FileNotFoundError(f"No se pudo extraer {shp_interno} desde {zip_path}")
    return temp_dir, shp_extraido


def encontrar_shapefile_hydrobasins() -> tuple[Path, str, int, tempfile.TemporaryDirectory | None]:
    archivos = listar_archivos_hydrobasins()
    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron archivos en {HYDROBASINS_RAW_DIR}."
        )

    for nivel in [7, 8]:
        etiqueta = f"lev{nivel:02d}"
        shps = [
            ruta
            for ruta in archivos
            if ruta.suffix.lower() == ".shp" and etiqueta in ruta.name.lower()
        ]
        if shps:
            return shps[0], str(shps[0]), nivel, None

        zips = [ruta for ruta in archivos if ruta.suffix.lower() == ".zip"]
        for zip_path in zips:
            if etiqueta in zip_path.name.lower():
                with ZipFile(zip_path) as zip_file:
                    candidatos = [
                        nombre
                        for nombre in zip_file.namelist()
                        if nombre.lower().endswith(".shp")
                        and etiqueta in Path(nombre).name.lower()
                    ]
                if candidatos:
                    temp_dir, shp_path = extraer_shapefile_de_zip(zip_path, candidatos[0])
                    return shp_path, f"{zip_path}!{candidatos[0]}", nivel, temp_dir

            with ZipFile(zip_path) as zip_file:
                candidatos = [
                    nombre
                    for nombre in zip_file.namelist()
                    if nombre.lower().endswith(".shp")
                    and etiqueta in Path(nombre).name.lower()
                ]
            if candidatos:
                temp_dir, shp_path = extraer_shapefile_de_zip(zip_path, candidatos[0])
                return shp_path, f"{zip_path}!{candidatos[0]}", nivel, temp_dir

    encontrados = "\n".join(f"  {ruta}" for ruta in archivos)
    raise FileNotFoundError(
        "No se encontro HydroBASINS nivel 7 ni nivel 8.\n"
        "Se esperaba un shapefile o zip con nombres similares a "
        "hybas_na_lev07_v1c.shp, hybas_na_lev07_v1c.zip, "
        "hybas_na_lev08_v1c.shp o hybas_na_lev08_v1c.zip.\n"
        f"Archivos encontrados en {HYDROBASINS_RAW_DIR}:\n{encontrados}"
    )


def leer_hydrobasins() -> tuple[gpd.GeoDataFrame, str, int, tempfile.TemporaryDirectory | None]:
    shp_path, archivo_usado, nivel, temp_dir = encontrar_shapefile_hydrobasins()
    cuencas = gpd.read_file(shp_path)
    if cuencas.empty:
        raise ValueError(f"El shapefile HydroBASINS esta vacio: {archivo_usado}")
    if cuencas.crs is None:
        cuencas = cuencas.set_crs(CRS_GEOGRAFICO)
    return cuencas, archivo_usado, nivel, temp_dir


def crear_punto_estacion() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"latitud_estacion": [LAT_ESTACION], "longitud_estacion": [LON_ESTACION]},
        geometry=[Point(LON_ESTACION, LAT_ESTACION)],
        crs=CRS_GEOGRAFICO,
    )


def seleccionar_cuenca(
    cuencas: gpd.GeoDataFrame, punto: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, str]:
    punto_mismo_crs = punto.to_crs(cuencas.crs)
    geom_punto = punto_mismo_crs.geometry.iloc[0]

    seleccion = cuencas[cuencas.geometry.contains(geom_punto)].copy()
    metodo = "contains"
    if seleccion.empty:
        seleccion = cuencas[cuencas.geometry.intersects(geom_punto)].copy()
        metodo = "intersects"
    if seleccion.empty:
        cuencas_m = cuencas.to_crs(CRS_AREA_NA)
        punto_m = punto.to_crs(CRS_AREA_NA).geometry.iloc[0]
        indice = cuencas_m.distance(punto_m).idxmin()
        seleccion = cuencas.loc[[indice]].copy()
        metodo = "nearest"

    if len(seleccion) > 1:
        seleccion = seleccion.iloc[[0]].copy()
    seleccion["metodo_seleccion"] = metodo
    return seleccion, metodo


def enriquecer_atributos_cuenca(
    cuenca: gpd.GeoDataFrame, nivel: int
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    cuenca_m = cuenca.to_crs(CRS_AREA_NA)
    area_km2 = float(cuenca_m.geometry.area.iloc[0] / 1_000_000)
    perimetro_km = float(cuenca_m.geometry.length.iloc[0] / 1_000)

    cuenca_salida = cuenca.to_crs(CRS_GEOGRAFICO).copy()
    cuenca_salida["nivel_hydrobasins"] = nivel
    cuenca_salida["area_calculada_km2"] = area_km2
    cuenca_salida["perimetro_calculado_km"] = perimetro_km
    cuenca_salida["latitud_estacion"] = LAT_ESTACION
    cuenca_salida["longitud_estacion"] = LON_ESTACION

    columnas_base = [
        "HYBAS_ID",
        "NEXT_DOWN",
        "NEXT_SINK",
        "MAIN_BAS",
        "DIST_SINK",
        "DIST_MAIN",
        "SUB_AREA",
        "UP_AREA",
        "PFAF_ID",
        "nivel_hydrobasins",
        "area_calculada_km2",
        "perimetro_calculado_km",
        "latitud_estacion",
        "longitud_estacion",
        "metodo_seleccion",
    ]
    columnas = [col for col in columnas_base if col in cuenca_salida.columns]
    atributos = cuenca_salida[columnas].drop(columns="geometry", errors="ignore").copy()
    return cuenca_salida, atributos


def guardar_mapa_cuenca(cuenca: gpd.GeoDataFrame, punto: gpd.GeoDataFrame) -> None:
    cuenca_geo = cuenca.to_crs(CRS_GEOGRAFICO)
    punto_geo = punto.to_crs(CRS_GEOGRAFICO)

    fig, ax = plt.subplots(figsize=(8, 7))
    cuenca_geo.plot(
        ax=ax,
        facecolor="#d9f0d3",
        edgecolor="#238b45",
        linewidth=1.8,
        label="Cuenca HydroBASINS",
    )
    punto_geo.plot(
        ax=ax,
        color="#d7301f",
        markersize=55,
        marker="*",
        label="Estacion CRNS",
    )
    ax.set_title("Cuenca HydroBASINS asociada a la estacion CRNS Champaign")
    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.grid(True, alpha=0.25)
    ax.annotate(
        "N",
        xy=(0.94, 0.88),
        xytext=(0.94, 0.76),
        xycoords="axes fraction",
        ha="center",
        va="center",
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color="black"),
        fontsize=12,
        fontweight="bold",
    )
    leyenda = [
        Patch(facecolor="#d9f0d3", edgecolor="#238b45", label="Cuenca HydroBASINS"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="w",
            markerfacecolor="#d7301f",
            markeredgecolor="#d7301f",
            markersize=11,
            label="Estacion CRNS",
        ),
    ]
    ax.legend(handles=leyenda, loc="best")
    fig.tight_layout()
    fig.savefig(SALIDA_MAPA, bbox_inches="tight")
    plt.close(fig)


def detectar_columna_fecha(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.lower() in {"fecha", "date", "time", "datetime", "fecha_hora"}:
            return col
    raise ValueError("No se encontro una columna de fecha en el CSV diario.")


def detectar_columna_estacion(df: pd.DataFrame) -> str:
    candidatas = []
    for col in df.columns:
        col_l = col.lower()
        if col_l in {"fecha", "date", "time", "datetime", "fecha_hora"}:
            continue
        if any(txt in col_l for txt in ["estacion", "crns", "station"]):
            candidatas.append(col)
    if candidatas:
        return candidatas[0]

    numericas = [
        col for col in df.columns if pd.api.types.is_numeric_dtype(pd.to_numeric(df[col], errors="coerce"))
    ]
    if numericas:
        return numericas[0]
    raise ValueError("No se pudo identificar la columna de precipitacion de estacion.")


def precipitacion_media_anual_estacion() -> float:
    diario = pd.read_csv(CSV_DIARIO)
    columna_fecha = detectar_columna_fecha(diario)
    columna_estacion = detectar_columna_estacion(diario)
    diario[columna_fecha] = pd.to_datetime(diario[columna_fecha], errors="coerce")
    diario[columna_estacion] = pd.to_numeric(diario[columna_estacion], errors="coerce")
    diario = diario[
        (diario[columna_fecha] >= INICIO) & (diario[columna_fecha] < FIN_EXCLUSIVO)
    ].copy()
    anual = diario.groupby(diario[columna_fecha].dt.year)[columna_estacion].sum(min_count=1)
    anual = anual.reindex(list(ANIOS)).dropna()
    if anual.empty:
        raise ValueError("No hay precipitacion anual valida para la estacion.")
    return float(anual.mean())


def extraer_anio(ruta: Path) -> int | None:
    coincidencia = re.search(r"(19|20)\d{2}", ruta.name)
    return int(coincidencia.group(0)) if coincidencia else None


def seleccionar_archivos_por_anio(carpeta: Path, patron: str) -> list[Path]:
    rutas = []
    for ruta in sorted(carpeta.glob(patron)):
        anio = extraer_anio(ruta)
        if anio in ANIOS:
            rutas.append(ruta)

    anios_encontrados = {extraer_anio(ruta) for ruta in rutas}
    anios_faltantes = [anio for anio in ANIOS if anio not in anios_encontrados]
    if anios_faltantes:
        raise FileNotFoundError(
            f"Faltan archivos en {carpeta} para estos anios: {anios_faltantes}"
        )
    return rutas


def construir_fecha_hora(df: pd.DataFrame, col_fecha: str, col_hora: str) -> pd.Series:
    fecha = df[col_fecha].astype("Int64").astype(str).str.zfill(8)
    hora = df[col_hora].astype("Int64").astype(str).str.zfill(4)
    return pd.to_datetime(fecha + hora, format="%Y%m%d%H%M", errors="coerce")


def filtrar_periodo(df: pd.DataFrame, columna_fecha: str) -> pd.DataFrame:
    return df[
        (df[columna_fecha] >= INICIO) & (df[columna_fecha] < FIN_EXCLUSIVO)
    ].copy()


def leer_estacion_5min() -> pd.DataFrame:
    rutas = seleccionar_archivos_por_anio(ESTACION_DIR, "CRNS*.txt")
    dataframes = []
    for ruta in rutas:
        df = pd.read_csv(
            ruta,
            sep=r"\s+",
            header=None,
            names=COLUMNAS_CRNS,
            na_values=VALORES_FALTANTES,
        )
        dataframes.append(df)

    estacion = pd.concat(dataframes, ignore_index=True)
    estacion["fecha_hora_utc"] = construir_fecha_hora(estacion, "utc_date", "utc_time")
    estacion["precip_mm"] = pd.to_numeric(estacion["p_calc_mm"], errors="coerce")
    estacion.loc[estacion["precip_mm"] < 0, "precip_mm"] = np.nan
    estacion = filtrar_periodo(estacion, "fecha_hora_utc")
    return estacion[["fecha_hora_utc", "precip_mm"]].sort_values("fecha_hora_utc")


def leer_imerg_30min() -> pd.DataFrame:
    rutas = seleccionar_archivos_por_anio(IMERG_DIR, "imerg_champaign_*.csv")
    dataframes = []
    for ruta in rutas:
        df = pd.read_csv(
            ruta,
            skiprows=8,
            skipinitialspace=True,
            na_values=VALORES_FALTANTES,
        )
        df.columns = [col.strip() for col in df.columns]
        columna_precip = next(col for col in df.columns if col != "time")
        df = df.rename(columns={columna_precip: "precip_mm_per_hr"})
        dataframes.append(df)

    imerg = pd.concat(dataframes, ignore_index=True)
    imerg["fecha_hora_utc"] = pd.to_datetime(imerg["time"], errors="coerce")
    imerg["precip_mm_per_hr"] = pd.to_numeric(
        imerg["precip_mm_per_hr"], errors="coerce"
    )
    imerg.loc[imerg["precip_mm_per_hr"] < 0, "precip_mm_per_hr"] = np.nan
    imerg["precip_mm"] = imerg["precip_mm_per_hr"] * 0.5
    imerg = filtrar_periodo(imerg, "fecha_hora_utc")
    return imerg[["fecha_hora_utc", "precip_mm"]].sort_values("fecha_hora_utc")


def leer_cmorph_30min() -> pd.DataFrame:
    if not CSV_CMORPH.exists():
        raise FileNotFoundError(f"No se encontro el archivo CMORPH: {CSV_CMORPH}")

    cmorph = pd.read_csv(CSV_CMORPH, comment="#", na_values=VALORES_FALTANTES)
    cmorph.columns = [col.strip() for col in cmorph.columns]
    columnas_requeridas = {"time", "precip_mm_per_hr"}
    faltantes = columnas_requeridas.difference(cmorph.columns)
    if faltantes:
        raise ValueError(f"CMORPH no tiene estas columnas requeridas: {sorted(faltantes)}")

    cmorph["fecha_hora_utc"] = pd.to_datetime(cmorph["time"], errors="coerce")
    cmorph["precip_mm_per_hr"] = pd.to_numeric(
        cmorph["precip_mm_per_hr"], errors="coerce"
    )
    cmorph.loc[cmorph["precip_mm_per_hr"] < 0, "precip_mm_per_hr"] = np.nan
    cmorph["precip_mm"] = cmorph["precip_mm_per_hr"] * 0.5
    cmorph = filtrar_periodo(cmorph, "fecha_hora_utc")
    return cmorph[["fecha_hora_utc", "precip_mm"]].sort_values("fecha_hora_utc")


def serie_regular(df: pd.DataFrame, frecuencia: str) -> pd.Series:
    indice = pd.date_range(INICIO, FIN_EXCLUSIVO, freq=frecuencia, inclusive="left")
    serie = (
        df.set_index("fecha_hora_utc")["precip_mm"]
        .sort_index()
        .resample(frecuencia)
        .sum(min_count=1)
        .reindex(indice)
    )
    return serie


def calcular_maximos_observados(
    fuente: str,
    datos: pd.DataFrame,
    duraciones_min: list[int],
    frecuencia_min: int,
) -> pd.DataFrame:
    serie = serie_regular(datos, f"{frecuencia_min}min")
    filas = []
    for duracion in duraciones_min:
        ventana = duracion // frecuencia_min
        if duracion % frecuencia_min != 0:
            continue
        acumulado = serie.rolling(window=ventana, min_periods=ventana).sum()
        maximo = acumulado.max()
        if pd.isna(maximo):
            continue
        filas.append(
            {
                "fuente": fuente,
                "duracion_min": duracion,
                "max_acumulado_mm": float(maximo),
                "intensidad_mm_h": float(maximo / (duracion / 60)),
                "origen_intensidad": "serie_subhoraria",
            }
        )
    return pd.DataFrame(filas)


def leer_maximos_desde_tabla(fuente: str) -> pd.DataFrame:
    candidatos = []
    for ruta in [CSV_MAXIMOS_ESTACION, CSV_COMPARACION_IDF]:
        if ruta.exists():
            df = pd.read_csv(ruta)
            if {"fuente", "duracion_min", "intensidad_mm_h"}.issubset(df.columns):
                df = df[df["fuente"].str.lower() == fuente].copy()
                candidatos.append(df)

    if not candidatos:
        raise FileNotFoundError(
            "No se pudo leer la serie subhoraria ni una tabla IDF de respaldo "
            f"para {fuente}. Archivos esperados: {CSV_MAXIMOS_ESTACION} o "
            f"{CSV_COMPARACION_IDF}."
        )

    tabla = pd.concat(candidatos, ignore_index=True)
    tabla = tabla[tabla["duracion_min"].isin(DURACIONES_OBJETIVO)].copy()
    if tabla.empty:
        raise ValueError(f"La tabla IDF de respaldo no contiene duraciones para {fuente}.")

    filas = []
    for duracion, grupo in tabla.groupby("duracion_min"):
        intensidad = pd.to_numeric(grupo["intensidad_mm_h"], errors="coerce").max()
        if pd.isna(intensidad):
            continue
        filas.append(
            {
                "fuente": fuente,
                "duracion_min": int(duracion),
                "max_acumulado_mm": float(intensidad * (duracion / 60)),
                "intensidad_mm_h": float(intensidad),
                "origen_intensidad": "tabla_idf_respaldo",
            }
        )
    return pd.DataFrame(filas)


def maximos_observados_por_fuente() -> tuple[pd.DataFrame, list[str]]:
    advertencias = []
    resultados = []
    lectores = {
        "estacion": (leer_estacion_5min, 5),
        "imerg": (leer_imerg_30min, 30),
        "cmorph": (leer_cmorph_30min, 30),
    }

    for fuente, (lector, frecuencia_min) in lectores.items():
        try:
            datos = lector()
            maximos = calcular_maximos_observados(
                fuente, datos, DURACIONES_OBJETIVO, frecuencia_min
            )
            if maximos.empty:
                raise ValueError("no se obtuvieron maximos observados")
            resultados.append(maximos)
        except Exception as exc:
            advertencias.append(
                f"No se pudo leer la serie subhoraria de {fuente}: {exc}. "
                "Se intenta usar tabla IDF de respaldo."
            )
            resultados.append(leer_maximos_desde_tabla(fuente))

    if not resultados:
        raise RuntimeError("No se pudieron obtener intensidades maximas observadas.")
    return pd.concat(resultados, ignore_index=True), advertencias


def construir_tabla_caudales(
    area_km2: float,
    p_media_anual_mm: float,
    maximos: pd.DataFrame,
) -> pd.DataFrame:
    area_m2 = area_km2 * 1_000_000
    q_medio = (p_media_anual_mm / 1000) * area_m2 / SEGUNDOS_ANIO
    filas = [
        {
            "fuente": "estacion",
            "tipo_caudal": "medio_anual_teorico",
            "duracion_min": np.nan,
            "intensidad_mm_h": np.nan,
            "precipitacion_o_intensidad": p_media_anual_mm,
            "unidad_precipitacion": "mm/anio",
            "area_km2": area_km2,
            "caudal_m3_s": q_medio,
            "comentario": "Limite superior anual con ET = 0 y deltaS = 0.",
        }
    ]

    for _, fila in maximos.iterrows():
        caudal = 0.27778 * float(fila["intensidad_mm_h"]) * area_km2
        filas.append(
            {
                "fuente": fila["fuente"],
                "tipo_caudal": "pico_teorico",
                "duracion_min": int(fila["duracion_min"]),
                "intensidad_mm_h": float(fila["intensidad_mm_h"]),
                "precipitacion_o_intensidad": float(fila["intensidad_mm_h"]),
                "unidad_precipitacion": "mm/h",
                "area_km2": area_km2,
                "caudal_m3_s": caudal,
                "comentario": (
                    "Limite superior absoluto: toda la lluvia se convierte "
                    f"instantaneamente en escorrentia ({fila['origen_intensidad']})."
                ),
            }
        )
    return pd.DataFrame(filas)


def guardar_fig19(tabla_caudal: pd.DataFrame) -> None:
    datos = tabla_caudal[
        (tabla_caudal["tipo_caudal"] == "pico_teorico")
        & (tabla_caudal["duracion_min"].isin(DURACIONES_OBJETIVO))
    ].copy()
    datos["duracion_min"] = datos["duracion_min"].astype(int)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for fuente in ["estacion", "imerg", "cmorph"]:
        curva = datos[datos["fuente"] == fuente].sort_values("duracion_min")
        if curva.empty:
            continue
        ax.plot(
            curva["duracion_min"],
            curva["caudal_m3_s"],
            marker="o",
            lw=1.9,
            color=COLORES_FUENTE[fuente],
            label=ETIQUETAS[fuente],
        )
    ax.set_title("Caudal maximo teorico de la cuenca asociada a la estacion CRNS")
    ax.set_xlabel("Duracion de lluvia (min)")
    ax.set_ylabel("Caudal maximo teorico (m3/s)")
    ax.set_xscale("log")
    ax.set_xticks(DURACIONES_OBJETIVO)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIG19_PNG, bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    for fuente in ["estacion", "imerg", "cmorph"]:
        curva = datos[datos["fuente"] == fuente].sort_values("duracion_min")
        if curva.empty:
            continue
        fig_html.add_trace(
            go.Scatter(
                x=curva["duracion_min"],
                y=curva["caudal_m3_s"],
                mode="lines+markers",
                name=ETIQUETAS[fuente],
                line=dict(color=COLORES_FUENTE[fuente]),
            )
        )
    fig_html.update_layout(
        title="Caudal maximo teorico de la cuenca asociada a la estacion CRNS",
        template="plotly_white",
        font=dict(size=13),
        margin=dict(l=75, r=35, t=70, b=65),
    )
    fig_html.update_xaxes(
        title_text="Duracion de lluvia (min)",
        type="log",
        tickvals=DURACIONES_OBJETIVO,
        ticktext=[str(d) for d in DURACIONES_OBJETIVO],
    )
    fig_html.update_yaxes(title_text="Caudal maximo teorico (m3/s)")
    fig_html.write_html(FIG19_HTML, include_plotlyjs="cdn")


def imprimir_resumen(
    archivo_hydro: str,
    nivel: int,
    atributos: pd.DataFrame,
    p_media_anual_mm: float,
    tabla_caudal: pd.DataFrame,
    advertencias: list[str],
    archivos: list[Path],
) -> None:
    fila = atributos.iloc[0]
    hybas_id = fila.get("HYBAS_ID", "no disponible")
    area_km2 = float(fila["area_calculada_km2"])
    perimetro_km = float(fila["perimetro_calculado_km"])

    print("\nResumen de cuenca HydroBASINS y caudal teorico")
    print(f"- Archivo HydroBASINS usado: {archivo_hydro}")
    print(f"- Nivel HydroBASINS usado: {nivel}")
    print(f"- HYBAS_ID de la cuenca: {hybas_id}")
    print(f"- Area de la cuenca: {area_km2:.3f} km2")
    print(f"- Perimetro de la cuenca: {perimetro_km:.3f} km")
    print(f"- Coordenadas de la estacion: lat {LAT_ESTACION:.4f}, lon {LON_ESTACION:.4f}")
    print(f"- Precipitacion media anual estacion CRNS: {p_media_anual_mm:.2f} mm/anio")

    q_medio = tabla_caudal[tabla_caudal["tipo_caudal"] == "medio_anual_teorico"][
        "caudal_m3_s"
    ].iloc[0]
    print(f"- Caudal medio anual teorico: {q_medio:.3f} m3/s")

    picos_estacion = tabla_caudal[
        (tabla_caudal["fuente"] == "estacion")
        & (tabla_caudal["tipo_caudal"] == "pico_teorico")
    ].copy()
    for duracion, etiqueta in [(30, "30 min"), (60, "1 h"), (360, "6 h"), (1440, "24 h")]:
        fila_d = picos_estacion[picos_estacion["duracion_min"] == duracion]
        if fila_d.empty:
            print(f"- No hay intensidad maxima observada de estacion para {etiqueta}.")
            continue
        intensidad = float(fila_d["intensidad_mm_h"].iloc[0])
        caudal = float(fila_d["caudal_m3_s"].iloc[0])
        print(f"- Intensidad maxima observada estacion {etiqueta}: {intensidad:.2f} mm/h")
        print(f"- Caudal pico teorico estacion {etiqueta}: {caudal:.3f} m3/s")

    for advertencia in advertencias:
        print(f"- Advertencia: {advertencia}")
    print(
        "- Advertencia: estos caudales son limites superiores absolutos porque "
        "se asume ET = 0 y deltaS = 0."
    )
    print(
        "- Advertencia: IMERG y CMORPH no se usan para duraciones menores a 30 min."
    )
    print("- Archivos generados:")
    for archivo in archivos:
        print(f"  {archivo}")


def main() -> None:
    configurar_matplotlib()
    CUENCA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

    temp_dir = None
    try:
        cuencas, archivo_hydro, nivel, temp_dir = leer_hydrobasins()
        punto = crear_punto_estacion()
        cuenca, metodo = seleccionar_cuenca(cuencas, punto)
        if metodo != "contains":
            print(
                f"Advertencia: la cuenca se selecciono con {metodo}; "
                "el punto pudo caer en un borde o fuera de los poligonos."
            )

        cuenca_salida, atributos = enriquecer_atributos_cuenca(cuenca, nivel)
        area_km2 = float(atributos["area_calculada_km2"].iloc[0])

        cuenca_salida.to_file(SALIDA_GPKG, driver="GPKG")
        atributos.to_csv(SALIDA_ATRIBUTOS, index=False)
        guardar_mapa_cuenca(cuenca_salida, punto)

        p_media_anual_mm = precipitacion_media_anual_estacion()
        maximos, advertencias = maximos_observados_por_fuente()
        tabla_caudal = construir_tabla_caudales(area_km2, p_media_anual_mm, maximos)
        tabla_caudal.to_csv(SALIDA_CAUDAL, index=False)
        guardar_fig19(tabla_caudal)

        archivos = [
            SALIDA_GPKG,
            SALIDA_ATRIBUTOS,
            SALIDA_MAPA,
            SALIDA_CAUDAL,
            FIG19_PNG,
            FIG19_HTML,
        ]
        imprimir_resumen(
            archivo_hydro,
            nivel,
            atributos,
            p_media_anual_mm,
            tabla_caudal,
            advertencias,
            archivos,
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
