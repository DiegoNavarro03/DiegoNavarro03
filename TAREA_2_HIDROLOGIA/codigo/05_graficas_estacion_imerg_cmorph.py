from pathlib import Path
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_PREC_DIR = BASE_DIR / "DATOS_PREC"
DATOS_LIMPIOS_DIR = BASE_DIR / "datos_limpios"
FIGURAS_DIR = BASE_DIR / "figuras"

CSV_DIARIO = DATOS_LIMPIOS_DIR / "precipitacion_diaria_estacion_imerg_cmorph.csv"
CSV_MENSUAL = DATOS_LIMPIOS_DIR / "precipitacion_mensual_estacion_imerg_cmorph.csv"

ESTACION_DIR = next(DATOS_PREC_DIR.glob("ESTACI*"), DATOS_PREC_DIR / "ESTACION")
SATELITE_DIR = next(DATOS_PREC_DIR.glob("SAT*LITE"), DATOS_PREC_DIR / "SATELITE")
IMERG_DIR = SATELITE_DIR / "IMERG"
CMORPH_CSV = SATELITE_DIR / "CMORPH" / "cmorph_champaign_1998_2025.csv"

INICIO_COMPARACION = pd.Timestamp("2015-01-01")
FIN_COMPARACION = pd.Timestamp("2024-12-31")
FIN_COMPARACION_EXCLUSIVO = FIN_COMPARACION + pd.Timedelta(days=1)
ANIOS = range(INICIO_COMPARACION.year, FIN_COMPARACION.year + 1)
PERIODO_TEXTO = "2015–2024"
ZONA_HORARIA_LOCAL = "America/Chicago"

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

COLORES = {
    "estacion": "#1f77b4",
    "imerg": "#d95f02",
    "cmorph": "#2ca25f",
    "referencia": "#4d4d4d",
}
ETIQUETAS = {
    "estacion_mm": "Estación CRNS",
    "imerg_mm": "IMERG",
    "cmorph_mm": "CMORPH",
    "estacion_30min": "Estación CRNS",
    "imerg_30min": "IMERG",
    "cmorph_30min": "CMORPH",
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


def aplicar_layout_plotly(fig: go.Figure, titulo: str) -> None:
    fig.update_layout(
        title=titulo,
        template="plotly_white",
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=75, r=35, t=85, b=65),
    )


def guardar_html(fig: go.Figure, nombre: str) -> None:
    fig.write_html(FIGURAS_DIR / f"{nombre}.html", include_plotlyjs="cdn")


def validar_archivos(rutas: list[Path]) -> None:
    faltantes = [ruta for ruta in rutas if not ruta.exists()]
    if faltantes:
        lista = "\n".join(f"- {ruta}" for ruta in faltantes)
        raise FileNotFoundError(f"No se encontraron estos archivos:\n{lista}")


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
            f"Faltan archivos en {carpeta} para estos años: {anios_faltantes}"
        )
    return rutas


def construir_fecha_hora(df: pd.DataFrame, col_fecha: str, col_hora: str) -> pd.Series:
    fecha = df[col_fecha].astype("Int64").astype(str).str.zfill(8)
    hora = df[col_hora].astype("Int64").astype(str).str.zfill(4)
    return pd.to_datetime(fecha + hora, format="%Y%m%d%H%M", errors="coerce")


def filtrar_periodo(df: pd.DataFrame, columna_fecha: str) -> pd.DataFrame:
    return df[
        (df[columna_fecha] >= INICIO_COMPARACION)
        & (df[columna_fecha] < FIN_COMPARACION_EXCLUSIVO)
    ].copy()


def leer_diario() -> pd.DataFrame:
    validar_archivos([CSV_DIARIO])
    diario = pd.read_csv(CSV_DIARIO, parse_dates=["fecha"])
    columnas = ["estacion_mm", "imerg_mm", "cmorph_mm"]
    diario[columnas] = diario[columnas].apply(pd.to_numeric, errors="coerce")
    return filtrar_periodo(diario, "fecha").sort_values("fecha").reset_index(drop=True)


def leer_mensual() -> pd.DataFrame:
    validar_archivos([CSV_MENSUAL])
    mensual = pd.read_csv(CSV_MENSUAL)
    mensual["fecha"] = pd.to_datetime(mensual["fecha"], format="%Y-%m", errors="coerce")
    columnas = ["estacion_mm", "imerg_mm", "cmorph_mm"]
    mensual[columnas] = mensual[columnas].apply(pd.to_numeric, errors="coerce")
    return mensual.sort_values("fecha").reset_index(drop=True)


def leer_estacion_30min() -> pd.DataFrame:
    rutas = seleccionar_archivos_por_anio(ESTACION_DIR, "CRNS*.txt")
    validar_archivos(rutas)

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

    estacion_30min = (
        estacion.set_index("fecha_hora_utc")["precip_mm"]
        .resample("30min", label="left", closed="left")
        .sum(min_count=1)
        .rename("estacion_30min")
        .reset_index()
    )
    return agregar_hora_local(estacion_30min, "fecha_hora_utc")


def leer_imerg_30min() -> pd.DataFrame:
    rutas = seleccionar_archivos_por_anio(IMERG_DIR, "imerg_champaign_*.csv")
    validar_archivos(rutas)

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
    imerg["precip_mm_per_hr"] = pd.to_numeric(imerg["precip_mm_per_hr"], errors="coerce")
    imerg.loc[imerg["precip_mm_per_hr"] < 0, "precip_mm_per_hr"] = np.nan
    imerg["imerg_30min"] = imerg["precip_mm_per_hr"] * 0.5
    imerg = filtrar_periodo(imerg, "fecha_hora_utc")
    return agregar_hora_local(imerg[["fecha_hora_utc", "imerg_30min"]], "fecha_hora_utc")


def leer_cmorph_30min() -> pd.DataFrame:
    validar_archivos([CMORPH_CSV])

    cmorph = pd.read_csv(CMORPH_CSV, comment="#", na_values=VALORES_FALTANTES)
    cmorph.columns = [col.strip() for col in cmorph.columns]
    cmorph["fecha_hora_utc"] = pd.to_datetime(cmorph["time"], errors="coerce")
    cmorph["precip_mm_per_hr"] = pd.to_numeric(
        cmorph["precip_mm_per_hr"],
        errors="coerce",
    )
    cmorph.loc[cmorph["precip_mm_per_hr"] < 0, "precip_mm_per_hr"] = np.nan
    cmorph["cmorph_30min"] = cmorph["precip_mm_per_hr"] * 0.5
    cmorph = filtrar_periodo(cmorph, "fecha_hora_utc")
    return agregar_hora_local(cmorph[["fecha_hora_utc", "cmorph_30min"]], "fecha_hora_utc")


def agregar_hora_local(df: pd.DataFrame, columna_utc: str) -> pd.DataFrame:
    datos = df.copy()
    utc = datos[columna_utc].dt.tz_localize("UTC")
    datos["fecha_hora_local"] = utc.dt.tz_convert(ZONA_HORARIA_LOCAL).dt.tz_localize(None)
    datos["hora_local"] = datos["fecha_hora_local"].dt.hour
    return datos


def combinar_30min(
    estacion_30min: pd.DataFrame,
    imerg_30min: pd.DataFrame,
    cmorph_30min: pd.DataFrame,
) -> pd.DataFrame:
    datos = estacion_30min[["fecha_hora_utc", "hora_local", "estacion_30min"]]
    datos = datos.merge(
        imerg_30min[["fecha_hora_utc", "imerg_30min"]],
        on="fecha_hora_utc",
        how="outer",
    )
    datos = datos.merge(
        cmorph_30min[["fecha_hora_utc", "cmorph_30min"]],
        on="fecha_hora_utc",
        how="outer",
    )
    datos = agregar_hora_local(datos.drop(columns=["hora_local"], errors="ignore"), "fecha_hora_utc")
    return datos.sort_values("fecha_hora_utc").reset_index(drop=True)


def calcular_metricas(obs: pd.Series, sim: pd.Series) -> dict[str, float]:
    pares = pd.concat([obs, sim], axis=1).dropna()
    observada = pares.iloc[:, 0]
    simulada = pares.iloc[:, 1]
    diferencia = simulada - observada
    suma_observada = observada.sum()
    return {
        "n": float(len(pares)),
        "correlacion": float(observada.corr(simulada)),
        "rmse": float(np.sqrt(np.mean(diferencia**2))),
        "sesgo_medio": float(diferencia.mean()),
        "pbias": float(100 * diferencia.sum() / suma_observada) if suma_observada != 0 else np.nan,
    }


def suavizar_histograma(valores: pd.Series, limite: float, n_bins: int = 55) -> tuple[np.ndarray, np.ndarray]:
    bins = np.linspace(0, limite, n_bins + 1)
    hist, bordes = np.histogram(valores.clip(upper=limite), bins=bins, density=True)
    centros = (bordes[:-1] + bordes[1:]) / 2
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    densidad = np.convolve(hist, kernel / kernel.sum(), mode="same")
    return centros, densidad


def figura_serie_diaria(diario: pd.DataFrame) -> None:
    nombre = "fig01_serie_diaria_estacion_imerg_cmorph"
    titulo = f"Promedio mensual de precipitación diaria: estación CRNS, IMERG y CMORPH ({PERIODO_TEXTO})"
    mensual_promedio = (
        diario.assign(fecha=pd.to_datetime(diario["fecha"]))
        .set_index("fecha")[["estacion_mm", "imerg_mm", "cmorph_mm"]]
        .resample("MS")
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12.5, 4.8))
    ax.plot(mensual_promedio["fecha"], mensual_promedio["estacion_mm"], color=COLORES["estacion"], marker="o", ms=3, lw=1.45, label="Estación CRNS")
    ax.plot(mensual_promedio["fecha"], mensual_promedio["imerg_mm"], color=COLORES["imerg"], marker="o", ms=3, lw=1.45, alpha=0.9, label="IMERG")
    ax.plot(mensual_promedio["fecha"], mensual_promedio["cmorph_mm"], color=COLORES["cmorph"], marker="o", ms=3, lw=1.45, alpha=0.9, label="CMORPH")
    ax.set_title(titulo)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Promedio mensual (mm/día)")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    for columna, color in [("estacion_mm", COLORES["estacion"]), ("imerg_mm", COLORES["imerg"]), ("cmorph_mm", COLORES["cmorph"])]:
        fig_html.add_trace(
            go.Scatter(
                x=mensual_promedio["fecha"],
                y=mensual_promedio[columna],
                mode="lines+markers",
                name=ETIQUETAS[columna],
                line=dict(color=color),
                hovertemplate="%{x|%Y-%m}<br>%{y:.2f} mm/día<extra>%{fullData.name}</extra>",
            )
        )
    aplicar_layout_plotly(fig_html, titulo)
    fig_html.update_xaxes(title_text="Mes")
    fig_html.update_yaxes(title_text="Promedio mensual (mm/día)")
    guardar_html(fig_html, nombre)

    corr_imerg = mensual_promedio["estacion_mm"].corr(mensual_promedio["imerg_mm"])
    corr_cmorph = mensual_promedio["estacion_mm"].corr(mensual_promedio["cmorph_mm"])
    mayor = mensual_promedio.loc[mensual_promedio["estacion_mm"].idxmax()]
    print("\nFigura 1 - Promedio mensual por año")
    print("- Pregunta: ¿Qué tanto IMERG y CMORPH reproducen la variabilidad mensual observada por la estación?")
    print(f"- Patrón principal: el mayor promedio mensual observado fue {mayor['estacion_mm']:.2f} mm/día en {mayor['fecha']:%Y-%m}.")
    print(f"- Evidencia: correlación mensual estación-IMERG = {corr_imerg:.3f}; estación-CMORPH = {corr_cmorph:.3f}.")


def figura_series_diarias_apiladas(diario: pd.DataFrame) -> None:
    nombre = "fig0101_series_diarias_apiladas_estacion_imerg_cmorph"
    titulo = f"Series diarias de precipitación por fuente: estación CRNS, IMERG y CMORPH ({PERIODO_TEXTO})"
    series = [
        ("estacion_mm", "Estación CRNS", COLORES["estacion"]),
        ("imerg_mm", "IMERG", COLORES["imerg"]),
        ("cmorph_mm", "CMORPH", COLORES["cmorph"]),
    ]
    limite_y = diario[["estacion_mm", "imerg_mm", "cmorph_mm"]].max(skipna=True).max() * 1.05

    fig, ejes = plt.subplots(3, 1, figsize=(12.5, 8.0), sharex=True)
    for ax, (columna, etiqueta, color) in zip(ejes, series):
        ax.plot(diario["fecha"], diario[columna], color=color, lw=0.75, alpha=0.85)
        ax.set_title(etiqueta, loc="left", fontsize=10, fontweight="bold")
        ax.set_ylabel("mm/día")
        ax.set_ylim(0, limite_y)
        ax.grid(True, alpha=0.25)
    ejes[-1].set_xlabel("Fecha")
    ejes[-1].xaxis.set_major_locator(mdates.YearLocator())
    ejes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.suptitle(titulo, y=0.995)
    fig.autofmt_xdate()
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[etiqueta for _, etiqueta, _ in series],
        vertical_spacing=0.08,
    )
    for fila, (columna, etiqueta, color) in enumerate(series, start=1):
        fig_html.add_trace(
            go.Scatter(
                x=diario["fecha"],
                y=diario[columna],
                mode="lines",
                name=etiqueta,
                line=dict(color=color, width=1),
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:.2f} mm/día<extra>%{fullData.name}</extra>",
            ),
            row=fila,
            col=1,
        )
    aplicar_layout_plotly(fig_html, titulo)
    fig_html.update_layout(height=850)
    fig_html.update_xaxes(title_text="Fecha", row=3, col=1)
    for fila in range(1, 4):
        fig_html.update_yaxes(title_text="mm/día", range=[0, limite_y], row=fila, col=1)
    guardar_html(fig_html, nombre)

    corr_imerg = diario["estacion_mm"].corr(diario["imerg_mm"])
    corr_cmorph = diario["estacion_mm"].corr(diario["cmorph_mm"])
    mayor = diario.loc[diario["estacion_mm"].idxmax()]
    print("\nFigura 1 - Series diarias apiladas")
    print("- Pregunta: ¿Qué tanto IMERG y CMORPH reproducen la variabilidad diaria observada por la estación?")
    print(f"- Patrón principal: el evento diario máximo observado fue {mayor['estacion_mm']:.2f} mm el {mayor['fecha'].date()}.")
    print(f"- Evidencia: correlación diaria estación-IMERG = {corr_imerg:.3f}; estación-CMORPH = {corr_cmorph:.3f}.")


def figura_acumulado_mensual(mensual: pd.DataFrame) -> None:
    nombre = "fig02_acumulado_mensual_estacion_imerg_cmorph"
    titulo = f"Acumulado mensual de precipitación: estación CRNS, IMERG y CMORPH ({PERIODO_TEXTO})"

    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    ax.plot(mensual["fecha"], mensual["estacion_mm"], color=COLORES["estacion"], marker="o", ms=3, lw=1.4, label="Estación CRNS")
    ax.plot(mensual["fecha"], mensual["imerg_mm"], color=COLORES["imerg"], marker="o", ms=3, lw=1.4, label="IMERG")
    ax.plot(mensual["fecha"], mensual["cmorph_mm"], color=COLORES["cmorph"], marker="o", ms=3, lw=1.4, label="CMORPH")
    ax.set_title(titulo)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Precipitación mensual acumulada (mm/mes)")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    for columna, color in [("estacion_mm", COLORES["estacion"]), ("imerg_mm", COLORES["imerg"]), ("cmorph_mm", COLORES["cmorph"])]:
        fig_html.add_trace(go.Scatter(x=mensual["fecha"], y=mensual[columna], mode="lines+markers", name=ETIQUETAS[columna], line=dict(color=color)))
    aplicar_layout_plotly(fig_html, titulo)
    fig_html.update_xaxes(title_text="Mes")
    fig_html.update_yaxes(title_text="Precipitación mensual acumulada (mm/mes)")
    guardar_html(fig_html, nombre)

    humedo = mensual.loc[mensual["estacion_mm"].idxmax()]
    r_imerg = mensual["estacion_mm"].corr(mensual["imerg_mm"])
    r_cmorph = mensual["estacion_mm"].corr(mensual["cmorph_mm"])
    print("\nFigura 2 - Acumulado mensual")
    print("- Pregunta: ¿Los productos satelitales representan bien la estacionalidad mensual de la precipitación?")
    print(f"- Patrón principal: el mes más lluvioso observado fue {humedo['fecha'].strftime('%Y-%m')} con {humedo['estacion_mm']:.2f} mm.")
    print(f"- Evidencia: correlación mensual estación-IMERG = {r_imerg:.3f}; estación-CMORPH = {r_cmorph:.3f}.")


def figura_ciclo_diurno(datos_30min: pd.DataFrame) -> None:
    nombre = "fig03_ciclo_diurno_estacion_imerg_cmorph"
    titulo = f"Ciclo diurno de la precipitación en hora local de Champaign ({PERIODO_TEXTO})"
    columnas = ["estacion_30min", "imerg_30min", "cmorph_30min"]
    ciclo = datos_30min.groupby("hora_local", as_index=False)[columnas].sum(min_count=1)

    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    for columna, color in zip(columnas, [COLORES["estacion"], COLORES["imerg"], COLORES["cmorph"]]):
        ax.plot(ciclo["hora_local"], ciclo[columna], color=color, marker="o", lw=1.7, label=ETIQUETAS[columna])
    ax.set_title(titulo)
    ax.set_xlabel("Hora local")
    ax.set_ylabel("Precipitación acumulada por hora (mm)")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    for columna, color in zip(columnas, [COLORES["estacion"], COLORES["imerg"], COLORES["cmorph"]]):
        fig_html.add_trace(go.Scatter(x=ciclo["hora_local"], y=ciclo[columna], mode="lines+markers", name=ETIQUETAS[columna], line=dict(color=color)))
    aplicar_layout_plotly(fig_html, titulo)
    fig_html.update_xaxes(title_text="Hora local", dtick=2)
    fig_html.update_yaxes(title_text="Precipitación acumulada por hora (mm)")
    guardar_html(fig_html, nombre)

    picos = {columna: ciclo.loc[ciclo[columna].idxmax(), "hora_local"] for columna in columnas}
    acumulados_pico = {columna: ciclo[columna].max() for columna in columnas}
    print("\nFigura 3 - Ciclo diurno de la precipitación")
    print("- Pregunta: ¿A qué hora local se concentra más la precipitación observada en tierra y cómo se compara con los productos satelitales?")
    print(f"- Patrón principal: la estación alcanza su máximo alrededor de las {int(picos['estacion_30min']):02d}:00 hora local.")
    print(
        "- Evidencia: horas pico estación/IMERG/CMORPH = "
        f"{int(picos['estacion_30min']):02d}:00, {int(picos['imerg_30min']):02d}:00 y {int(picos['cmorph_30min']):02d}:00; "
        f"acumulado en hora pico de la estación = {acumulados_pico['estacion_30min']:.2f} mm."
    )


def figura_distribucion_intensidades(datos_30min: pd.DataFrame) -> None:
    nombre = "fig04_distribucion_intensidades_estacion_imerg_cmorph"
    titulo = f"Distribución de intensidades positivas a resolución de 30 minutos ({PERIODO_TEXTO})"
    columnas = ["estacion_30min", "imerg_30min", "cmorph_30min"]
    positivos = {col: datos_30min[col].dropna().loc[lambda serie: serie > 0] for col in columnas}
    limite = pd.concat(positivos.values(), ignore_index=True).quantile(0.99)

    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    fig_html = go.Figure()
    for columna, color in zip(columnas, [COLORES["estacion"], COLORES["imerg"], COLORES["cmorph"]]):
        x, y = suavizar_histograma(positivos[columna], limite)
        ax.plot(x, y, color=color, lw=2.0, label=ETIQUETAS[columna])
        fig_html.add_trace(go.Scatter(x=x, y=y, mode="lines", name=ETIQUETAS[columna], line=dict(color=color)))

    ax.set_title(titulo)
    ax.set_xlabel("Precipitación en 30 minutos (mm/30 min)")
    ax.set_ylabel("Densidad suavizada")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    aplicar_layout_plotly(fig_html, titulo)
    fig_html.update_xaxes(title_text="Precipitación en 30 minutos (mm/30 min)")
    fig_html.update_yaxes(title_text="Densidad suavizada")
    guardar_html(fig_html, nombre)

    medianas = {col: positivos[col].median() for col in columnas}
    p95 = {col: positivos[col].quantile(0.95) for col in columnas}
    print("\nFigura 4 - Distribución de intensidades de precipitación")
    print("- Pregunta: ¿La lluvia ocurre principalmente como eventos débiles o intensos, y cómo representan IMERG y CMORPH esa distribución?")
    print(f"- Patrón principal: predominan intensidades positivas bajas; la mediana observada es {medianas['estacion_30min']:.2f} mm/30 min.")
    print(
        "- Evidencia: percentil 95 estación/IMERG/CMORPH = "
        f"{p95['estacion_30min']:.2f}, {p95['imerg_30min']:.2f} y {p95['cmorph_30min']:.2f} mm/30 min."
    )


def figura_dispersion_diaria(diario: pd.DataFrame) -> dict[str, dict[str, float]]:
    nombre = "fig05_dispersion_diaria_estacion_imerg_cmorph"
    titulo = f"Dispersión diaria: estación CRNS frente a IMERG y CMORPH ({PERIODO_TEXTO})"
    metricas = {
        "IMERG": calcular_metricas(diario["estacion_mm"], diario["imerg_mm"]),
        "CMORPH": calcular_metricas(diario["estacion_mm"], diario["cmorph_mm"]),
    }
    pares = diario[["estacion_mm", "imerg_mm", "cmorph_mm"]].dropna()
    limite = float(np.nanmax(pares.to_numpy()) * 1.05)

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.scatter(pares["estacion_mm"], pares["imerg_mm"], s=22, alpha=0.62, color=COLORES["imerg"], edgecolor="none", label="IMERG")
    ax.scatter(pares["estacion_mm"], pares["cmorph_mm"], s=22, alpha=0.62, color=COLORES["cmorph"], edgecolor="none", label="CMORPH")
    ax.plot([0, limite], [0, limite], color=COLORES["referencia"], lw=1.3, ls="--", label="Línea 1:1")
    ax.set_title(titulo)
    ax.set_xlabel("Estación CRNS (mm/día)")
    ax.set_ylabel("Producto satelital (mm/día)")
    ax.set_xlim(0, limite)
    ax.set_ylim(0, limite)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    fig_html.add_trace(go.Scatter(x=pares["estacion_mm"], y=pares["imerg_mm"], mode="markers", name="IMERG", marker=dict(color=COLORES["imerg"], opacity=0.62, size=7)))
    fig_html.add_trace(go.Scatter(x=pares["estacion_mm"], y=pares["cmorph_mm"], mode="markers", name="CMORPH", marker=dict(color=COLORES["cmorph"], opacity=0.62, size=7)))
    fig_html.add_trace(go.Scatter(x=[0, limite], y=[0, limite], mode="lines", name="Línea 1:1", line=dict(color=COLORES["referencia"], dash="dash")))
    aplicar_layout_plotly(fig_html, titulo)
    fig_html.update_xaxes(title_text="Estación CRNS (mm/día)", range=[0, limite])
    fig_html.update_yaxes(title_text="Producto satelital (mm/día)", range=[0, limite], scaleanchor="x", scaleratio=1)
    guardar_html(fig_html, nombre)

    mejor = min(metricas, key=lambda producto: metricas[producto]["rmse"])
    print("\nFigura 5 - Dispersión diaria estación vs satélites")
    print("- Pregunta: ¿Cuál producto satelital se parece más a la estación: IMERG o CMORPH?")
    print(f"- Patrón principal: {mejor} presenta el menor RMSE diario frente a la estación.")
    print(
        "- Evidencia: "
        f"IMERG r = {metricas['IMERG']['correlacion']:.3f}, RMSE = {metricas['IMERG']['rmse']:.2f} mm/día, "
        f"sesgo = {metricas['IMERG']['sesgo_medio']:.2f} mm/día, PBIAS = {metricas['IMERG']['pbias']:.1f}%; "
        f"CMORPH r = {metricas['CMORPH']['correlacion']:.3f}, RMSE = {metricas['CMORPH']['rmse']:.2f} mm/día, "
        f"sesgo = {metricas['CMORPH']['sesgo_medio']:.2f} mm/día, PBIAS = {metricas['CMORPH']['pbias']:.1f}%."
    )
    return metricas


def imprimir_metricas(metricas: dict[str, dict[str, float]]) -> None:
    print("\nMétricas diarias de desempeño")
    for producto, valores in metricas.items():
        print(
            f"- {producto}: n = {valores['n']:.0f}, "
            f"correlación = {valores['correlacion']:.3f}, "
            f"RMSE = {valores['rmse']:.2f} mm/día, "
            f"sesgo medio = {valores['sesgo_medio']:.2f} mm/día, "
            f"PBIAS = {valores['pbias']:.1f}%"
        )


def imprimir_resumen_archivos() -> None:
    archivos = [
        "fig01_serie_diaria_estacion_imerg_cmorph",
        "fig02_acumulado_mensual_estacion_imerg_cmorph",
        "fig03_ciclo_diurno_estacion_imerg_cmorph",
        "fig04_distribucion_intensidades_estacion_imerg_cmorph",
        "fig05_dispersion_diaria_estacion_imerg_cmorph",
        "fig0101_series_diarias_apiladas_estacion_imerg_cmorph",
    ]
    print("\nArchivos regenerados")
    for nombre in archivos:
        print(f"- {FIGURAS_DIR / f'{nombre}.png'}")
        print(f"- {FIGURAS_DIR / f'{nombre}.html'}")
    print("- Confirmación: no se crearon archivos duplicados innecesarios; se sobrescribieron las figuras existentes.")


def main() -> None:
    configurar_matplotlib()
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

    diario = leer_diario()
    mensual = leer_mensual()
    estacion_30min = leer_estacion_30min()
    imerg_30min = leer_imerg_30min()
    cmorph_30min = leer_cmorph_30min()
    datos_30min = combinar_30min(estacion_30min, imerg_30min, cmorph_30min)

    figura_serie_diaria(diario)
    figura_series_diarias_apiladas(diario)
    figura_acumulado_mensual(mensual)
    figura_ciclo_diurno(datos_30min)
    figura_distribucion_intensidades(datos_30min)
    metricas = figura_dispersion_diaria(diario)
    imprimir_metricas(metricas)
    imprimir_resumen_archivos()


if __name__ == "__main__":
    main()
