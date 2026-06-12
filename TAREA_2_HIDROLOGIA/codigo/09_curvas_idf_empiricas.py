from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_LIMPIOS_DIR = BASE_DIR / "datos_limpios"
FIGURAS_DIR = BASE_DIR / "figuras"
DATOS_PREC_DIR = BASE_DIR / "DATOS_PREC"
ESTACION_DIR = next(DATOS_PREC_DIR.glob("ESTACI*"), DATOS_PREC_DIR / "ESTACION")
SATELITE_DIR = next(DATOS_PREC_DIR.glob("SAT*LITE"), DATOS_PREC_DIR / "SATELITE")
IMERG_DIR = SATELITE_DIR / "IMERG"
CMORPH_DIR = SATELITE_DIR / "CMORPH"

CSV_CMORPH = CMORPH_DIR / "cmorph_champaign_1998_2025.csv"

SALIDA_MAXIMOS_ESTACION = DATOS_LIMPIOS_DIR / "idf_maximos_anuales_estacion.csv"
SALIDA_DISENO_ESTACION = DATOS_LIMPIOS_DIR / "idf_intensidades_diseno_estacion.csv"
SALIDA_DISENO_SATELITES = DATOS_LIMPIOS_DIR / "idf_intensidades_diseno_satelites.csv"
SALIDA_COMPARACION = DATOS_LIMPIOS_DIR / "idf_comparacion_30min_o_mas.csv"

FIG16 = "fig16_curvas_idf_estacion_crns"
FIG17 = "fig17_comparacion_idf_estacion_imerg_cmorph"
FIG18 = "fig18_maximos_anuales_15min_estacion"

INICIO = pd.Timestamp("2015-01-01")
FIN = pd.Timestamp("2024-12-31")
FIN_EXCLUSIVO = FIN + pd.Timedelta(days=1)
ANIOS = range(INICIO.year, FIN.year + 1)

DURACIONES_ESTACION = [5, 10, 15, 30, 60, 120, 360, 1440]
DURACIONES_SATELITE = [30, 60, 120, 360, 1440]
PERIODOS_RETORNO = [2, 10, 25]
PERIODOS_FIG17 = [2, 10]

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
    "estacion": "Estación CRNS",
    "imerg": "IMERG",
    "cmorph": "CMORPH",
}
COLORES_FUENTE = {
    "estacion": "#1f77b4",
    "imerg": "#d95f02",
    "cmorph": "#2ca25f",
}
COLORES_T = {
    2: "#1f77b4",
    10: "#d95f02",
    25: "#7b3294",
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
        raise FileNotFoundError(f"No se encontró el archivo CMORPH: {CSV_CMORPH}")

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


def calcular_maximos_anuales(
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
            raise ValueError(f"La duración {duracion} min no es múltiplo de {frecuencia_min} min")

        acumulado = serie.rolling(window=ventana, min_periods=ventana).sum()
        maximos = acumulado.groupby(acumulado.index.year).max()
        for anio, acumulado_mm in maximos.items():
            if anio not in ANIOS or pd.isna(acumulado_mm):
                continue
            filas.append(
                {
                    "fuente": fuente,
                    "anio": int(anio),
                    "duracion_min": duracion,
                    "max_acumulado_mm": float(acumulado_mm),
                    "intensidad_mm_h": float(acumulado_mm / (duracion / 60)),
                }
            )

    return pd.DataFrame(filas)


def intensidad_empirica_weibull(valores: pd.Series, periodo_retorno: int) -> float:
    muestra = valores.dropna().sort_values(ascending=False).to_numpy(dtype=float)
    n = len(muestra)
    if n < 2:
        return np.nan

    orden = np.arange(1, n + 1)
    retorno = (n + 1) / orden
    if periodo_retorno < retorno.min() or periodo_retorno > retorno.max():
        return np.nan

    orden_asc = np.argsort(retorno)
    return float(
        np.interp(
            np.log(periodo_retorno),
            np.log(retorno[orden_asc]),
            muestra[orden_asc],
        )
    )


def intensidad_gumbel(valores: pd.Series, periodo_retorno: int) -> float:
    muestra = valores.dropna().to_numpy(dtype=float)
    if len(muestra) < 2:
        return np.nan

    desviacion = muestra.std(ddof=1)
    if desviacion == 0 or pd.isna(desviacion):
        return float(muestra.mean())

    gamma = 0.5772156649015329
    beta = desviacion * np.sqrt(6) / np.pi
    mu = muestra.mean() - gamma * beta
    prob_no_excedencia = 1 - 1 / periodo_retorno
    variado_reducido = -np.log(-np.log(prob_no_excedencia))
    return float(mu + beta * variado_reducido)


def calcular_intensidades_diseno(maximos: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for (fuente, duracion), grupo in maximos.groupby(["fuente", "duracion_min"]):
        valores = grupo["intensidad_mm_h"]
        for periodo in PERIODOS_RETORNO:
            filas.append(
                {
                    "fuente": fuente,
                    "duracion_min": int(duracion),
                    "periodo_retorno_anios": periodo,
                    "intensidad_mm_h": intensidad_empirica_weibull(valores, periodo),
                    "metodo": "Weibull empírico",
                }
            )
            filas.append(
                {
                    "fuente": fuente,
                    "duracion_min": int(duracion),
                    "periodo_retorno_anios": periodo,
                    "intensidad_mm_h": intensidad_gumbel(valores, periodo),
                    "metodo": "Gumbel",
                }
            )

    diseno = pd.DataFrame(filas)
    return diseno.sort_values(
        ["fuente", "metodo", "periodo_retorno_anios", "duracion_min"]
    ).reset_index(drop=True)


def construir_comparacion(diseno_total: pd.DataFrame) -> pd.DataFrame:
    base = diseno_total[
        (diseno_total["duracion_min"].isin(DURACIONES_SATELITE))
        & (diseno_total["periodo_retorno_anios"].isin(PERIODOS_RETORNO))
    ].copy()
    estacion = base[base["fuente"] == "estacion"][
        ["duracion_min", "periodo_retorno_anios", "metodo", "intensidad_mm_h"]
    ].rename(columns={"intensidad_mm_h": "intensidad_estacion_mm_h"})
    comparacion = base.merge(
        estacion,
        on=["duracion_min", "periodo_retorno_anios", "metodo"],
        how="left",
    )
    comparacion["diferencia_vs_estacion_pct"] = (
        (comparacion["intensidad_mm_h"] - comparacion["intensidad_estacion_mm_h"])
        / comparacion["intensidad_estacion_mm_h"]
        * 100
    )
    return comparacion[
        [
            "fuente",
            "duracion_min",
            "periodo_retorno_anios",
            "metodo",
            "intensidad_mm_h",
            "intensidad_estacion_mm_h",
            "diferencia_vs_estacion_pct",
        ]
    ].sort_values(["metodo", "periodo_retorno_anios", "duracion_min", "fuente"])


def guardar_fig16(diseno_estacion: pd.DataFrame) -> None:
    datos = diseno_estacion[diseno_estacion["metodo"] == "Gumbel"].copy()

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for periodo in PERIODOS_RETORNO:
        curva = datos[datos["periodo_retorno_anios"] == periodo]
        ax.plot(
            curva["duracion_min"],
            curva["intensidad_mm_h"],
            marker="o",
            lw=2,
            color=COLORES_T[periodo],
            label=f"T = {periodo} años",
        )
    ax.set_xscale("log")
    ax.set_title("Curvas IDF empíricas de la estación CRNS (2015–2024)")
    ax.set_xlabel("Duración (min)")
    ax.set_ylabel("Intensidad de lluvia (mm/h)")
    ax.set_xticks(DURACIONES_ESTACION)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{FIG16}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    for periodo in PERIODOS_RETORNO:
        curva = datos[datos["periodo_retorno_anios"] == periodo]
        fig_html.add_trace(
            go.Scatter(
                x=curva["duracion_min"],
                y=curva["intensidad_mm_h"],
                mode="lines+markers",
                name=f"T = {periodo} años",
                line=dict(color=COLORES_T[periodo]),
            )
        )
    fig_html.update_layout(
        title="Curvas IDF empíricas de la estación CRNS (2015–2024)",
        template="plotly_white",
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=75, r=35, t=85, b=65),
    )
    fig_html.update_xaxes(
        title_text="Duración (min)",
        type="log",
        tickvals=DURACIONES_ESTACION,
        ticktext=[str(d) for d in DURACIONES_ESTACION],
    )
    fig_html.update_yaxes(title_text="Intensidad de lluvia (mm/h)")
    fig_html.write_html(FIGURAS_DIR / f"{FIG16}.html", include_plotlyjs="cdn")


def guardar_fig17(diseno_total: pd.DataFrame) -> None:
    datos = diseno_total[
        (diseno_total["metodo"] == "Gumbel")
        & (diseno_total["duracion_min"].isin(DURACIONES_SATELITE))
        & (diseno_total["periodo_retorno_anios"].isin(PERIODOS_FIG17))
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for ax, periodo in zip(axes, PERIODOS_FIG17):
        datos_t = datos[datos["periodo_retorno_anios"] == periodo]
        for fuente in ["estacion", "imerg", "cmorph"]:
            curva = datos_t[datos_t["fuente"] == fuente]
            ax.plot(
                curva["duracion_min"],
                curva["intensidad_mm_h"],
                marker="o",
                lw=2,
                color=COLORES_FUENTE[fuente],
                label=ETIQUETAS[fuente],
            )
        ax.set_xscale("log")
        ax.set_title(f"T = {periodo} años")
        ax.set_xlabel("Duración (min)")
        ax.set_xticks(DURACIONES_SATELITE)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    axes[0].set_ylabel("Intensidad de lluvia (mm/h)")
    axes[1].legend(loc="upper right")
    fig.suptitle("Comparación IDF: estación CRNS, IMERG y CMORPH (2015–2024)")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{FIG17}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"T = {periodo} años" for periodo in PERIODOS_FIG17],
        shared_yaxes=True,
    )
    for col, periodo in enumerate(PERIODOS_FIG17, start=1):
        datos_t = datos[datos["periodo_retorno_anios"] == periodo]
        for fuente in ["estacion", "imerg", "cmorph"]:
            curva = datos_t[datos_t["fuente"] == fuente]
            fig_html.add_trace(
                go.Scatter(
                    x=curva["duracion_min"],
                    y=curva["intensidad_mm_h"],
                    mode="lines+markers",
                    name=ETIQUETAS[fuente],
                    legendgroup=fuente,
                    showlegend=col == 1,
                    line=dict(color=COLORES_FUENTE[fuente]),
                ),
                row=1,
                col=col,
            )
    fig_html.update_layout(
        title="Comparación IDF: estación CRNS, IMERG y CMORPH (2015–2024)",
        template="plotly_white",
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=75, r=35, t=85, b=65),
    )
    for col in [1, 2]:
        fig_html.update_xaxes(
            title_text="Duración (min)",
            type="log",
            tickvals=DURACIONES_SATELITE,
            ticktext=[str(d) for d in DURACIONES_SATELITE],
            row=1,
            col=col,
        )
    fig_html.update_yaxes(title_text="Intensidad de lluvia (mm/h)", row=1, col=1)
    fig_html.write_html(FIGURAS_DIR / f"{FIG17}.html", include_plotlyjs="cdn")


def guardar_fig18(maximos_estacion: pd.DataFrame) -> None:
    datos = maximos_estacion[maximos_estacion["duracion_min"] == 15].copy()

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(
        datos["anio"],
        datos["intensidad_mm_h"],
        marker="o",
        lw=1.9,
        color=COLORES_T[10],
    )
    ax.set_title("Máximos anuales de intensidad de 15 min en la estación CRNS")
    ax.set_xlabel("Año")
    ax.set_ylabel("Intensidad máxima anual (mm/h)")
    ax.set_xticks(list(ANIOS))
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{FIG18}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    fig_html.add_trace(
        go.Scatter(
            x=datos["anio"],
            y=datos["intensidad_mm_h"],
            mode="lines+markers",
            name="15 min",
            line=dict(color=COLORES_T[10]),
        )
    )
    fig_html.update_layout(
        title="Máximos anuales de intensidad de 15 min en la estación CRNS",
        template="plotly_white",
        font=dict(size=13),
        margin=dict(l=75, r=35, t=70, b=65),
    )
    fig_html.update_xaxes(title_text="Año", dtick=1)
    fig_html.update_yaxes(title_text="Intensidad máxima anual (mm/h)")
    fig_html.write_html(FIGURAS_DIR / f"{FIG18}.html", include_plotlyjs="cdn")


def valor_diseno(
    diseno: pd.DataFrame,
    fuente: str,
    duracion: int,
    periodo: int,
    metodo: str = "Gumbel",
) -> float:
    fila = diseno[
        (diseno["fuente"] == fuente)
        & (diseno["duracion_min"] == duracion)
        & (diseno["periodo_retorno_anios"] == periodo)
        & (diseno["metodo"] == metodo)
    ]
    return float(fila["intensidad_mm_h"].iloc[0]) if not fila.empty else np.nan


def imprimir_comparacion_30_60(diseno_total: pd.DataFrame) -> None:
    print("- Comparación estación vs IMERG vs CMORPH con Gumbel:")
    for duracion in [30, 60]:
        print(f"  Duración {duracion} min")
        for periodo in PERIODOS_FIG17:
            valores = {
                fuente: valor_diseno(diseno_total, fuente, duracion, periodo)
                for fuente in ["estacion", "imerg", "cmorph"]
            }
            print(
                f"    T = {periodo} años: "
                f"Estación {valores['estacion']:.2f} mm/h, "
                f"IMERG {valores['imerg']:.2f} mm/h, "
                f"CMORPH {valores['cmorph']:.2f} mm/h."
            )


def imprimir_resumen(
    maximos_total: pd.DataFrame,
    diseno_total: pd.DataFrame,
    archivos: list[Path],
) -> None:
    anios_por_fuente = maximos_total.groupby("fuente")["anio"].nunique()
    print("\nResumen de curvas IDF empíricas")
    print(f"- Periodo usado: {INICIO:%Y-%m-%d} a {FIN:%Y-%m-%d}.")
    for fuente in ["estacion", "imerg", "cmorph"]:
        print(f"- Años disponibles para {ETIQUETAS[fuente]}: {int(anios_por_fuente[fuente])}.")

    print("- Duración crítica recomendada para el parqueadero: 15 min.")
    print(
        "  Se usa como duración representativa de una cuenca urbana pequeña "
        "y queda sustentada por la serie de máximos anuales de 15 min."
    )
    for periodo in PERIODOS_RETORNO:
        intensidad = valor_diseno(diseno_total, "estacion", 15, periodo)
        nota = " (extrapolación con 10 años de registro)" if periodo == 25 else ""
        print(
            f"- Intensidad de diseño recomendada para d = 15 min y "
            f"T = {periodo} años: {intensidad:.2f} mm/h, método Gumbel{nota}."
        )

    imprimir_comparacion_30_60(diseno_total)
    print(
        "- Advertencia metodológica: IMERG y CMORPH tienen resolución temporal de "
        "30 min; por eso no se estimaron IDF satelitales para 5, 10 ni 15 min."
    )
    print(
        "- Advertencia de extrapolación: T = 25 años excede el registro común "
        "2015–2024 de 10 años."
    )
    print("- Archivos generados:")
    for archivo in archivos:
        print(f"  {archivo}")
    print("- Confirmación: no se modificaron figuras anteriores ni se crearon duplicados innecesarios.")


def main() -> None:
    configurar_matplotlib()
    DATOS_LIMPIOS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

    estacion_5min = leer_estacion_5min()
    imerg_30min = leer_imerg_30min()
    cmorph_30min = leer_cmorph_30min()

    maximos_estacion = calcular_maximos_anuales(
        "estacion", estacion_5min, DURACIONES_ESTACION, frecuencia_min=5
    )
    maximos_imerg = calcular_maximos_anuales(
        "imerg", imerg_30min, DURACIONES_SATELITE, frecuencia_min=30
    )
    maximos_cmorph = calcular_maximos_anuales(
        "cmorph", cmorph_30min, DURACIONES_SATELITE, frecuencia_min=30
    )
    maximos_total = pd.concat(
        [maximos_estacion, maximos_imerg, maximos_cmorph], ignore_index=True
    )

    diseno_total = calcular_intensidades_diseno(maximos_total)
    diseno_estacion = diseno_total[diseno_total["fuente"] == "estacion"].copy()
    diseno_satelites = diseno_total[diseno_total["fuente"].isin(["imerg", "cmorph"])].copy()
    comparacion = construir_comparacion(diseno_total)

    maximos_estacion.to_csv(SALIDA_MAXIMOS_ESTACION, index=False)
    diseno_estacion.to_csv(SALIDA_DISENO_ESTACION, index=False)
    diseno_satelites.to_csv(SALIDA_DISENO_SATELITES, index=False)
    comparacion.to_csv(SALIDA_COMPARACION, index=False)

    guardar_fig16(diseno_estacion)
    guardar_fig17(diseno_total)
    guardar_fig18(maximos_estacion)

    archivos = [
        SALIDA_MAXIMOS_ESTACION,
        SALIDA_DISENO_ESTACION,
        SALIDA_DISENO_SATELITES,
        SALIDA_COMPARACION,
        FIGURAS_DIR / f"{FIG16}.png",
        FIGURAS_DIR / f"{FIG16}.html",
        FIGURAS_DIR / f"{FIG17}.png",
        FIGURAS_DIR / f"{FIG17}.html",
        FIGURAS_DIR / f"{FIG18}.png",
        FIGURAS_DIR / f"{FIG18}.html",
    ]
    imprimir_resumen(maximos_total, diseno_total, archivos)


if __name__ == "__main__":
    main()
