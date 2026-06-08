from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_LIMPIOS_DIR = BASE_DIR / "datos_limpios"
FIGURAS_DIR = BASE_DIR / "figuras"

CSV_DIARIO = DATOS_LIMPIOS_DIR / "precipitacion_diaria_estacion_imerg_cmorph.csv"
CSV_MENSUAL = DATOS_LIMPIOS_DIR / "precipitacion_mensual_estacion_imerg_cmorph.csv"
CSV_ANUAL = DATOS_LIMPIOS_DIR / "precipitacion_anual_estacion_imerg_cmorph.csv"
CSV_MOMENTOS = DATOS_LIMPIOS_DIR / "momentos_estadisticos_diarios_estacion_imerg_cmorph.csv"
CSV_PERCENTILES = DATOS_LIMPIOS_DIR / "percentiles_diarios_estacion_imerg_cmorph.csv"
CSV_L_MOMENTOS = DATOS_LIMPIOS_DIR / "l_momentos_diarios_estacion_imerg_cmorph.csv"

INICIO = pd.Timestamp("2015-01-01")
FIN = pd.Timestamp("2024-12-31")
PERIODO_TEXTO = "2015–2024"

COLUMNAS = ["estacion_mm", "imerg_mm", "cmorph_mm"]
FUENTES = {
    "estacion_mm": "Estación CRNS",
    "imerg_mm": "IMERG",
    "cmorph_mm": "CMORPH",
}
COLORES = {
    "estacion_mm": "#1f77b4",
    "imerg_mm": "#d95f02",
    "cmorph_mm": "#2ca25f",
    "referencia": "#4d4d4d",
}
PERCENTILES = [0, 5, 10, 25, 50, 75, 90, 95, 99, 100]


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


def leer_diario() -> pd.DataFrame:
    if not CSV_DIARIO.exists():
        raise FileNotFoundError(f"No se encontró el archivo diario: {CSV_DIARIO}")

    diario = pd.read_csv(CSV_DIARIO)
    diario["fecha"] = pd.to_datetime(diario["fecha"], errors="coerce")
    diario[COLUMNAS] = diario[COLUMNAS].apply(pd.to_numeric, errors="coerce")
    diario = diario[(diario["fecha"] >= INICIO) & (diario["fecha"] <= FIN)].copy()
    diario["anio"] = diario["fecha"].dt.year
    return diario.sort_values("fecha").reset_index(drop=True)


def leer_mensual() -> pd.DataFrame:
    if not CSV_MENSUAL.exists():
        raise FileNotFoundError(f"No se encontró el archivo mensual: {CSV_MENSUAL}")

    mensual = pd.read_csv(CSV_MENSUAL)
    mensual["fecha"] = pd.to_datetime(mensual["fecha"], format="%Y-%m", errors="coerce")
    mensual[COLUMNAS] = mensual[COLUMNAS].apply(pd.to_numeric, errors="coerce")
    return mensual[(mensual["fecha"] >= INICIO) & (mensual["fecha"] <= FIN)].sort_values("fecha").reset_index(drop=True)


def calcular_anual(diario: pd.DataFrame) -> pd.DataFrame:
    anual = diario.groupby("anio", as_index=False)[COLUMNAS].sum(min_count=1)
    return anual[["anio", *COLUMNAS]]


def calcular_momentos(diario: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for columna in COLUMNAS:
        serie = diario[columna].dropna()
        filas.append(
            {
                "fuente": FUENTES[columna],
                "n_dias_validos": int(serie.count()),
                "dias_lluvia": int((serie > 0).sum()),
                "media_mm": serie.mean(),
                "varianza_mm2": serie.var(ddof=1),
                "desviacion_estandar_mm": serie.std(ddof=1),
                "asimetria": serie.skew(),
                "curtosis": serie.kurt(),
                "minimo_mm": serie.min(),
                "maximo_mm": serie.max(),
                "total_mm": serie.sum(),
            }
        )
    return pd.DataFrame(filas)


def calcular_percentiles(diario: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for columna in COLUMNAS:
        serie = diario[columna].dropna()
        fila = {"fuente": FUENTES[columna]}
        for percentil in PERCENTILES:
            fila[f"p{percentil:02d}_mm"] = np.percentile(serie, percentil)
        filas.append(fila)
    return pd.DataFrame(filas)


def calcular_l_momentos_serie(serie: pd.Series) -> dict[str, float]:
    valores = np.sort(serie.dropna().to_numpy(dtype=float))
    n = len(valores)
    if n < 3:
        return {
            "n_dias_validos": n,
            "l1_media_l_mm": np.nan,
            "l2_escala_l_mm": np.nan,
            "l3_asimetria_l_mm": np.nan,
            "t_l_cv": np.nan,
            "t3_l_asimetria": np.nan,
        }

    i = np.arange(n, dtype=float)
    b0 = valores.mean()
    b1 = np.sum((i / (n - 1)) * valores) / n
    b2 = np.sum((i * (i - 1) / ((n - 1) * (n - 2))) * valores) / n
    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    return {
        "n_dias_validos": n,
        "l1_media_l_mm": l1,
        "l2_escala_l_mm": l2,
        "l3_asimetria_l_mm": l3,
        "t_l_cv": l2 / l1 if l1 != 0 else np.nan,
        "t3_l_asimetria": l3 / l2 if l2 != 0 else np.nan,
    }


def calcular_l_momentos(diario: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for columna in COLUMNAS:
        fila = {"fuente": FUENTES[columna]}
        fila.update(calcular_l_momentos_serie(diario[columna]))
        filas.append(fila)
    return pd.DataFrame(filas)


def figura_precipitacion_anual(anual: pd.DataFrame) -> None:
    nombre = "fig09_precipitacion_anual_estacion_imerg_cmorph"
    titulo = f"Precipitación anual acumulada: estación CRNS, IMERG y CMORPH ({PERIODO_TEXTO})"

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for columna in COLUMNAS:
        ax.plot(
            anual["anio"],
            anual[columna],
            marker="o",
            lw=1.8,
            color=COLORES[columna],
            label=FUENTES[columna],
        )
    ax.set_title(titulo)
    ax.set_xlabel("Año")
    ax.set_ylabel("Precipitación anual (mm/año)")
    ax.set_xticks(anual["anio"])
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    for columna in COLUMNAS:
        fig_html.add_trace(
            go.Scatter(
                x=anual["anio"],
                y=anual[columna],
                mode="lines+markers",
                name=FUENTES[columna],
                line=dict(color=COLORES[columna]),
            )
        )
    aplicar_layout_plotly(fig_html, titulo)
    fig_html.update_xaxes(title_text="Año", dtick=1)
    fig_html.update_yaxes(title_text="Precipitación anual (mm/año)")
    guardar_html(fig_html, nombre)


def figura_percentiles(percentiles: pd.DataFrame) -> None:
    nombre = "fig10_percentiles_diarios_estacion_imerg_cmorph"
    titulo = f"Percentiles de precipitación diaria ({PERIODO_TEXTO})"
    columnas_percentiles = [f"p{percentil:02d}_mm" for percentil in PERCENTILES]

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    fig_html = go.Figure()
    for columna_fuente in COLUMNAS:
        fuente = FUENTES[columna_fuente]
        fila = percentiles.loc[percentiles["fuente"] == fuente].iloc[0]
        valores = [fila[col] for col in columnas_percentiles]
        ax.plot(PERCENTILES, valores, marker="o", lw=1.8, color=COLORES[columna_fuente], label=fuente)
        fig_html.add_trace(
            go.Scatter(
                x=PERCENTILES,
                y=valores,
                mode="lines+markers",
                name=fuente,
                line=dict(color=COLORES[columna_fuente]),
            )
        )

    ax.set_title(titulo)
    ax.set_xlabel("Percentil")
    ax.set_ylabel("Precipitación diaria (mm/día)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    aplicar_layout_plotly(fig_html, titulo)
    fig_html.update_xaxes(title_text="Percentil")
    fig_html.update_yaxes(title_text="Precipitación diaria (mm/día)")
    guardar_html(fig_html, nombre)


def figura_ecdf(diario: pd.DataFrame) -> None:
    nombre = "fig11_ecdf_precipitacion_diaria_estacion_imerg_cmorph"
    titulo = f"Distribución empírica acumulada de precipitación diaria ({PERIODO_TEXTO})"

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    fig_html = go.Figure()
    for columna in COLUMNAS:
        valores = np.sort(diario[columna].dropna().to_numpy(dtype=float))
        probabilidad = np.arange(1, len(valores) + 1) / len(valores)
        ax.plot(valores, probabilidad, lw=1.8, color=COLORES[columna], label=FUENTES[columna])
        fig_html.add_trace(
            go.Scatter(
                x=valores,
                y=probabilidad,
                mode="lines",
                name=FUENTES[columna],
                line=dict(color=COLORES[columna]),
            )
        )

    ax.set_title(titulo)
    ax.set_xlabel("Precipitación diaria (mm/día)")
    ax.set_ylabel("Probabilidad acumulada")
    ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    aplicar_layout_plotly(fig_html, titulo)
    fig_html.update_xaxes(title_text="Precipitación diaria (mm/día)")
    fig_html.update_yaxes(title_text="Probabilidad acumulada", range=[0, 1.01])
    guardar_html(fig_html, nombre)


def imprimir_resumen(diario: pd.DataFrame, archivos: list[Path]) -> None:
    numero_anios = FIN.year - INICIO.year + 1
    dias_comunes = int(diario[COLUMNAS].notna().all(axis=1).sum())

    print("\nResumen final del análisis 2015–2024")
    print(f"- Periodo usado: {INICIO:%Y}–{FIN:%Y}")
    print(f"- Número de años analizados: {numero_anios}")
    print(f"- Número de días comunes entre estación, IMERG y CMORPH: {dias_comunes:,}")
    print("- Precipitación total por fuente:")
    for columna in COLUMNAS:
        print(f"  {FUENTES[columna]}: {diario[columna].sum(skipna=True):.2f} mm")
    print("- Archivos regenerados:")
    for archivo in archivos:
        print(f"  {archivo}")
    print("- Confirmación: no se crearon archivos duplicados innecesarios; se sobrescribieron los archivos existentes.")


def main() -> None:
    configurar_matplotlib()
    DATOS_LIMPIOS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

    diario = leer_diario()
    leer_mensual()

    anual = calcular_anual(diario)
    momentos = calcular_momentos(diario)
    percentiles = calcular_percentiles(diario)
    l_momentos = calcular_l_momentos(diario)

    anual.to_csv(CSV_ANUAL, index=False)
    momentos.to_csv(CSV_MOMENTOS, index=False)
    percentiles.to_csv(CSV_PERCENTILES, index=False)
    l_momentos.to_csv(CSV_L_MOMENTOS, index=False)

    figura_precipitacion_anual(anual)
    figura_percentiles(percentiles)
    figura_ecdf(diario)

    archivos = [
        CSV_ANUAL,
        CSV_MOMENTOS,
        CSV_PERCENTILES,
        CSV_L_MOMENTOS,
        FIGURAS_DIR / "fig09_precipitacion_anual_estacion_imerg_cmorph.png",
        FIGURAS_DIR / "fig09_precipitacion_anual_estacion_imerg_cmorph.html",
        FIGURAS_DIR / "fig10_percentiles_diarios_estacion_imerg_cmorph.png",
        FIGURAS_DIR / "fig10_percentiles_diarios_estacion_imerg_cmorph.html",
        FIGURAS_DIR / "fig11_ecdf_precipitacion_diaria_estacion_imerg_cmorph.png",
        FIGURAS_DIR / "fig11_ecdf_precipitacion_diaria_estacion_imerg_cmorph.html",
    ]
    imprimir_resumen(diario, archivos)


if __name__ == "__main__":
    main()
