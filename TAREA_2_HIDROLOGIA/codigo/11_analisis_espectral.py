from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_LIMPIOS_DIR = BASE_DIR / "datos_limpios"
FIGURAS_DIR = BASE_DIR / "figuras"
HTML_DIR = BASE_DIR / "html"

CSV_DIARIO = DATOS_LIMPIOS_DIR / "precipitacion_diaria_estacion_imerg_cmorph.csv"
CSV_PICOS = DATOS_LIMPIOS_DIR / "analisis_espectral_picos.csv"

FIGURA_NOMBRE = "fig21_analisis_espectral_precipitacion"

INICIO = pd.Timestamp("2015-01-01")
FIN = pd.Timestamp("2024-12-31")
PERIODO_TEXTO = "2015-2024"
DIAS_POR_MES = 365.25 / 12
PERIODOS_REFERENCIA = {
    365: "Ciclo anual",
    180: "Ciclo semestral",
    30: "Escala mensual",
}

COLUMNAS = ["estacion_mm", "imerg_mm", "cmorph_mm"]
FUENTES = {
    "estacion_mm": "Estacion CRNS",
    "imerg_mm": "IMERG",
    "cmorph_mm": "CMORPH",
}
COLORES = {
    "estacion_mm": "#1f77b4",
    "imerg_mm": "#d95f02",
    "cmorph_mm": "#2ca25f",
    "referencia": "#4d4d4d",
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


def preparar_directorios() -> None:
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    HTML_DIR.mkdir(parents=True, exist_ok=True)


def leer_diario() -> pd.DataFrame:
    if not CSV_DIARIO.exists():
        raise FileNotFoundError(f"No se encontro el archivo diario: {CSV_DIARIO}")

    diario = pd.read_csv(CSV_DIARIO)
    columnas_requeridas = {"fecha", *COLUMNAS}
    faltantes = columnas_requeridas.difference(diario.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el CSV diario: {sorted(faltantes)}")

    diario["fecha"] = pd.to_datetime(diario["fecha"], errors="coerce")
    diario = diario.dropna(subset=["fecha"]).copy()
    diario[COLUMNAS] = diario[COLUMNAS].apply(pd.to_numeric, errors="coerce")
    diario = diario[(diario["fecha"] >= INICIO) & (diario["fecha"] <= FIN)].copy()
    diario = diario.sort_values("fecha").drop_duplicates(subset="fecha", keep="last")

    calendario = pd.date_range(INICIO, FIN, freq="D")
    diario = diario.set_index("fecha").reindex(calendario)
    diario.index.name = "fecha"

    print("Manejo de valores faltantes:")
    print(f"- Se reindexo la serie al calendario diario completo {PERIODO_TEXTO}.")
    for columna in COLUMNAS:
        n_faltantes = int(diario[columna].isna().sum())
        diario[columna] = diario[columna].fillna(0.0)
        print(
            f"- {FUENTES[columna]}: {n_faltantes} dias sin dato o valores invalidos "
            "se rellenaron con 0.0 mm."
        )

    return diario.reset_index()


def calcular_promedio_mensual(diario: pd.DataFrame) -> pd.DataFrame:
    mensual = (
        diario.set_index("fecha")
        .resample("MS")[COLUMNAS]
        .mean()
        .reset_index()
    )
    print("\nAgregacion temporal:")
    print(
        "- Para reducir la saturacion del espectro diario, se calculo el "
        "promedio mensual de precipitacion diaria antes de aplicar la FFT."
    )
    print(
        "- La frecuencia mensual permite interpretar ciclos de 2 meses o mas; "
        "la referencia de 30 dias queda como escala orientativa, no como pico resoluble."
    )
    return mensual


def calcular_espectro(serie: pd.Series) -> pd.DataFrame:
    valores = serie.to_numpy(dtype=float)
    valores = valores - np.nanmean(valores)
    n = len(valores)

    frecuencias = np.fft.rfftfreq(n, d=DIAS_POR_MES)
    transformada = np.fft.rfft(valores)
    potencia = (np.abs(transformada) ** 2) / n

    espectro = pd.DataFrame(
        {
            "frecuencia_ciclos_dia": frecuencias,
            "periodo_dias": np.divide(
                1.0,
                frecuencias,
                out=np.full_like(frecuencias, np.nan, dtype=float),
                where=frecuencias > 0,
            ),
            "potencia": potencia,
        }
    )
    espectro = espectro[espectro["frecuencia_ciclos_dia"] > 0].copy()
    espectro = espectro[np.isfinite(espectro["periodo_dias"])].copy()
    return espectro.sort_values("periodo_dias").reset_index(drop=True)


def seleccionar_picos(espectro: pd.DataFrame, fuente: str, n_picos: int = 5) -> pd.DataFrame:
    candidatos = espectro[
        (espectro["periodo_dias"] >= 2.0 * DIAS_POR_MES)
        & (espectro["periodo_dias"] <= 365.25 * 5)
    ].sort_values("potencia", ascending=False)

    seleccionados = []
    for _, fila in candidatos.iterrows():
        periodo = float(fila["periodo_dias"])
        if any(abs(periodo - previo["periodo_dias"]) / periodo < 0.05 for previo in seleccionados):
            continue
        seleccionados.append(
            {
                "fuente": fuente,
                "ranking": len(seleccionados) + 1,
                "periodo_dias": periodo,
                "frecuencia_ciclos_dia": float(fila["frecuencia_ciclos_dia"]),
                "potencia": float(fila["potencia"]),
            }
        )
        if len(seleccionados) == n_picos:
            break

    return pd.DataFrame(seleccionados)


def graficar_espectros(espectros: dict[str, pd.DataFrame]) -> None:
    titulo = f"Analisis espectral de Fourier del promedio mensual de precipitacion ({PERIODO_TEXTO})"

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    for columna, espectro in espectros.items():
        ax.plot(
            espectro["periodo_dias"],
            espectro["potencia"],
            lw=1.3,
            color=COLORES[columna],
            label=FUENTES[columna],
        )

    for periodo, etiqueta in PERIODOS_REFERENCIA.items():
        ax.axvline(periodo, color=COLORES["referencia"], lw=1.0, ls="--", alpha=0.75)
        ax.text(
            periodo,
            ax.get_ylim()[1] * 0.92,
            f"{etiqueta}\n{periodo} d",
            rotation=90,
            va="top",
            ha="right",
            fontsize=8.5,
            color=COLORES["referencia"],
        )

    ax.set_xscale("log")
    ax.set_xlim(30, 2000)
    ax.set_title(titulo)
    ax.set_xlabel("Periodo (dias, escala logaritmica)")
    ax.set_ylabel("Potencia espectral")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{FIGURA_NOMBRE}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    for columna, espectro in espectros.items():
        fig_html.add_trace(
            go.Scatter(
                x=espectro["periodo_dias"],
                y=espectro["potencia"],
                mode="lines",
                name=FUENTES[columna],
                line=dict(color=COLORES[columna]),
            )
        )

    for periodo, etiqueta in PERIODOS_REFERENCIA.items():
        fig_html.add_vline(
            x=periodo,
            line_dash="dash",
            line_color=COLORES["referencia"],
            annotation_text=f"{etiqueta}: {periodo} d",
            annotation_position="top",
        )

    fig_html.update_layout(
        title=titulo,
        template="plotly_white",
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=75, r=35, t=85, b=65),
    )
    fig_html.update_xaxes(
        title_text="Periodo (dias, escala logaritmica)",
        type="log",
        range=[np.log10(30), np.log10(2000)],
    )
    fig_html.update_yaxes(title_text="Potencia espectral")
    fig_html.write_html(HTML_DIR / f"{FIGURA_NOMBRE}.html", include_plotlyjs="cdn")


def hay_senal_anual(espectro: pd.DataFrame) -> bool:
    ventana_anual = espectro[
        (espectro["periodo_dias"] >= 330)
        & (espectro["periodo_dias"] <= 400)
    ]
    if ventana_anual.empty:
        return False

    umbral_percentil = espectro["potencia"].quantile(0.95)
    return bool(ventana_anual["potencia"].max() >= umbral_percentil)


def imprimir_interpretacion(picos: pd.DataFrame, espectros: dict[str, pd.DataFrame]) -> None:
    print("\nInterpretacion corta del analisis espectral:")
    for columna in COLUMNAS:
        fuente = FUENTES[columna]
        dominante = picos[(picos["fuente"] == fuente) & (picos["ranking"] == 1)].iloc[0]
        print(f"- Periodo dominante de {fuente}: {dominante['periodo_dias']:.1f} dias.")

    fuentes_anuales = [
        FUENTES[columna]
        for columna, espectro in espectros.items()
        if hay_senal_anual(espectro)
    ]
    if fuentes_anuales:
        print(
            "- Aparece senal cercana al ciclo anual en: "
            + ", ".join(fuentes_anuales)
            + "."
        )
    else:
        print("- No se detecta una senal anual entre las potencias mas altas del espectro.")


def main() -> None:
    configurar_matplotlib()
    preparar_directorios()
    diario = leer_diario()
    mensual = calcular_promedio_mensual(diario)

    espectros = {}
    tablas_picos = []
    for columna in COLUMNAS:
        espectro = calcular_espectro(mensual[columna])
        espectros[columna] = espectro
        tablas_picos.append(seleccionar_picos(espectro, FUENTES[columna]))

    picos = pd.concat(tablas_picos, ignore_index=True)
    picos.to_csv(CSV_PICOS, index=False)
    graficar_espectros(espectros)

    print(f"\nFigura PNG guardada en: {FIGURAS_DIR / (FIGURA_NOMBRE + '.png')}")
    print(f"Figura HTML guardada en: {HTML_DIR / (FIGURA_NOMBRE + '.html')}")
    print(f"Tabla de picos guardada en: {CSV_PICOS}")
    imprimir_interpretacion(picos, espectros)


if __name__ == "__main__":
    main()
