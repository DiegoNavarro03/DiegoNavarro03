from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_LIMPIOS_DIR = BASE_DIR / "datos_limpios"
FIGURAS_DIR = BASE_DIR / "figuras"

CSV_MENSUAL = DATOS_LIMPIOS_DIR / "precipitacion_mensual_estacion_imerg_cmorph.csv"
CSV_ANOMALIAS = DATOS_LIMPIOS_DIR / "anomalias_mensuales_estacion_imerg_cmorph.csv"

INICIO = pd.Timestamp("2015-01-01")
FIN = pd.Timestamp("2024-12-31")
PERIODO_TEXTO = "2015–2024"

COLUMNAS_SERIE_COMPLETA = ["estacion_mm", "imerg_mm", "cmorph_mm"]
COLUMNAS_ANOMALIAS = [
    "estacion_anomalia_mm",
    "imerg_anomalia_mm",
    "cmorph_anomalia_mm",
]
REZAGOS = list(range(1, 13))
REZAGOS_RESUMEN = [1, 2, 3, 6, 12]

ETIQUETAS = {
    "estacion_mm": "Estación CRNS",
    "imerg_mm": "IMERG",
    "cmorph_mm": "CMORPH",
    "estacion_anomalia_mm": "Estación CRNS",
    "imerg_anomalia_mm": "IMERG",
    "cmorph_anomalia_mm": "CMORPH",
}

COLORES = {
    "estacion_mm": "#1f77b4",
    "imerg_mm": "#d95f02",
    "cmorph_mm": "#2ca25f",
    "estacion_anomalia_mm": "#1f77b4",
    "imerg_anomalia_mm": "#d95f02",
    "cmorph_anomalia_mm": "#2ca25f",
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


def leer_mensual() -> pd.DataFrame:
    if not CSV_MENSUAL.exists():
        raise FileNotFoundError(f"No se encontró la tabla mensual: {CSV_MENSUAL}")

    mensual = pd.read_csv(CSV_MENSUAL)
    mensual["fecha"] = pd.to_datetime(mensual["fecha"], format="%Y-%m", errors="coerce")
    mensual[COLUMNAS_SERIE_COMPLETA] = mensual[COLUMNAS_SERIE_COMPLETA].apply(
        pd.to_numeric,
        errors="coerce",
    )
    mensual = mensual[(mensual["fecha"] >= INICIO) & (mensual["fecha"] <= FIN)].copy()
    mensual["mes"] = mensual["fecha"].dt.month
    return mensual.sort_values("fecha").reset_index(drop=True)


def calcular_anomalias(mensual: pd.DataFrame) -> pd.DataFrame:
    climatologia = mensual.groupby("mes")[COLUMNAS_SERIE_COMPLETA].transform("mean")
    anomalias = mensual[["fecha", *COLUMNAS_SERIE_COMPLETA]].copy()
    for columna, columna_anomalia in zip(COLUMNAS_SERIE_COMPLETA, COLUMNAS_ANOMALIAS):
        anomalias[columna_anomalia] = mensual[columna] - climatologia[columna]
        anomalias[f"{columna.replace('_mm', '')}_climatologia_mm"] = climatologia[columna]
    anomalias["fecha"] = anomalias["fecha"].dt.strftime("%Y-%m")
    return anomalias


def preparar_fechas(anomalias: pd.DataFrame) -> pd.DataFrame:
    datos = anomalias.copy()
    datos["fecha"] = pd.to_datetime(datos["fecha"], format="%Y-%m", errors="coerce")
    return datos


def calcular_acf(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    filas = []
    for rezago in REZAGOS:
        fila = {"rezago_meses": rezago}
        for columna in columnas:
            fila[columna] = df[columna].autocorr(lag=rezago)
        filas.append(fila)
    return pd.DataFrame(filas)


def figura_anomalias(anomalias: pd.DataFrame) -> None:
    nombre = "fig06_anomalias_mensuales_estacion_imerg_cmorph"
    titulo = f"Anomalías mensuales de precipitación respecto a la climatología mensual ({PERIODO_TEXTO})"
    datos = preparar_fechas(anomalias)

    fig, ax = plt.subplots(figsize=(12.5, 5.0))
    for columna in COLUMNAS_ANOMALIAS:
        ax.plot(
            datos["fecha"],
            datos[columna],
            marker="o",
            ms=3,
            lw=1.3,
            color=COLORES[columna],
            label=ETIQUETAS[columna],
        )
    ax.axhline(0, color=COLORES["referencia"], lw=1.1, ls="--")
    ax.set_title(titulo)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Anomalía mensual (mm)")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    for columna in COLUMNAS_ANOMALIAS:
        fig_html.add_trace(
            go.Scatter(
                x=datos["fecha"],
                y=datos[columna],
                mode="lines+markers",
                name=ETIQUETAS[columna],
                line=dict(color=COLORES[columna]),
            )
        )
    fig_html.add_hline(y=0, line_dash="dash", line_color=COLORES["referencia"])
    aplicar_layout_plotly(fig_html, titulo)
    fig_html.update_xaxes(title_text="Mes")
    fig_html.update_yaxes(title_text="Anomalía mensual (mm)")
    guardar_html(fig_html, nombre)


def graficar_acf(
    acf: pd.DataFrame,
    columnas: list[str],
    nombre: str,
    titulo: str,
    pregunta: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    for columna in columnas:
        ax.plot(
            acf["rezago_meses"],
            acf[columna],
            marker="o",
            lw=1.8,
            color=COLORES[columna],
            label=ETIQUETAS[columna],
        )

    ax.axhline(0, color=COLORES["referencia"], lw=1.0, ls="--")
    ax.set_title(titulo)
    ax.set_xlabel("Rezago (meses)")
    ax.set_ylabel("Autocorrelación")
    ax.set_xticks(REZAGOS)
    ax.set_ylim(-1, 1)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    for columna in columnas:
        fig_html.add_trace(
            go.Scatter(
                x=acf["rezago_meses"],
                y=acf[columna],
                mode="lines+markers",
                name=ETIQUETAS[columna],
                line=dict(color=COLORES[columna]),
            )
        )
    fig_html.add_hline(y=0, line_dash="dash", line_color=COLORES["referencia"])
    aplicar_layout_plotly(fig_html, titulo)
    fig_html.update_layout(
        annotations=[
            dict(
                text=pregunta,
                x=0,
                y=-0.22,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                font=dict(size=12),
            )
        ]
    )
    fig_html.update_xaxes(title_text="Rezago (meses)", dtick=1)
    fig_html.update_yaxes(title_text="Autocorrelación", range=[-1, 1])
    guardar_html(fig_html, nombre)


def imprimir_acf(tabla: pd.DataFrame, columnas: list[str], titulo: str) -> None:
    print(f"\n{titulo}")
    resumen = tabla[tabla["rezago_meses"].isin(REZAGOS_RESUMEN)]
    for _, fila in resumen.iterrows():
        valores = [f"{ETIQUETAS[columna]} = {fila[columna]:.3f}" for columna in columnas]
        print(f"- Rezago {int(fila['rezago_meses'])} meses: " + "; ".join(valores))


def interpretar_memoria(acf_original: pd.DataFrame, acf_anomalias: pd.DataFrame) -> None:
    memoria_original = acf_original.loc[
        acf_original["rezago_meses"].isin([1, 2, 3]),
        COLUMNAS_SERIE_COMPLETA,
    ].abs().mean().mean()
    memoria_anomalias = acf_anomalias.loc[
        acf_anomalias["rezago_meses"].isin([1, 2, 3]),
        COLUMNAS_ANOMALIAS,
    ].abs().mean().mean()
    diferencia = memoria_anomalias - memoria_original

    print("\nComparación entre serie completa y anomalías")
    print(
        "- Persistencia media absoluta en rezagos 1-3 meses de la serie completa: "
        f"{memoria_original:.3f}."
    )
    print(
        "- Persistencia media absoluta en rezagos 1-3 meses de las anomalías: "
        f"{memoria_anomalias:.3f}."
    )

    print("\nInterpretación corta")
    if diferencia < 0:
        print(
            "- La memoria aparente disminuye al eliminar el ciclo anual. Esto indica "
            "que parte de la persistencia de la serie mensual estaba asociada a la "
            "estacionalidad."
        )
    else:
        print(
            "- La memoria aparente no disminuye al eliminar el ciclo anual. En este "
            "periodo, las anomalías conservan una autocorrelación media de corto "
            "rezago igual o mayor que la serie completa."
        )
    print(
        "- Advertencia: el periodo común 2015–2024 tiene 120 meses; la interpretación "
        "es más estable que en 2022–2024, pero las autocorrelaciones siguen siendo "
        "estimaciones muestrales."
    )


def imprimir_resumen_archivos() -> None:
    archivos = [
        CSV_ANOMALIAS,
        FIGURAS_DIR / "fig06_anomalias_mensuales_estacion_imerg_cmorph.png",
        FIGURAS_DIR / "fig06_anomalias_mensuales_estacion_imerg_cmorph.html",
        FIGURAS_DIR / "fig07_acf_serie_mensual_estacion_imerg_cmorph.png",
        FIGURAS_DIR / "fig07_acf_serie_mensual_estacion_imerg_cmorph.html",
        FIGURAS_DIR / "fig08_acf_anomalias_mensuales_estacion_imerg_cmorph.png",
        FIGURAS_DIR / "fig08_acf_anomalias_mensuales_estacion_imerg_cmorph.html",
    ]
    print("\nArchivos regenerados")
    for archivo in archivos:
        print(f"- {archivo}")
    print("- Confirmación: no se crearon archivos duplicados innecesarios; se sobrescribieron las salidas existentes.")


def main() -> None:
    configurar_matplotlib()
    DATOS_LIMPIOS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

    mensual = leer_mensual()
    anomalias = calcular_anomalias(mensual)
    anomalias.to_csv(CSV_ANOMALIAS, index=False)

    datos_anomalias = preparar_fechas(anomalias)
    acf_original = calcular_acf(datos_anomalias, COLUMNAS_SERIE_COMPLETA)
    acf_anomalias = calcular_acf(datos_anomalias, COLUMNAS_ANOMALIAS)

    figura_anomalias(anomalias)
    graficar_acf(
        acf_original,
        COLUMNAS_SERIE_COMPLETA,
        "fig07_acf_serie_mensual_estacion_imerg_cmorph",
        f"Autocorrelograma de la precipitación mensual original ({PERIODO_TEXTO})",
        "Pregunta: ¿La serie mensual original muestra persistencia temporal aparente?",
    )
    graficar_acf(
        acf_anomalias,
        COLUMNAS_ANOMALIAS,
        "fig08_acf_anomalias_mensuales_estacion_imerg_cmorph",
        f"Autocorrelograma de las anomalías mensuales de precipitación ({PERIODO_TEXTO})",
        "Pregunta: ¿La precipitación mantiene memoria después de eliminar el ciclo anual?",
    )

    imprimir_acf(
        acf_original,
        COLUMNAS_SERIE_COMPLETA,
        "Autocorrelaciones de la serie mensual completa",
    )
    imprimir_acf(
        acf_anomalias,
        COLUMNAS_ANOMALIAS,
        "Autocorrelaciones de las anomalías mensuales",
    )
    interpretar_memoria(acf_original, acf_anomalias)
    imprimir_resumen_archivos()


if __name__ == "__main__":
    main()
