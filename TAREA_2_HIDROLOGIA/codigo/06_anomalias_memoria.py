from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_LIMPIOS_DIR = BASE_DIR / "datos_limpios"
FIGURAS_DIR = BASE_DIR / "figuras"

CSV_ANOMALIAS = DATOS_LIMPIOS_DIR / "anomalias_mensuales_estacion_imerg_cmorph.csv"

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


def leer_anomalias() -> pd.DataFrame:
    if not CSV_ANOMALIAS.exists():
        raise FileNotFoundError(f"No se encontró la tabla de anomalías: {CSV_ANOMALIAS}")

    datos = pd.read_csv(CSV_ANOMALIAS)
    datos["fecha"] = pd.to_datetime(datos["fecha"], errors="coerce")
    columnas_requeridas = ["fecha", *COLUMNAS_SERIE_COMPLETA, *COLUMNAS_ANOMALIAS]
    faltantes = [columna for columna in columnas_requeridas if columna not in datos.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en la tabla de anomalías: {faltantes}")

    datos[[*COLUMNAS_SERIE_COMPLETA, *COLUMNAS_ANOMALIAS]] = datos[
        [*COLUMNAS_SERIE_COMPLETA, *COLUMNAS_ANOMALIAS]
    ].apply(pd.to_numeric, errors="coerce")
    return datos.sort_values("fecha").reset_index(drop=True)


def calcular_acf(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    filas = []
    for rezago in REZAGOS:
        fila = {"rezago_meses": rezago}
        for columna in columnas:
            fila[columna] = df[columna].autocorr(lag=rezago)
        filas.append(fila)
    return pd.DataFrame(filas)


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
            "periodo corto, las anomalías conservan una autocorrelación media de corto "
            "rezago igual o mayor que la serie completa."
        )
    print(
        "- Advertencia: el periodo común 2022-2024 tiene solo 36 meses, por lo que "
        "las autocorrelaciones y la interpretación de memoria deben tomarse con cautela."
    )


def main() -> None:
    configurar_matplotlib()
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

    anomalias = leer_anomalias()
    acf_original = calcular_acf(anomalias, COLUMNAS_SERIE_COMPLETA)
    acf_anomalias = calcular_acf(anomalias, COLUMNAS_ANOMALIAS)

    graficar_acf(
        acf_original,
        COLUMNAS_SERIE_COMPLETA,
        "fig07_acf_serie_mensual_estacion_imerg_cmorph",
        "Autocorrelograma de la precipitación mensual original",
        "Pregunta: ¿La serie mensual original muestra persistencia temporal aparente?",
    )
    graficar_acf(
        acf_anomalias,
        COLUMNAS_ANOMALIAS,
        "fig08_acf_anomalias_mensuales_estacion_imerg_cmorph",
        "Autocorrelograma de las anomalías mensuales de precipitación",
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

    print("\nArchivos generados")
    print(f"- {FIGURAS_DIR / 'fig07_acf_serie_mensual_estacion_imerg_cmorph.png'}")
    print(f"- {FIGURAS_DIR / 'fig07_acf_serie_mensual_estacion_imerg_cmorph.html'}")
    print(f"- {FIGURAS_DIR / 'fig08_acf_anomalias_mensuales_estacion_imerg_cmorph.png'}")
    print(f"- {FIGURAS_DIR / 'fig08_acf_anomalias_mensuales_estacion_imerg_cmorph.html'}")
    print("Nota: los HTML se generan con Plotly y los PNG con Matplotlib; no se requiere Kaleido.")


if __name__ == "__main__":
    main()
