from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_LIMPIOS_DIR = BASE_DIR / "datos_limpios"
FIGURAS_DIR = BASE_DIR / "figuras"

CSV_MENSUAL = DATOS_LIMPIOS_DIR / "precipitacion_mensual_estacion_imerg_cmorph.csv"
CSV_CLIMATOLOGIA = DATOS_LIMPIOS_DIR / "climatologia_mensual_estacion_imerg_cmorph.csv"

NOMBRE_FIGURA = "fig12_climatologia_mensual_estacion_imerg_cmorph"
INICIO = pd.Timestamp("2015-01-01")
FIN = pd.Timestamp("2024-12-31")
TITULO = "Climatología mensual promedio de precipitación: estación CRNS, IMERG y CMORPH (2015–2024)"

COLUMNAS = ["estacion_mm", "imerg_mm", "cmorph_mm"]
ETIQUETAS = {
    "estacion_mm": "Estación CRNS",
    "imerg_mm": "IMERG",
    "cmorph_mm": "CMORPH",
}
COLORES = {
    "estacion_mm": "#1f77b4",
    "imerg_mm": "#d95f02",
    "cmorph_mm": "#2ca25f",
}
MESES = [
    "Enero",
    "Febrero",
    "Marzo",
    "Abril",
    "Mayo",
    "Junio",
    "Julio",
    "Agosto",
    "Septiembre",
    "Octubre",
    "Noviembre",
    "Diciembre",
]


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


def leer_mensual() -> pd.DataFrame:
    if not CSV_MENSUAL.exists():
        raise FileNotFoundError(f"No se encontró el archivo mensual: {CSV_MENSUAL}")

    mensual = pd.read_csv(CSV_MENSUAL)
    mensual["fecha"] = pd.to_datetime(mensual["fecha"], format="%Y-%m", errors="coerce")
    mensual[COLUMNAS] = mensual[COLUMNAS].apply(pd.to_numeric, errors="coerce")
    mensual = mensual[(mensual["fecha"] >= INICIO) & (mensual["fecha"] <= FIN)].copy()
    mensual["mes"] = mensual["fecha"].dt.month
    return mensual


def calcular_climatologia(mensual: pd.DataFrame) -> pd.DataFrame:
    climatologia = mensual.groupby("mes", as_index=False)[COLUMNAS].mean()
    return climatologia[["mes", *COLUMNAS]]


def guardar_figuras(climatologia: pd.DataFrame) -> None:
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    for columna in COLUMNAS:
        ax.plot(
            climatologia["mes"],
            climatologia[columna],
            marker="o",
            lw=1.9,
            color=COLORES[columna],
            label=ETIQUETAS[columna],
        )
    ax.set_title(TITULO)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Precipitación mensual promedio (mm/mes)")
    ax.set_xticks(range(1, 13), MESES, rotation=35, ha="right")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{NOMBRE_FIGURA}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    for columna in COLUMNAS:
        fig_html.add_trace(
            go.Scatter(
                x=MESES,
                y=climatologia[columna],
                mode="lines+markers",
                name=ETIQUETAS[columna],
                line=dict(color=COLORES[columna]),
            )
        )
    fig_html.update_layout(
        title=TITULO,
        template="plotly_white",
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=75, r=35, t=85, b=65),
    )
    fig_html.update_xaxes(title_text="Mes")
    fig_html.update_yaxes(title_text="Precipitación mensual promedio (mm/mes)")
    fig_html.write_html(FIGURAS_DIR / f"{NOMBRE_FIGURA}.html", include_plotlyjs="cdn")


def imprimir_resumen(climatologia: pd.DataFrame) -> None:
    print("\nResumen de climatología mensual promedio 2015–2024")
    for columna in COLUMNAS:
        lluvioso = climatologia.loc[climatologia[columna].idxmax()]
        seco = climatologia.loc[climatologia[columna].idxmin()]
        print(
            f"- Mes más lluvioso promedio para {ETIQUETAS[columna]}: "
            f"{MESES[int(lluvioso['mes']) - 1]} ({lluvioso[columna]:.2f} mm/mes)."
        )
        print(
            f"- Mes más seco promedio para {ETIQUETAS[columna]}: "
            f"{MESES[int(seco['mes']) - 1]} ({seco[columna]:.2f} mm/mes)."
        )

    corr_imerg = climatologia["estacion_mm"].corr(climatologia["imerg_mm"])
    corr_cmorph = climatologia["estacion_mm"].corr(climatologia["cmorph_mm"])
    print(f"- Correlación de la climatología mensual estación-IMERG: {corr_imerg:.3f}")
    print(f"- Correlación de la climatología mensual estación-CMORPH: {corr_cmorph:.3f}")
    print("\nArchivos creados")
    print(f"- {CSV_CLIMATOLOGIA}")
    print(f"- {FIGURAS_DIR / f'{NOMBRE_FIGURA}.png'}")
    print(f"- {FIGURAS_DIR / f'{NOMBRE_FIGURA}.html'}")
    print("- Confirmación: no se modificaron las figuras 1 a 11.")


def main() -> None:
    configurar_matplotlib()
    DATOS_LIMPIOS_DIR.mkdir(parents=True, exist_ok=True)

    mensual = leer_mensual()
    climatologia = calcular_climatologia(mensual)
    climatologia.to_csv(CSV_CLIMATOLOGIA, index=False)
    guardar_figuras(climatologia)
    imprimir_resumen(climatologia)


if __name__ == "__main__":
    main()
