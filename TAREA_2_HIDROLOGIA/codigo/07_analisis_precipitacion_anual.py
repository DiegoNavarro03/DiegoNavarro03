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

INICIO = pd.Timestamp("2022-01-01")
FIN = pd.Timestamp("2024-12-31")

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
    return mensual.sort_values("fecha").reset_index(drop=True)


def calcular_anual(diario: pd.DataFrame) -> pd.DataFrame:
    anual = diario.groupby("anio", as_index=False)[COLUMNAS].sum(min_count=1)
    return anual[["anio", *COLUMNAS]]


def calcular_momentos(diario: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for columna in COLUMNAS:
        serie = diario[columna].dropna()
        dias_lluvia = int((serie > 0).sum())
        filas.append(
            {
                "fuente": FUENTES[columna],
                "media_mm": serie.mean(),
                "varianza_mm2": serie.var(ddof=1),
                "desviacion_estandar_mm": serie.std(ddof=1),
                "asimetria": serie.skew(),
                "curtosis": serie.kurt(),
                "minimo_mm": serie.min(),
                "maximo_mm": serie.max(),
                "numero_dias": int(serie.count()),
                "numero_dias_con_lluvia": dias_lluvia,
                "porcentaje_dias_con_lluvia": 100 * dias_lluvia / serie.count(),
            }
        )
    return pd.DataFrame(filas)


def calcular_percentiles(diario: pd.DataFrame) -> pd.DataFrame:
    percentiles = [50, 75, 90, 95, 99]
    filas = []
    for columna in COLUMNAS:
        serie = diario[columna].dropna()
        fila = {"fuente": FUENTES[columna]}
        for percentil in percentiles:
            fila[f"P{percentil}"] = np.percentile(serie, percentil)
        filas.append(fila)
    return pd.DataFrame(filas)


def calcular_l_momentos_serie(serie: pd.Series) -> dict[str, float]:
    valores = np.sort(serie.dropna().to_numpy(dtype=float))
    valores = valores[valores > 0]
    n = len(valores)
    if n < 4:
        return {
            "L1": np.nan,
            "L2": np.nan,
            "L_CV": np.nan,
            "L_asimetria": np.nan,
            "L_curtosis": np.nan,
            "n_eventos_positivos": n,
        }

    # Implementación sencilla con momentos ponderados por probabilidad.
    # Para datos ordenados x_i, se estiman b0, b1, b2 y b3 con pesos
    # combinatorios muestrales; luego se transforman a L1-L4.
    i = np.arange(n)
    b0 = np.mean(valores)
    b1 = np.mean((i / (n - 1)) * valores)
    b2 = np.mean((i * (i - 1) / ((n - 1) * (n - 2))) * valores)
    b3 = np.mean((i * (i - 1) * (i - 2) / ((n - 1) * (n - 2) * (n - 3))) * valores)

    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0

    return {
        "L1": l1,
        "L2": l2,
        "L_CV": l2 / l1 if l1 != 0 else np.nan,
        "L_asimetria": l3 / l2 if l2 != 0 else np.nan,
        "L_curtosis": l4 / l2 if l2 != 0 else np.nan,
        "n_eventos_positivos": n,
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
    titulo = "Precipitación anual acumulada por fuente"

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for columna in COLUMNAS:
        ax.plot(
            anual["anio"],
            anual[columna],
            marker="o",
            lw=1.9,
            color=COLORES[columna],
            label=FUENTES[columna],
        )
    ax.set_title(titulo)
    ax.set_xlabel("Año")
    ax.set_ylabel("Precipitación anual acumulada (mm/año)")
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
    fig_html.update_layout(
        annotations=[
            dict(
                text="Pregunta: ¿Cómo varía el acumulado anual entre estación CRNS, IMERG y CMORPH?",
                x=0,
                y=-0.22,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
            )
        ]
    )
    fig_html.update_xaxes(title_text="Año", dtick=1)
    fig_html.update_yaxes(title_text="Precipitación anual acumulada (mm/año)")
    guardar_html(fig_html, nombre)


def figura_percentiles(percentiles: pd.DataFrame) -> None:
    nombre = "fig10_percentiles_diarios_estacion_imerg_cmorph"
    titulo = "Percentiles altos de precipitación diaria"
    percentiles_altos = ["P75", "P90", "P95", "P99"]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for _, fila in percentiles.iterrows():
        fuente = fila["fuente"]
        columna_color = next(col for col, etiqueta in FUENTES.items() if etiqueta == fuente)
        ax.plot(
            percentiles_altos,
            [fila[p] for p in percentiles_altos],
            marker="o",
            lw=1.9,
            color=COLORES[columna_color],
            label=fuente,
        )
    ax.set_title(titulo)
    ax.set_xlabel("Percentil")
    ax.set_ylabel("Precipitación diaria (mm/día)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    for _, fila in percentiles.iterrows():
        fuente = fila["fuente"]
        columna_color = next(col for col, etiqueta in FUENTES.items() if etiqueta == fuente)
        fig_html.add_trace(
            go.Scatter(
                x=percentiles_altos,
                y=[fila[p] for p in percentiles_altos],
                mode="lines+markers",
                name=fuente,
                line=dict(color=COLORES[columna_color]),
            )
        )
    aplicar_layout_plotly(fig_html, titulo)
    fig_html.update_layout(
        annotations=[
            dict(
                text="Pregunta: ¿Qué fuente representa mayores intensidades diarias extremas?",
                x=0,
                y=-0.22,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
            )
        ]
    )
    fig_html.update_xaxes(title_text="Percentil")
    fig_html.update_yaxes(title_text="Precipitación diaria (mm/día)")
    guardar_html(fig_html, nombre)


def ecdf(serie: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    valores = np.sort(serie.dropna().to_numpy(dtype=float))
    valores = valores[valores > 0]
    probabilidades = np.arange(1, len(valores) + 1) / len(valores)
    return valores, probabilidades


def figura_ecdf(diario: pd.DataFrame) -> None:
    nombre = "fig11_ecdf_precipitacion_diaria_estacion_imerg_cmorph"
    titulo = "Distribución acumulada empírica de la precipitación diaria positiva"

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    fig_html = go.Figure()
    for columna in COLUMNAS:
        x, y = ecdf(diario[columna])
        ax.plot(x, y, lw=1.9, color=COLORES[columna], label=FUENTES[columna])
        fig_html.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=FUENTES[columna],
                line=dict(color=COLORES[columna]),
            )
        )

    ax.set_title(titulo)
    ax.set_xlabel("Precipitación diaria positiva (mm/día)")
    ax.set_ylabel("Probabilidad acumulada")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    aplicar_layout_plotly(fig_html, titulo)
    fig_html.update_layout(
        annotations=[
            dict(
                text="Pregunta: ¿Cómo se distribuyen los eventos diarios de precipitación en cada fuente?",
                x=0,
                y=-0.22,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
            )
        ]
    )
    fig_html.update_xaxes(title_text="Precipitación diaria positiva (mm/día)")
    fig_html.update_yaxes(title_text="Probabilidad acumulada", range=[0, 1])
    guardar_html(fig_html, nombre)


def imprimir_resumen(
    anual: pd.DataFrame,
    momentos: pd.DataFrame,
    percentiles: pd.DataFrame,
) -> None:
    totales = anual[COLUMNAS].sum()
    mayor_total = FUENTES[totales.idxmax()]
    maximos = momentos.set_index("fuente")["maximo_mm"]
    mayor_maximo = maximos.idxmax()
    p95 = percentiles.set_index("fuente")["P95"]
    p99 = percentiles.set_index("fuente")["P99"]

    print("\nPrecipitación anual por fuente")
    for _, fila in anual.iterrows():
        print(
            f"- {int(fila['anio'])}: Estación CRNS = {fila['estacion_mm']:.2f} mm; "
            f"IMERG = {fila['imerg_mm']:.2f} mm; CMORPH = {fila['cmorph_mm']:.2f} mm."
        )

    print("\nResumen interpretativo")
    print(f"- En el total 2022-2024, la fuente que acumula más precipitación es {mayor_total}.")
    print(f"- El mayor máximo diario corresponde a {mayor_maximo}: {maximos.max():.2f} mm/día.")
    print(f"- El mayor P95 corresponde a {p95.idxmax()}: {p95.max():.2f} mm/día.")
    print(f"- El mayor P99 corresponde a {p99.idxmax()}: {p99.max():.2f} mm/día.")

    print("\nMedia, asimetría y curtosis diaria")
    for _, fila in momentos.iterrows():
        print(
            f"- {fila['fuente']}: media = {fila['media_mm']:.2f} mm/día, "
            f"asimetría = {fila['asimetria']:.2f}, curtosis = {fila['curtosis']:.2f}."
        )

    estacion_total = totales["estacion_mm"]
    for columna in ["imerg_mm", "cmorph_mm"]:
        diferencia = totales[columna] - estacion_total
        porcentaje = 100 * diferencia / estacion_total
        verbo = "sobreestima" if diferencia > 0 else "subestima"
        print(
            f"- {FUENTES[columna]} {verbo} el acumulado total frente a la estación "
            f"en {abs(diferencia):.2f} mm ({abs(porcentaje):.1f}%)."
        )

    print("\nPreguntas respondidas por las figuras")
    print("- Figura 9: ¿Cómo varía el acumulado anual entre estación CRNS, IMERG y CMORPH?")
    print("- Figura 10: ¿Qué fuente representa mayores intensidades diarias extremas?")
    print("- Figura 11: ¿Cómo se distribuyen los eventos diarios de precipitación en cada fuente?")


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

    anual.to_csv(CSV_ANUAL, index=False, encoding="utf-8-sig")
    momentos.to_csv(CSV_MOMENTOS, index=False, encoding="utf-8-sig")
    percentiles.to_csv(CSV_PERCENTILES, index=False, encoding="utf-8-sig")
    l_momentos.to_csv(CSV_L_MOMENTOS, index=False, encoding="utf-8-sig")

    figura_precipitacion_anual(anual)
    figura_percentiles(percentiles)
    figura_ecdf(diario)
    imprimir_resumen(anual, momentos, percentiles)

    print("\nArchivos generados")
    print(f"- {CSV_ANUAL}")
    print(f"- {CSV_MOMENTOS}")
    print(f"- {CSV_PERCENTILES}")
    print(f"- {CSV_L_MOMENTOS}")
    print(f"- {FIGURAS_DIR}")
    print("Nota: los HTML se generan con Plotly y los PNG con Matplotlib; no se requiere Kaleido.")


if __name__ == "__main__":
    main()
