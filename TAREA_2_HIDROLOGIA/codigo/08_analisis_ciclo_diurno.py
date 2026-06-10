from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_PREC_DIR = BASE_DIR / "DATOS_PREC"
DATOS_LIMPIOS_DIR = BASE_DIR / "datos_limpios"
FIGURAS_DIR = BASE_DIR / "figuras"

ESTACION_DIR = next(DATOS_PREC_DIR.glob("ESTACI*"), DATOS_PREC_DIR / "ESTACION")
SATELITE_DIR = next(DATOS_PREC_DIR.glob("SAT*LITE"), DATOS_PREC_DIR / "SATELITE")
IMERG_DIR = SATELITE_DIR / "IMERG"
CMORPH_CSV = SATELITE_DIR / "CMORPH" / "cmorph_champaign_1998_2025.csv"

CSV_CICLO_DIURNO = DATOS_LIMPIOS_DIR / "ciclo_diurno_horario_estacion_imerg_cmorph.csv"

INICIO = pd.Timestamp("2015-01-01")
FIN = pd.Timestamp("2024-12-31")
FIN_EXCLUSIVO = FIN + pd.Timedelta(days=1)
ANIOS = range(INICIO.year, FIN.year + 1)
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

FUENTES = {
    "estacion": "Estación CRNS",
    "imerg": "IMERG",
    "cmorph": "CMORPH",
}
COLORES = {
    "estacion": "#1f77b4",
    "imerg": "#d95f02",
    "cmorph": "#2ca25f",
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


def aplicar_layout_plotly(fig: go.Figure, titulo: str) -> None:
    fig.update_layout(
        title=titulo,
        template="plotly_white",
        font=dict(size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=75, r=35, t=85, b=65),
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


def validar_archivos(rutas: list[Path]) -> None:
    faltantes = [ruta for ruta in rutas if not ruta.exists()]
    if faltantes:
        lista = "\n".join(f"- {ruta}" for ruta in faltantes)
        raise FileNotFoundError(f"No se encontraron estos archivos:\n{lista}")


def construir_fecha_hora(df: pd.DataFrame, col_fecha: str, col_hora: str) -> pd.Series:
    fecha = df[col_fecha].astype("Int64").astype(str).str.zfill(8)
    hora = df[col_hora].astype("Int64").astype(str).str.zfill(4)
    return pd.to_datetime(fecha + hora, format="%Y%m%d%H%M", errors="coerce")


def filtrar_periodo_utc(df: pd.DataFrame, columna_fecha: str) -> pd.DataFrame:
    return df[(df[columna_fecha] >= INICIO) & (df[columna_fecha] < FIN_EXCLUSIVO)].copy()


def agregar_hora_local_desde_utc(df: pd.DataFrame, columna_utc: str) -> pd.DataFrame:
    datos = df.copy()
    utc = datos[columna_utc].dt.tz_localize("UTC")
    datos["fecha_hora_local"] = utc.dt.tz_convert(ZONA_HORARIA_LOCAL).dt.tz_localize(None)
    datos["hora_local"] = datos["fecha_hora_local"].dt.hour
    datos["fecha_local"] = datos["fecha_hora_local"].dt.date
    datos["mes"] = datos["fecha_hora_local"].dt.month
    return datos


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
    estacion["fecha_hora_local"] = construir_fecha_hora(estacion, "lst_date", "lst_time")
    estacion["precip_mm"] = pd.to_numeric(estacion["p_calc_mm"], errors="coerce")
    estacion.loc[estacion["precip_mm"] < 0, "precip_mm"] = np.nan
    estacion = estacion[
        (estacion["fecha_hora_local"] >= INICIO)
        & (estacion["fecha_hora_local"] < FIN_EXCLUSIVO)
    ].copy()

    estacion_30min = (
        estacion.set_index("fecha_hora_local")["precip_mm"]
        .resample("30min", label="left", closed="left")
        .sum(min_count=1)
        .rename("estacion_mm_30min")
        .reset_index()
    )
    estacion_30min["hora_local"] = estacion_30min["fecha_hora_local"].dt.hour
    estacion_30min["fecha_local"] = estacion_30min["fecha_hora_local"].dt.date
    estacion_30min["mes"] = estacion_30min["fecha_hora_local"].dt.month
    return estacion_30min


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
        df = df.rename(columns={columna_precip: "precip_mm_hora"})
        dataframes.append(df)

    imerg = pd.concat(dataframes, ignore_index=True)
    imerg["fecha_hora_utc"] = pd.to_datetime(imerg["time"], errors="coerce")
    imerg["precip_mm_hora"] = pd.to_numeric(imerg["precip_mm_hora"], errors="coerce")
    imerg.loc[imerg["precip_mm_hora"] < 0, "precip_mm_hora"] = np.nan
    imerg["imerg_mm_30min"] = imerg["precip_mm_hora"] * 0.5
    imerg = filtrar_periodo_utc(imerg, "fecha_hora_utc")
    return agregar_hora_local_desde_utc(
        imerg[["fecha_hora_utc", "imerg_mm_30min"]],
        "fecha_hora_utc",
    )


def leer_cmorph_30min() -> pd.DataFrame:
    validar_archivos([CMORPH_CSV])

    cmorph = pd.read_csv(CMORPH_CSV, comment="#", na_values=VALORES_FALTANTES)
    cmorph.columns = [col.strip() for col in cmorph.columns]
    cmorph["fecha_hora_utc"] = pd.to_datetime(cmorph["time"], errors="coerce")
    cmorph["precip_mm_hora"] = pd.to_numeric(
        cmorph["precip_mm_per_hr"],
        errors="coerce",
    )
    cmorph.loc[cmorph["precip_mm_hora"] < 0, "precip_mm_hora"] = np.nan
    cmorph["cmorph_mm_30min"] = cmorph["precip_mm_hora"] * 0.5
    cmorph = filtrar_periodo_utc(cmorph, "fecha_hora_utc")
    return agregar_hora_local_desde_utc(
        cmorph[["fecha_hora_utc", "cmorph_mm_30min"]],
        "fecha_hora_utc",
    )


def resumen_horario(datos: pd.DataFrame, columna: str, prefijo: str) -> pd.DataFrame:
    horario = (
        datos.groupby(["fecha_local", "hora_local"], as_index=False)[columna]
        .sum(min_count=1)
        .rename(columns={columna: f"{prefijo}_mm_hora"})
    )
    agrupado = (
        horario.groupby("hora_local")[f"{prefijo}_mm_hora"]
        .agg(promedio="mean", acumulado="sum")
        .reindex(range(24))
        .reset_index()
    )
    agrupado.columns = [
        "hora_local",
        f"{prefijo}_mm_promedio",
        f"{prefijo}_mm_acumulado",
    ]
    total = agrupado[f"{prefijo}_mm_acumulado"].sum()
    agrupado[f"{prefijo}_porcentaje_diario"] = (
        100 * agrupado[f"{prefijo}_mm_acumulado"] / total if total != 0 else np.nan
    )
    return agrupado


def construir_tabla_ciclo_diurno(
    estacion: pd.DataFrame,
    imerg: pd.DataFrame,
    cmorph: pd.DataFrame,
) -> pd.DataFrame:
    tabla = resumen_horario(estacion, "estacion_mm_30min", "estacion")
    tabla = tabla.merge(resumen_horario(imerg, "imerg_mm_30min", "imerg"), on="hora_local")
    tabla = tabla.merge(resumen_horario(cmorph, "cmorph_mm_30min", "cmorph"), on="hora_local")
    return tabla[
        [
            "hora_local",
            "estacion_mm_promedio",
            "imerg_mm_promedio",
            "cmorph_mm_promedio",
            "estacion_mm_acumulado",
            "imerg_mm_acumulado",
            "cmorph_mm_acumulado",
            "estacion_porcentaje_diario",
            "imerg_porcentaje_diario",
            "cmorph_porcentaje_diario",
        ]
    ]


def construir_matriz_mes_hora_estacion(estacion: pd.DataFrame) -> pd.DataFrame:
    horario = (
        estacion.groupby(["fecha_local", "mes", "hora_local"], as_index=False)[
            "estacion_mm_30min"
        ]
        .sum(min_count=1)
        .rename(columns={"estacion_mm_30min": "estacion_mm_hora"})
    )
    matriz = horario.pivot_table(
        values="estacion_mm_hora",
        index="mes",
        columns="hora_local",
        aggfunc="mean",
    )
    return matriz.reindex(index=range(1, 13), columns=range(24))


def guardar_figura_13(tabla: pd.DataFrame) -> None:
    nombre = "fig13_ciclo_diurno_promedio_estacion_imerg_cmorph"
    titulo = f"Ciclo diurno promedio de la precipitación: estación CRNS, IMERG y CMORPH ({PERIODO_TEXTO})"

    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    for prefijo in FUENTES:
        ax.plot(
            tabla["hora_local"],
            tabla[f"{prefijo}_mm_promedio"],
            marker="o",
            lw=1.8,
            color=COLORES[prefijo],
            label=FUENTES[prefijo],
        )
    ax.set_title(titulo)
    ax.set_xlabel("Hora local")
    ax.set_ylabel("Precipitación promedio por hora local (mm/hora)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    for prefijo in FUENTES:
        fig_html.add_trace(
            go.Scatter(
                x=tabla["hora_local"],
                y=tabla[f"{prefijo}_mm_promedio"],
                mode="lines+markers",
                name=FUENTES[prefijo],
                line=dict(color=COLORES[prefijo]),
            )
        )
    aplicar_layout_plotly(fig_html, titulo)
    fig_html.update_xaxes(title_text="Hora local", dtick=2)
    fig_html.update_yaxes(title_text="Precipitación promedio por hora local (mm/hora)")
    fig_html.write_html(FIGURAS_DIR / f"{nombre}.html", include_plotlyjs="cdn")


def guardar_figura_14(tabla: pd.DataFrame) -> None:
    nombre = "fig14_ciclo_diurno_porcentaje_estacion_imerg_cmorph"
    titulo = f"Distribución porcentual del ciclo diurno de precipitación: estación CRNS, IMERG y CMORPH ({PERIODO_TEXTO})"

    fig, ax = plt.subplots(figsize=(10.2, 5.0))
    for prefijo in FUENTES:
        ax.plot(
            tabla["hora_local"],
            tabla[f"{prefijo}_porcentaje_diario"],
            marker="o",
            lw=1.8,
            color=COLORES[prefijo],
            label=FUENTES[prefijo],
        )
    ax.set_title(titulo)
    ax.set_xlabel("Hora local")
    ax.set_ylabel("Porcentaje de la precipitación diaria total (%)")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    for prefijo in FUENTES:
        fig_html.add_trace(
            go.Scatter(
                x=tabla["hora_local"],
                y=tabla[f"{prefijo}_porcentaje_diario"],
                mode="lines+markers",
                name=FUENTES[prefijo],
                line=dict(color=COLORES[prefijo]),
            )
        )
    aplicar_layout_plotly(fig_html, titulo)
    fig_html.update_xaxes(title_text="Hora local", dtick=2)
    fig_html.update_yaxes(title_text="Porcentaje de la precipitación diaria total (%)")
    fig_html.write_html(FIGURAS_DIR / f"{nombre}.html", include_plotlyjs="cdn")


def guardar_figura_15(matriz: pd.DataFrame) -> None:
    nombre = "fig15_ciclo_diurno_mes_hora_estacion_crns"
    titulo = f"Ciclo diurno por mes de la precipitación observada en estación CRNS ({PERIODO_TEXTO})"

    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    imagen = ax.imshow(matriz.to_numpy(), aspect="auto", cmap="YlGnBu", origin="upper")
    ax.set_title(titulo)
    ax.set_xlabel("Hora local")
    ax.set_ylabel("Mes")
    ax.set_xticks(range(0, 24, 2))
    ax.set_yticks(range(12), MESES)
    cbar = fig.colorbar(imagen, ax=ax)
    cbar.set_label("Precipitación promedio (mm/hora)")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure(
        data=go.Heatmap(
            z=matriz.to_numpy(),
            x=list(range(24)),
            y=MESES,
            colorscale="YlGnBu",
            colorbar=dict(title="mm/hora"),
            hovertemplate="Mes: %{y}<br>Hora local: %{x}:00<br>Promedio: %{z:.3f} mm/hora<extra></extra>",
        )
    )
    fig_html.update_layout(
        title=titulo,
        template="plotly_white",
        font=dict(size=13),
        margin=dict(l=95, r=35, t=85, b=65),
    )
    fig_html.update_xaxes(title_text="Hora local", dtick=2)
    fig_html.update_yaxes(title_text="Mes", autorange="reversed")
    fig_html.write_html(FIGURAS_DIR / f"{nombre}.html", include_plotlyjs="cdn")


def imprimir_resumen(tabla: pd.DataFrame, archivos: list[Path]) -> None:
    pico_estacion = tabla.loc[tabla["estacion_mm_promedio"].idxmax()]
    pico_imerg = tabla.loc[tabla["imerg_mm_promedio"].idxmax()]
    pico_cmorph = tabla.loc[tabla["cmorph_mm_promedio"].idxmax()]
    corr_imerg = tabla["estacion_mm_promedio"].corr(tabla["imerg_mm_promedio"])
    corr_cmorph = tabla["estacion_mm_promedio"].corr(tabla["cmorph_mm_promedio"])
    mejor = "IMERG" if corr_imerg >= corr_cmorph else "CMORPH"

    print("\nResumen del análisis del ciclo diurno")
    print(f"- Periodo usado: {INICIO:%Y-%m-%d} a {FIN:%Y-%m-%d}")
    print(f"- Número de años analizados: {FIN.year - INICIO.year + 1}")
    print(
        f"- Hora pico de la estación CRNS: {int(pico_estacion['hora_local']):02d}:00 "
        f"({pico_estacion['estacion_mm_promedio']:.4f} mm/hora promedio; "
        f"{pico_estacion['estacion_mm_acumulado']:.2f} mm acumulados)."
    )
    print(
        f"- Hora pico de IMERG: {int(pico_imerg['hora_local']):02d}:00 "
        f"({pico_imerg['imerg_mm_promedio']:.4f} mm/hora promedio; "
        f"{pico_imerg['imerg_mm_acumulado']:.2f} mm acumulados)."
    )
    print(
        f"- Hora pico de CMORPH: {int(pico_cmorph['hora_local']):02d}:00 "
        f"({pico_cmorph['cmorph_mm_promedio']:.4f} mm/hora promedio; "
        f"{pico_cmorph['cmorph_mm_acumulado']:.2f} mm acumulados)."
    )
    print(f"- Correlación del ciclo diurno horario estación-IMERG: {corr_imerg:.3f}")
    print(f"- Correlación del ciclo diurno horario estación-CMORPH: {corr_cmorph:.3f}")
    print(
        f"- Interpretación corta: {mejor} representa mejor la forma del ciclo diurno "
        "observado por la estación según la correlación horaria promedio."
    )
    print("- Archivos generados:")
    for archivo in archivos:
        print(f"  {archivo}")
    print("- Confirmación: no se modificaron las figuras 1 a 12 ni los scripts anteriores.")


def main() -> None:
    configurar_matplotlib()
    DATOS_LIMPIOS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

    estacion = leer_estacion_30min()
    imerg = leer_imerg_30min()
    cmorph = leer_cmorph_30min()
    tabla = construir_tabla_ciclo_diurno(estacion, imerg, cmorph)
    matriz_estacion = construir_matriz_mes_hora_estacion(estacion)

    tabla.to_csv(CSV_CICLO_DIURNO, index=False)
    guardar_figura_13(tabla)
    guardar_figura_14(tabla)
    guardar_figura_15(matriz_estacion)

    archivos = [
        CSV_CICLO_DIURNO,
        FIGURAS_DIR / "fig13_ciclo_diurno_promedio_estacion_imerg_cmorph.png",
        FIGURAS_DIR / "fig13_ciclo_diurno_promedio_estacion_imerg_cmorph.html",
        FIGURAS_DIR / "fig14_ciclo_diurno_porcentaje_estacion_imerg_cmorph.png",
        FIGURAS_DIR / "fig14_ciclo_diurno_porcentaje_estacion_imerg_cmorph.html",
        FIGURAS_DIR / "fig15_ciclo_diurno_mes_hora_estacion_crns.png",
        FIGURAS_DIR / "fig15_ciclo_diurno_mes_hora_estacion_crns.html",
    ]
    imprimir_resumen(tabla, archivos)


if __name__ == "__main__":
    main()
