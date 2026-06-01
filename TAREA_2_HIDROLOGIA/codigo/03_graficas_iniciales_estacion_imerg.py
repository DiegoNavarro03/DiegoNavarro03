from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go


BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_PREC_DIR = BASE_DIR / "DATOS_PREC"
DATOS_LIMPIOS_DIR = BASE_DIR / "datos_limpios"
FIGURAS_DIR = BASE_DIR / "figuras"

CSV_DIARIO = DATOS_LIMPIOS_DIR / "precipitacion_diaria_estacion_imerg.csv"
ESTACION_DIR = next(DATOS_PREC_DIR.glob("ESTACI*"), DATOS_PREC_DIR / "ESTACION")
IMERG_DIR = next(DATOS_PREC_DIR.glob("SAT*LITE"), DATOS_PREC_DIR / "SATELITE") / "IMERG"

ARCHIVOS_ESTACION = [
    "CRNS0101-05-2022-IL_Champaign_9_SW.txt",
    "CRNS0101-05-2023-IL_Champaign_9_SW.txt",
    "CRNS0101-05-2024-IL_Champaign_9_SW.txt",
]

ARCHIVOS_IMERG = [
    "imerg_champaign_2022.csv",
    "imerg_champaign_2023.csv",
    "imerg_champaign_2024.csv",
]

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
COLOR_ESTACION = "#1f77b4"
COLOR_IMERG = "#d95f02"
COLOR_REFERENCIA = "#4d4d4d"


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
        margin=dict(l=70, r=35, t=85, b=65),
    )


def guardar_html(fig: go.Figure, nombre: str) -> None:
    fig.write_html(FIGURAS_DIR / f"{nombre}.html", include_plotlyjs="cdn")


def validar_archivos(rutas: list[Path]) -> None:
    faltantes = [ruta for ruta in rutas if not ruta.exists()]
    if faltantes:
        lista = "\n".join(f"- {ruta}" for ruta in faltantes)
        raise FileNotFoundError(f"No se encontraron estos archivos:\n{lista}")


def leer_diario() -> pd.DataFrame:
    if not CSV_DIARIO.exists():
        raise FileNotFoundError(f"No se encontro el CSV diario: {CSV_DIARIO}")
    diario = pd.read_csv(CSV_DIARIO, parse_dates=["fecha"])
    columnas = ["precip_estacion_mm_dia", "precip_imerg_mm_dia"]
    diario[columnas] = diario[columnas].apply(pd.to_numeric, errors="coerce")
    return diario.sort_values("fecha").reset_index(drop=True)


def leer_estacion_crns() -> pd.DataFrame:
    rutas = [ESTACION_DIR / nombre for nombre in ARCHIVOS_ESTACION]
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
        df.insert(0, "archivo", ruta.name)
        dataframes.append(df)

    estacion = pd.concat(dataframes, ignore_index=True)
    estacion["fecha_hora_utc"] = construir_fecha_hora(estacion, "utc_date", "utc_time")
    estacion["fecha_hora_local"] = construir_fecha_hora(estacion, "lst_date", "lst_time")
    estacion["precip_estacion_mm_5min"] = pd.to_numeric(estacion["p_calc_mm"], errors="coerce")
    estacion.loc[estacion["precip_estacion_mm_5min"] < 0, "precip_estacion_mm_5min"] = np.nan
    return estacion.sort_values("fecha_hora_utc").reset_index(drop=True)


def construir_fecha_hora(df: pd.DataFrame, col_fecha: str, col_hora: str) -> pd.Series:
    fecha = df[col_fecha].astype("Int64").astype(str).str.zfill(8)
    hora = df[col_hora].astype("Int64").astype(str).str.zfill(4)
    return pd.to_datetime(fecha + hora, format="%Y%m%d%H%M", errors="coerce")


def leer_imerg() -> pd.DataFrame:
    rutas = [IMERG_DIR / nombre for nombre in ARCHIVOS_IMERG]
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
        df = df.rename(columns={columna_precip: "precip_imerg_mm_h"})
        df["archivo"] = ruta.name
        dataframes.append(df)

    imerg = pd.concat(dataframes, ignore_index=True)
    imerg["fecha_hora_utc"] = pd.to_datetime(imerg["time"], errors="coerce")
    imerg["precip_imerg_mm_h"] = pd.to_numeric(imerg["precip_imerg_mm_h"], errors="coerce")
    imerg.loc[imerg["precip_imerg_mm_h"] < 0, "precip_imerg_mm_h"] = np.nan
    imerg["precip_imerg_mm_30min"] = imerg["precip_imerg_mm_h"] * 0.5
    return imerg.sort_values("fecha_hora_utc").reset_index(drop=True)


def agregar_estacion_30min(estacion: pd.DataFrame) -> pd.DataFrame:
    return (
        estacion.set_index("fecha_hora_utc")["precip_estacion_mm_5min"]
        .resample("30min", label="left", closed="left")
        .sum(min_count=1)
        .rename("precip_estacion_mm_30min")
        .reset_index()
    )


def calcular_metricas_diarias(diario: pd.DataFrame) -> dict[str, float]:
    pares = diario[["precip_estacion_mm_dia", "precip_imerg_mm_dia"]].dropna()
    obs = pares["precip_estacion_mm_dia"]
    sim = pares["precip_imerg_mm_dia"]
    diferencia = sim - obs
    return {
        "n": float(len(pares)),
        "correlacion": float(obs.corr(sim)),
        "rmse": float(np.sqrt(np.mean(diferencia**2))),
        "sesgo_medio": float(diferencia.mean()),
        "pbias": float(100 * diferencia.sum() / obs.sum()) if obs.sum() != 0 else np.nan,
    }


def figura_serie_diaria(diario: pd.DataFrame) -> None:
    nombre = "fig01_serie_diaria_estacion_imerg"
    titulo = "Serie diaria de precipitacion: estacion CRNS vs IMERG (2022-2024)"

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(diario["fecha"], diario["precip_estacion_mm_dia"], color=COLOR_ESTACION, lw=1.1, label="Estacion CRNS")
    ax.plot(diario["fecha"], diario["precip_imerg_mm_dia"], color=COLOR_IMERG, lw=1.1, alpha=0.85, label="IMERG")
    ax.set_title(titulo)
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precipitacion diaria (mm/dia)")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    fig_html.add_trace(go.Scatter(x=diario["fecha"], y=diario["precip_estacion_mm_dia"], mode="lines", name="Estacion CRNS", line=dict(color=COLOR_ESTACION)))
    fig_html.add_trace(go.Scatter(x=diario["fecha"], y=diario["precip_imerg_mm_dia"], mode="lines", name="IMERG", line=dict(color=COLOR_IMERG)))
    aplicar_layout_plotly(fig_html, titulo)
    fig_html.update_xaxes(title_text="Fecha")
    fig_html.update_yaxes(title_text="Precipitacion diaria (mm/dia)")
    guardar_html(fig_html, nombre)

    mayor = diario.loc[diario["precip_estacion_mm_dia"].idxmax()]
    corr = diario["precip_estacion_mm_dia"].corr(diario["precip_imerg_mm_dia"])
    print("\nFigura 1 - Serie diaria completa")
    print("- Pregunta: como varia la precipitacion diaria y cuanto sigue IMERG a la estacion.")
    print(f"- Patron principal: el mayor evento observado fue {mayor['precip_estacion_mm_dia']:.2f} mm el {mayor['fecha'].date()}.")
    print(f"- Metrica: correlacion diaria estacion-IMERG = {corr:.3f}.")


def figura_acumulado_mensual(diario: pd.DataFrame) -> pd.DataFrame:
    nombre = "fig02_acumulado_mensual_estacion_imerg"
    titulo = "Acumulado mensual de precipitacion: estacion CRNS vs IMERG"

    mensual = diario.copy()
    mensual["mes"] = mensual["fecha"].dt.to_period("M").dt.to_timestamp()
    mensual = (
        mensual.groupby("mes", as_index=False)[["precip_estacion_mm_dia", "precip_imerg_mm_dia"]]
        .sum(min_count=1)
        .sort_values("mes")
    )

    x = np.arange(len(mensual))
    ancho = 0.42
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.bar(x - ancho / 2, mensual["precip_estacion_mm_dia"], width=ancho, color=COLOR_ESTACION, label="Estacion CRNS")
    ax.bar(x + ancho / 2, mensual["precip_imerg_mm_dia"], width=ancho, color=COLOR_IMERG, label="IMERG")
    ax.set_title(titulo)
    ax.set_xlabel("Mes")
    ax.set_ylabel("Precipitacion mensual (mm/mes)")
    ax.set_xticks(x[::2])
    ax.set_xticklabels(mensual["mes"].dt.strftime("%Y-%m").iloc[::2], rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    fig_html.add_trace(go.Bar(x=mensual["mes"], y=mensual["precip_estacion_mm_dia"], name="Estacion CRNS", marker_color=COLOR_ESTACION))
    fig_html.add_trace(go.Bar(x=mensual["mes"], y=mensual["precip_imerg_mm_dia"], name="IMERG", marker_color=COLOR_IMERG))
    aplicar_layout_plotly(fig_html, titulo)
    fig_html.update_layout(barmode="group")
    fig_html.update_xaxes(title_text="Mes")
    fig_html.update_yaxes(title_text="Precipitacion mensual (mm/mes)")
    guardar_html(fig_html, nombre)

    humedo = mensual.loc[mensual["precip_estacion_mm_dia"].idxmax()]
    seco = mensual.loc[mensual["precip_estacion_mm_dia"].idxmin()]
    diferencia_total = mensual["precip_imerg_mm_dia"].sum() - mensual["precip_estacion_mm_dia"].sum()
    print("\nFigura 2 - Acumulado mensual")
    print("- Pregunta: cuales meses son mas lluviosos o secos y como se comparan los acumulados.")
    print(f"- Patron principal: mes mas lluvioso observado {humedo['mes'].strftime('%Y-%m')} ({humedo['precip_estacion_mm_dia']:.2f} mm); mes mas seco {seco['mes'].strftime('%Y-%m')} ({seco['precip_estacion_mm_dia']:.2f} mm).")
    print(f"- Metrica: diferencia total mensual IMERG - estacion = {diferencia_total:.2f} mm.")
    return mensual


def figura_ciclo_diurno(estacion: pd.DataFrame) -> None:
    nombre = "fig03_ciclo_diurno_estacion"
    titulo = "Ciclo diurno de precipitacion observada en estacion CRNS"

    datos = estacion.dropna(subset=["fecha_hora_local", "precip_estacion_mm_5min"]).copy()
    datos["hora_local"] = datos["fecha_hora_local"].dt.hour
    ciclo = datos.groupby("hora_local", as_index=False)["precip_estacion_mm_5min"].sum()

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(ciclo["hora_local"], ciclo["precip_estacion_mm_5min"], color=COLOR_ESTACION, marker="o", lw=1.8, label="Estacion CRNS")
    ax.set_title(titulo)
    ax.set_xlabel("Hora local de la estacion (LST)")
    ax.set_ylabel("Precipitacion acumulada por hora (mm)")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    fig_html.add_trace(go.Scatter(x=ciclo["hora_local"], y=ciclo["precip_estacion_mm_5min"], mode="lines+markers", name="Estacion CRNS", line=dict(color=COLOR_ESTACION)))
    aplicar_layout_plotly(fig_html, titulo)
    fig_html.update_xaxes(title_text="Hora local de la estacion (LST)", dtick=2)
    fig_html.update_yaxes(title_text="Precipitacion acumulada por hora (mm)")
    guardar_html(fig_html, nombre)

    pico = ciclo.loc[ciclo["precip_estacion_mm_5min"].idxmax()]
    print("\nFigura 3 - Ciclo diurno de la estacion")
    print("- Pregunta: a que hora del dia se concentra mas la lluvia observada.")
    print(f"- Patron principal: el maximo acumulado ocurre alrededor de las {int(pico['hora_local']):02d}:00 LST.")
    print(f"- Metrica: acumulado en la hora pico = {pico['precip_estacion_mm_5min']:.2f} mm.")


def figura_histograma_intensidades(estacion_30min: pd.DataFrame, imerg: pd.DataFrame) -> None:
    nombre = "fig04_histograma_intensidades_estacion_imerg"
    titulo = "Distribucion de intensidades positivas a 30 minutos"

    est = estacion_30min["precip_estacion_mm_30min"].dropna()
    ime = imerg["precip_imerg_mm_30min"].dropna()
    est = est[est > 0]
    ime = ime[ime > 0]
    combinado = pd.concat([est, ime], ignore_index=True)
    limite = combinado.quantile(0.99)
    bins = np.linspace(0, limite, 35)

    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    ax.hist(est.clip(upper=limite), bins=bins, color=COLOR_ESTACION, alpha=0.55, label="Estacion CRNS")
    ax.hist(ime.clip(upper=limite), bins=bins, color=COLOR_IMERG, alpha=0.55, label="IMERG")
    ax.set_title(titulo)
    ax.set_xlabel("Precipitacion en 30 min (mm/30 min)")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    fig_html.add_trace(go.Histogram(x=est.clip(upper=limite), name="Estacion CRNS", opacity=0.58, marker_color=COLOR_ESTACION, xbins=dict(start=0, end=float(limite), size=float(limite / 34))))
    fig_html.add_trace(go.Histogram(x=ime.clip(upper=limite), name="IMERG", opacity=0.58, marker_color=COLOR_IMERG, xbins=dict(start=0, end=float(limite), size=float(limite / 34))))
    aplicar_layout_plotly(fig_html, titulo)
    fig_html.update_layout(barmode="overlay")
    fig_html.update_xaxes(title_text="Precipitacion en 30 min (mm/30 min)")
    fig_html.update_yaxes(title_text="Frecuencia")
    guardar_html(fig_html, nombre)

    print("\nFigura 4 - Histograma de intensidades")
    print("- Pregunta: si domina la lluvia debil o intensa y si IMERG suaviza intensidades.")
    print(f"- Patron principal: mediana positiva estacion = {est.median():.2f} mm/30 min; IMERG = {ime.median():.2f} mm/30 min.")
    print(f"- Metrica: percentil 95 estacion = {est.quantile(0.95):.2f}; IMERG = {ime.quantile(0.95):.2f} mm/30 min.")


def figura_dispersion_diaria(diario: pd.DataFrame, metricas: dict[str, float]) -> None:
    nombre = "fig05_dispersion_diaria_estacion_imerg"
    titulo = "Dispersion diaria: IMERG vs estacion CRNS"

    pares = diario[["precip_estacion_mm_dia", "precip_imerg_mm_dia"]].dropna()
    maximo = float(np.nanmax(pares.to_numpy()))
    limite = maximo * 1.05
    texto = (
        f"r = {metricas['correlacion']:.3f}\n"
        f"RMSE = {metricas['rmse']:.2f} mm/dia\n"
        f"Sesgo = {metricas['sesgo_medio']:.2f} mm/dia\n"
        f"PBIAS = {metricas['pbias']:.1f}%"
    )

    fig, ax = plt.subplots(figsize=(6.4, 6.1))
    ax.scatter(pares["precip_estacion_mm_dia"], pares["precip_imerg_mm_dia"], s=22, alpha=0.65, color="#3182bd", edgecolor="none", label="Dias")
    ax.plot([0, limite], [0, limite], color=COLOR_REFERENCIA, lw=1.3, ls="--", label="Linea 1:1")
    ax.text(0.04, 0.96, texto, transform=ax.transAxes, va="top", ha="left", bbox=dict(facecolor="white", alpha=0.85, edgecolor="#cccccc"))
    ax.set_title(titulo)
    ax.set_xlabel("Estacion CRNS (mm/dia)")
    ax.set_ylabel("IMERG (mm/dia)")
    ax.set_xlim(0, limite)
    ax.set_ylim(0, limite)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / f"{nombre}.png", bbox_inches="tight")
    plt.close(fig)

    fig_html = go.Figure()
    fig_html.add_trace(go.Scatter(x=pares["precip_estacion_mm_dia"], y=pares["precip_imerg_mm_dia"], mode="markers", name="Dias", marker=dict(color="#3182bd", opacity=0.65, size=7)))
    fig_html.add_trace(go.Scatter(x=[0, limite], y=[0, limite], mode="lines", name="Linea 1:1", line=dict(color=COLOR_REFERENCIA, dash="dash")))
    fig_html.add_annotation(x=0.04 * limite, y=0.96 * limite, text=texto.replace("\n", "<br>"), showarrow=False, align="left", bgcolor="rgba(255,255,255,0.85)", bordercolor="#cccccc")
    aplicar_layout_plotly(fig_html, titulo)
    fig_html.update_xaxes(title_text="Estacion CRNS (mm/dia)", range=[0, limite])
    fig_html.update_yaxes(title_text="IMERG (mm/dia)", range=[0, limite], scaleanchor="x", scaleratio=1)
    guardar_html(fig_html, nombre)

    print("\nFigura 5 - Dispersion diaria")
    print("- Pregunta: cuanto se parecen los acumulados diarios de IMERG a los observados.")
    print(f"- Patron principal: el sesgo medio IMERG - estacion es {metricas['sesgo_medio']:.2f} mm/dia.")
    print(f"- Metrica: r = {metricas['correlacion']:.3f}, RMSE = {metricas['rmse']:.2f} mm/dia, PBIAS = {metricas['pbias']:.1f}%.")


def main() -> None:
    configurar_matplotlib()
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

    diario = leer_diario()
    estacion = leer_estacion_crns()
    imerg = leer_imerg()
    estacion_30min = agregar_estacion_30min(estacion)
    metricas = calcular_metricas_diarias(diario)

    figura_serie_diaria(diario)
    figura_acumulado_mensual(diario)
    figura_ciclo_diurno(estacion)
    figura_histograma_intensidades(estacion_30min, imerg)
    figura_dispersion_diaria(diario, metricas)

    print("\nArchivos generados en:")
    print(f"- {FIGURAS_DIR}")
    print("\nNota: los HTML se generan con Plotly. Los PNG se generan con Matplotlib, por lo que no dependen de Kaleido.")


if __name__ == "__main__":
    main()
