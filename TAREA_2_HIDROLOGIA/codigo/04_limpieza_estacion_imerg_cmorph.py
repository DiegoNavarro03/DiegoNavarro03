from pathlib import Path
import re

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_LIMPIOS_DIR = BASE_DIR / "datos_limpios"
DATOS_PREC_DIR = BASE_DIR / "DATOS_PREC"
ESTACION_DIR = next(DATOS_PREC_DIR.glob("ESTACI*"), DATOS_PREC_DIR / "ESTACION")
SATELITE_DIR = next(DATOS_PREC_DIR.glob("SAT*LITE"), DATOS_PREC_DIR / "SATELITE")
IMERG_DIR = SATELITE_DIR / "IMERG"
CMORPH_DIR = SATELITE_DIR / "CMORPH"

ENTRADA_CSV_CMORPH = CMORPH_DIR / "cmorph_champaign_1998_2025.csv"
SALIDA_CSV_DIARIO = DATOS_LIMPIOS_DIR / "precipitacion_diaria_estacion_imerg_cmorph.csv"
SALIDA_CSV_MENSUAL = DATOS_LIMPIOS_DIR / "precipitacion_mensual_estacion_imerg_cmorph.csv"

INICIO_COMPARACION = pd.Timestamp("2015-01-01")
FIN_COMPARACION = pd.Timestamp("2024-12-31")
FIN_COMPARACION_EXCLUSIVO = FIN_COMPARACION + pd.Timedelta(days=1)
ANIOS = range(INICIO_COMPARACION.year, FIN_COMPARACION.year + 1)

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
COLUMNAS_SALIDA = ["estacion_mm", "imerg_mm", "cmorph_mm"]


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


def leer_estacion_5min() -> pd.DataFrame:
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
    return filtrar_periodo(estacion, "fecha_hora_utc")[["fecha_hora_utc", "precip_mm"]]


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
    imerg["precip_mm_per_hr"] = pd.to_numeric(
        imerg["precip_mm_per_hr"],
        errors="coerce",
    )
    imerg.loc[imerg["precip_mm_per_hr"] < 0, "precip_mm_per_hr"] = np.nan
    imerg["imerg_mm_30min"] = imerg["precip_mm_per_hr"] * 0.5
    return filtrar_periodo(imerg, "fecha_hora_utc")[["fecha_hora_utc", "imerg_mm_30min"]]


def leer_cmorph_30min() -> pd.DataFrame:
    validar_archivos([ENTRADA_CSV_CMORPH])

    cmorph = pd.read_csv(
        ENTRADA_CSV_CMORPH,
        comment="#",
        na_values=VALORES_FALTANTES,
    )
    cmorph.columns = [col.strip() for col in cmorph.columns]

    columnas_requeridas = {"time", "precip_mm_per_hr"}
    faltantes = columnas_requeridas.difference(cmorph.columns)
    if faltantes:
        raise ValueError(f"CMORPH no tiene estas columnas requeridas: {sorted(faltantes)}")

    cmorph["fecha_hora_utc"] = pd.to_datetime(cmorph["time"], errors="coerce")
    cmorph["precip_mm_per_hr"] = pd.to_numeric(
        cmorph["precip_mm_per_hr"],
        errors="coerce",
    )
    cmorph.loc[cmorph["precip_mm_per_hr"] < 0, "precip_mm_per_hr"] = np.nan
    cmorph = filtrar_periodo(cmorph, "fecha_hora_utc")

    # CMORPH viene como tasa en mm/h y el intervalo temporal es de 30 minutos.
    cmorph["cmorph_mm_30min"] = cmorph["precip_mm_per_hr"] * 0.5
    return cmorph[["fecha_hora_utc", "cmorph_mm_30min"]]


def agregar_diario(
    estacion_5min: pd.DataFrame,
    imerg_30min: pd.DataFrame,
    cmorph_30min: pd.DataFrame,
) -> pd.DataFrame:
    estacion_diaria = (
        estacion_5min.set_index("fecha_hora_utc")["precip_mm"]
        .resample("D")
        .sum(min_count=1)
        .rename("estacion_mm")
        .reset_index()
        .rename(columns={"fecha_hora_utc": "fecha"})
    )
    imerg_diario = (
        imerg_30min.set_index("fecha_hora_utc")["imerg_mm_30min"]
        .resample("D")
        .sum(min_count=1)
        .rename("imerg_mm")
        .reset_index()
        .rename(columns={"fecha_hora_utc": "fecha"})
    )
    cmorph_diario = (
        cmorph_30min.set_index("fecha_hora_utc")["cmorph_mm_30min"]
        .resample("D")
        .sum(min_count=1)
        .rename("cmorph_mm")
        .reset_index()
        .rename(columns={"fecha_hora_utc": "fecha"})
    )

    fechas = pd.DataFrame(
        {"fecha": pd.date_range(INICIO_COMPARACION, FIN_COMPARACION, freq="D")}
    )
    diario = fechas.merge(estacion_diaria, on="fecha", how="left")
    diario = diario.merge(imerg_diario, on="fecha", how="left")
    diario = diario.merge(cmorph_diario, on="fecha", how="left")
    diario = diario[["fecha", *COLUMNAS_SALIDA]]
    diario["fecha"] = diario["fecha"].dt.strftime("%Y-%m-%d")
    return diario


def construir_tabla_mensual(diario: pd.DataFrame) -> pd.DataFrame:
    mensual = diario.copy()
    mensual["fecha"] = pd.to_datetime(mensual["fecha"])
    mensual = (
        mensual.set_index("fecha")[COLUMNAS_SALIDA]
        .resample("MS")
        .sum(min_count=1)
        .reset_index()
    )
    mensual["fecha"] = mensual["fecha"].dt.strftime("%Y-%m")
    return mensual


def porcentaje_faltantes(serie: pd.Series) -> float:
    return serie.isna().mean() * 100


def imprimir_resumen(diario: pd.DataFrame, archivos: list[Path]) -> None:
    fechas = pd.to_datetime(diario["fecha"])
    dias_comunes = int(diario[COLUMNAS_SALIDA].notna().all(axis=1).sum())
    numero_anios = FIN_COMPARACION.year - INICIO_COMPARACION.year + 1

    print("\nResumen final del flujo base")
    print(f"- Periodo usado: {INICIO_COMPARACION:%Y}–{FIN_COMPARACION:%Y}")
    print(f"- Número de años analizados: {numero_anios}")
    print(f"- Fecha inicial: {fechas.min().date()}")
    print(f"- Fecha final: {fechas.max().date()}")
    print(f"- Número de días del periodo: {len(diario):,}")
    print(f"- Número de días comunes entre estación, IMERG y CMORPH: {dias_comunes:,}")
    print("- Precipitación total por fuente:")
    print(f"  Estación CRNS: {diario['estacion_mm'].sum(skipna=True):.2f} mm")
    print(f"  IMERG: {diario['imerg_mm'].sum(skipna=True):.2f} mm")
    print(f"  CMORPH: {diario['cmorph_mm'].sum(skipna=True):.2f} mm")
    print("- Porcentaje de datos faltantes por fuente:")
    print(f"  Estación CRNS: {porcentaje_faltantes(diario['estacion_mm']):.2f}%")
    print(f"  IMERG: {porcentaje_faltantes(diario['imerg_mm']):.2f}%")
    print(f"  CMORPH: {porcentaje_faltantes(diario['cmorph_mm']):.2f}%")
    print("- Archivos regenerados:")
    for archivo in archivos:
        print(f"  {archivo}")
    print("- Confirmación: no se crearon archivos duplicados innecesarios; se sobrescribieron las salidas existentes.")


def main() -> None:
    estacion_5min = leer_estacion_5min()
    imerg_30min = leer_imerg_30min()
    cmorph_30min = leer_cmorph_30min()
    diario = agregar_diario(estacion_5min, imerg_30min, cmorph_30min)
    mensual = construir_tabla_mensual(diario)

    DATOS_LIMPIOS_DIR.mkdir(parents=True, exist_ok=True)
    diario.to_csv(SALIDA_CSV_DIARIO, index=False)
    mensual.to_csv(SALIDA_CSV_MENSUAL, index=False)

    imprimir_resumen(diario, [SALIDA_CSV_DIARIO, SALIDA_CSV_MENSUAL])


if __name__ == "__main__":
    main()
