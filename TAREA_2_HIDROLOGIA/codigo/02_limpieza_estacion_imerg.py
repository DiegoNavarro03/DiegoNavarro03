from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_PREC_DIR = BASE_DIR / "DATOS_PREC"
ESTACION_DIR = DATOS_PREC_DIR / "ESTACIÓN"
IMERG_DIR = DATOS_PREC_DIR / "SATÉLITE" / "IMERG"
SALIDA_CSV = BASE_DIR / "datos_limpios" / "precipitacion_diaria_estacion_imerg.csv"

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


def validar_archivos(rutas: list[Path]) -> None:
    faltantes = [ruta for ruta in rutas if not ruta.exists()]
    if faltantes:
        lista = "\n".join(f"- {ruta}" for ruta in faltantes)
        raise FileNotFoundError(f"No se encontraron estos archivos:\n{lista}")


def detectar_resolucion(serie: pd.Series) -> str:
    diferencias = serie.dropna().sort_values().diff().dropna()
    if diferencias.empty:
        return "sin datos suficientes"
    return str(diferencias.mode().iloc[0])


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
    fecha = estacion["utc_date"].astype("Int64").astype(str).str.zfill(8)
    hora = estacion["utc_time"].astype("Int64").astype(str).str.zfill(4)
    estacion["fecha_hora_utc"] = pd.to_datetime(
        fecha + hora,
        format="%Y%m%d%H%M",
        errors="coerce",
    )
    estacion["precip_estacion_mm_5min"] = pd.to_numeric(
        estacion["p_calc_mm"],
        errors="coerce",
    )
    estacion.loc[estacion["precip_estacion_mm_5min"] < 0, "precip_estacion_mm_5min"] = np.nan

    return estacion.sort_values("fecha_hora_utc").reset_index(drop=True)


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

    # IMERG half-hourly precipitation is a rate in mm/h; 30 min = 0.5 h.
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


def agregar_diario(df: pd.DataFrame, columna_precip: str, nombre_salida: str) -> pd.DataFrame:
    return (
        df.set_index("fecha_hora_utc")[columna_precip]
        .resample("D")
        .sum(min_count=1)
        .rename(nombre_salida)
        .reset_index()
        .rename(columns={"fecha_hora_utc": "fecha"})
    )


def construir_csv_diario(estacion_30min: pd.DataFrame, imerg: pd.DataFrame) -> pd.DataFrame:
    estacion_diaria = agregar_diario(
        estacion_30min,
        "precip_estacion_mm_30min",
        "precip_estacion_mm_dia",
    )
    imerg_diario = agregar_diario(
        imerg,
        "precip_imerg_mm_30min",
        "precip_imerg_mm_dia",
    )

    diario = pd.merge(estacion_diaria, imerg_diario, on="fecha", how="outer")
    diario = diario.sort_values("fecha").reset_index(drop=True)
    diario["fecha"] = diario["fecha"].dt.date
    return diario


def imprimir_resumen_fuente(nombre: str, df: pd.DataFrame, columna_precip: str) -> None:
    fechas = df["fecha_hora_utc"].dropna()
    precip = df[columna_precip]
    print(f"\n{nombre}")
    print(f"- Fecha inicial: {fechas.min()}")
    print(f"- Fecha final: {fechas.max()}")
    print(f"- Filas: {len(df):,}")
    print(f"- Datos validos de precipitacion: {precip.notna().sum():,}")
    print(f"- Resolucion temporal detectada: {detectar_resolucion(fechas)}")
    print(f"- Precipitacion total: {precip.sum(skipna=True):.2f} mm")


def imprimir_resumen_final(
    estacion: pd.DataFrame,
    estacion_30min: pd.DataFrame,
    imerg: pd.DataFrame,
    diario: pd.DataFrame,
) -> None:
    print("Columna de precipitacion CRNS identificada: p_calc_mm (columna 10)")
    print("Columna de precipitacion IMERG identificada: mean_GPM_3IMERGHH_07_precipitation")

    imprimir_resumen_fuente("Estacion CRNS original 5 min", estacion, "precip_estacion_mm_5min")
    imprimir_resumen_fuente(
        "Estacion CRNS agregada 30 min",
        estacion_30min,
        "precip_estacion_mm_30min",
    )
    imprimir_resumen_fuente("IMERG original 30 min", imerg, "precip_imerg_mm_30min")

    print("\nCSV diario estacion vs IMERG")
    print(f"- Fecha inicial: {diario['fecha'].min()}")
    print(f"- Fecha final: {diario['fecha'].max()}")
    print(f"- Dias: {len(diario):,}")
    print(f"- Total estacion diario: {diario['precip_estacion_mm_dia'].sum(skipna=True):.2f} mm")
    print(f"- Total IMERG diario: {diario['precip_imerg_mm_dia'].sum(skipna=True):.2f} mm")
    print(f"- Archivo guardado: {SALIDA_CSV}")


def main() -> None:
    estacion = leer_estacion_crns()
    imerg = leer_imerg()
    estacion_30min = agregar_estacion_30min(estacion)
    diario = construir_csv_diario(estacion_30min, imerg)

    SALIDA_CSV.parent.mkdir(parents=True, exist_ok=True)
    diario.to_csv(SALIDA_CSV, index=False)

    imprimir_resumen_final(estacion, estacion_30min, imerg, diario)


if __name__ == "__main__":
    main()
