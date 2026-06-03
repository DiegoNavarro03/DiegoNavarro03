from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_PREC_DIR = BASE_DIR / "DATOS_PREC"
ESTACION_DIR = next(DATOS_PREC_DIR.glob("ESTACI*"), DATOS_PREC_DIR / "ESTACION")
SATELITE_DIR = next(DATOS_PREC_DIR.glob("SAT*LITE"), DATOS_PREC_DIR / "SATELITE")
IMERG_DIR = SATELITE_DIR / "IMERG"
CMORPH_DIR = SATELITE_DIR / "CMORPH"

SALIDA_CSV_ESTACION_IMERG = (
    BASE_DIR / "datos_limpios" / "precipitacion_diaria_estacion_imerg.csv"
)
SALIDA_CSV_TRES_FUENTES = (
    BASE_DIR / "datos_limpios" / "precipitacion_diaria_estacion_imerg_cmorph.csv"
)

INICIO_COMPARACION = pd.Timestamp("2022-01-01")
FIN_COMPARACION_EXCLUSIVO = pd.Timestamp("2025-01-01")

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

ARCHIVOS_CMORPH = [
    "cmorph_champaign_1998_2025.csv",
    "cmorph_champaign_1998_2025.csv.csv",
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


def construir_fecha_hora(df: pd.DataFrame, col_fecha: str, col_hora: str) -> pd.Series:
    fecha = df[col_fecha].astype("Int64").astype(str).str.zfill(8)
    hora = df[col_hora].astype("Int64").astype(str).str.zfill(4)
    return pd.to_datetime(fecha + hora, format="%Y%m%d%H%M", errors="coerce")


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


def buscar_archivo_cmorph() -> Path:
    for nombre in ARCHIVOS_CMORPH:
        ruta = CMORPH_DIR / nombre
        if ruta.exists():
            return ruta

    validar_archivos([CMORPH_DIR / ARCHIVOS_CMORPH[0]])
    raise RuntimeError("No se pudo resolver el archivo CMORPH.")


def leer_cmorph() -> pd.DataFrame:
    ruta = buscar_archivo_cmorph()
    cmorph = pd.read_csv(
        ruta,
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

    # CMORPH viene como tasa en mm/h y tiene resolucion de 30 min.
    cmorph["precip_cmorph_mm_30min"] = cmorph["precip_mm_per_hr"] * 0.5

    cmorph = cmorph[
        (cmorph["fecha_hora_utc"] >= INICIO_COMPARACION)
        & (cmorph["fecha_hora_utc"] < FIN_COMPARACION_EXCLUSIVO)
    ]
    return cmorph.sort_values("fecha_hora_utc").reset_index(drop=True)


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


def construir_csv_tres_fuentes(diario: pd.DataFrame, cmorph: pd.DataFrame) -> pd.DataFrame:
    cmorph_diario = agregar_diario(
        cmorph,
        "precip_cmorph_mm_30min",
        "cmorph_mm",
    )

    base = diario.copy()
    base["fecha"] = pd.to_datetime(base["fecha"])
    base = base[
        (base["fecha"] >= INICIO_COMPARACION)
        & (base["fecha"] < FIN_COMPARACION_EXCLUSIVO)
    ]
    base = base.rename(
        columns={
            "precip_estacion_mm_dia": "estacion_mm",
            "precip_imerg_mm_dia": "imerg_mm",
        }
    )
    base = base[["fecha", "estacion_mm", "imerg_mm"]]

    tres_fuentes = pd.merge(base, cmorph_diario, on="fecha", how="outer")
    tres_fuentes = tres_fuentes[
        (tres_fuentes["fecha"] >= INICIO_COMPARACION)
        & (tres_fuentes["fecha"] < FIN_COMPARACION_EXCLUSIVO)
    ]
    tres_fuentes = tres_fuentes.sort_values("fecha").reset_index(drop=True)
    tres_fuentes["fecha"] = tres_fuentes["fecha"].dt.date
    return tres_fuentes[["fecha", "estacion_mm", "imerg_mm", "cmorph_mm"]]


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
    cmorph: pd.DataFrame,
    diario: pd.DataFrame,
    tres_fuentes: pd.DataFrame,
) -> None:
    print("Columna de precipitacion CRNS identificada: p_calc_mm (columna 10)")
    print("Columna de precipitacion IMERG identificada: mean_GPM_3IMERGHH_07_precipitation")
    print("Columna de precipitacion CMORPH identificada: precip_mm_per_hr")

    imprimir_resumen_fuente("Estacion CRNS original 5 min", estacion, "precip_estacion_mm_5min")
    imprimir_resumen_fuente(
        "Estacion CRNS agregada 30 min",
        estacion_30min,
        "precip_estacion_mm_30min",
    )
    imprimir_resumen_fuente("IMERG original 30 min", imerg, "precip_imerg_mm_30min")
    imprimir_resumen_fuente("CMORPH filtrado 30 min", cmorph, "precip_cmorph_mm_30min")

    print("\nCSV diario estacion vs IMERG")
    print(f"- Fecha inicial: {diario['fecha'].min()}")
    print(f"- Fecha final: {diario['fecha'].max()}")
    print(f"- Dias: {len(diario):,}")
    print(f"- Total estacion diario: {diario['precip_estacion_mm_dia'].sum(skipna=True):.2f} mm")
    print(f"- Total IMERG diario: {diario['precip_imerg_mm_dia'].sum(skipna=True):.2f} mm")
    print(f"- Archivo guardado: {SALIDA_CSV_ESTACION_IMERG}")

    dias_comparados = tres_fuentes[["estacion_mm", "imerg_mm", "cmorph_mm"]].dropna()
    print("\nCSV diario estacion vs IMERG vs CMORPH")
    print(f"- Fecha inicial CMORPH filtrado: {cmorph['fecha_hora_utc'].min()}")
    print(f"- Fecha final CMORPH filtrado: {cmorph['fecha_hora_utc'].max()}")
    print(f"- Registros CMORPH filtrados: {len(cmorph):,}")
    print(f"- Total CMORPH 2022-2024: {tres_fuentes['cmorph_mm'].sum(skipna=True):.2f} mm")
    print(f"- Total estacion 2022-2024: {tres_fuentes['estacion_mm'].sum(skipna=True):.2f} mm")
    print(f"- Total IMERG 2022-2024: {tres_fuentes['imerg_mm'].sum(skipna=True):.2f} mm")
    print(f"- Dias comparados entre tres fuentes: {len(dias_comparados):,}")
    print(f"- Archivo guardado: {SALIDA_CSV_TRES_FUENTES}")


def main() -> None:
    estacion = leer_estacion_crns()
    imerg = leer_imerg()
    cmorph = leer_cmorph()
    estacion_30min = agregar_estacion_30min(estacion)
    diario = construir_csv_diario(estacion_30min, imerg)
    tres_fuentes = construir_csv_tres_fuentes(diario, cmorph)

    SALIDA_CSV_ESTACION_IMERG.parent.mkdir(parents=True, exist_ok=True)
    diario.to_csv(SALIDA_CSV_ESTACION_IMERG, index=False)
    tres_fuentes.to_csv(SALIDA_CSV_TRES_FUENTES, index=False)

    imprimir_resumen_final(estacion, estacion_30min, imerg, cmorph, diario, tres_fuentes)


if __name__ == "__main__":
    main()
