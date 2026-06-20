from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_PREC_DIR = BASE_DIR / "TAREA_2_HIDROLOGIA" / "DATOS_PREC"
DATOS_DIR = next(DATOS_PREC_DIR.glob("ESTACI*"), DATOS_PREC_DIR / "ESTACION")
SALIDA_CSV = BASE_DIR / "datos" / "limpios" / "estacion_champaign_preliminar.csv"

ARCHIVOS = [
    "CRNS0101-05-2022-IL_Champaign_9_SW.txt",
    "CRNS0101-05-2023-IL_Champaign_9_SW.txt",
    "CRNS0101-05-2024-IL_Champaign_9_SW.txt",
]

SENTINELAS_FALTANTES = {-99, -99.0, -9999, -9999.0}


def leer_archivo(ruta: Path) -> pd.DataFrame:
    """Lee un archivo sin encabezado y conserva nombres genericos de columnas."""
    df = pd.read_csv(ruta, sep=r"\s+", header=None, dtype=str)
    df.columns = [f"col{i}" for i in range(1, df.shape[1] + 1)]
    df.insert(0, "archivo", ruta.name)
    return df


def crear_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Combina col2 (AAAAMMDD) y col3 (HHMM) en una columna datetime."""
    fecha = df["col2"].astype(str).str.zfill(8)
    hora = df["col3"].astype(str).str.zfill(4)
    df.insert(
        1,
        "datetime",
        pd.to_datetime(fecha + hora, format="%Y%m%d%H%M", errors="coerce"),
    )
    return df


def detectar_resolucion(serie_datetime: pd.Series) -> pd.Series:
    fechas = serie_datetime.dropna().sort_values()
    return fechas.diff().dropna()


def resumen_resolucion(df: pd.DataFrame) -> None:
    print("\nResolucion temporal detectada")
    for archivo, grupo in df.groupby("archivo", sort=True):
        diffs = detectar_resolucion(grupo["datetime"])
        moda = diffs.mode()
        resolucion = moda.iloc[0] if not moda.empty else pd.NaT
        print(f"- {archivo}: {resolucion} (filas: {len(grupo):,})")

    diffs_globales = detectar_resolucion(df["datetime"])
    moda_global = diffs_globales.mode()
    resolucion_global = moda_global.iloc[0] if not moda_global.empty else pd.NaT
    print(f"- Global: {resolucion_global}")


def resumen_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    cols_datos = [col for col in df.columns if col.startswith("col")]
    resumen = pd.DataFrame(index=cols_datos)
    resumen["faltantes_nan"] = df[cols_datos].isna().sum()

    datos_numericos = df[cols_datos].apply(pd.to_numeric, errors="coerce")
    resumen["no_numericos_o_nan"] = datos_numericos.isna().sum()
    resumen["sentinelas_-99_-9999"] = datos_numericos.isin(SENTINELAS_FALTANTES).sum()

    return resumen


def resumen_para_precipitacion(df: pd.DataFrame) -> pd.DataFrame:
    cols_datos = [col for col in df.columns if col.startswith("col")]
    datos_numericos = df[cols_datos].apply(pd.to_numeric, errors="coerce")

    resumen = datos_numericos.describe(percentiles=[0.25, 0.5, 0.75]).T
    resumen["n_validos"] = datos_numericos.notna().sum()
    resumen["n_ceros"] = datos_numericos.eq(0).sum()
    resumen["pct_ceros"] = resumen["n_ceros"] / len(df) * 100
    resumen["n_positivos"] = datos_numericos.gt(0).sum()
    resumen["n_negativos"] = datos_numericos.lt(0).sum()
    resumen["n_sentinelas"] = datos_numericos.isin(SENTINELAS_FALTANTES).sum()

    columnas = [
        "n_validos",
        "n_ceros",
        "pct_ceros",
        "n_positivos",
        "n_negativos",
        "n_sentinelas",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
        "mean",
        "std",
    ]
    return resumen[columnas].sort_values(
        by=["n_sentinelas", "n_negativos", "pct_ceros"], ascending=[True, True, False]
    )


def main() -> None:
    dataframes = []
    for nombre in ARCHIVOS:
        ruta = DATOS_DIR / nombre
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontro el archivo: {ruta}")
        dataframes.append(leer_archivo(ruta))

    estacion = pd.concat(dataframes, ignore_index=True)
    estacion = crear_datetime(estacion)
    estacion = estacion.sort_values("datetime").reset_index(drop=True)

    print("Primeras filas del archivo combinado")
    print(estacion.head())

    resumen_resolucion(estacion)

    print("\nConteo de datos faltantes por columna")
    print(resumen_faltantes(estacion))

    print("\nResumen numerico para identificar la posible columna de precipitacion")
    print(resumen_para_precipitacion(estacion))

    SALIDA_CSV.parent.mkdir(parents=True, exist_ok=True)
    estacion.to_csv(SALIDA_CSV, index=False)
    print(f"\nCSV preliminar exportado en: {SALIDA_CSV}")


if __name__ == "__main__":
    main()
