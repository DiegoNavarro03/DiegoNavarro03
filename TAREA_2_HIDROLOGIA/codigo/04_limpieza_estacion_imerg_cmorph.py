from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_LIMPIOS_DIR = BASE_DIR / "datos_limpios"
DATOS_PREC_DIR = BASE_DIR / "DATOS_PREC"
SATELITE_DIR = next(DATOS_PREC_DIR.glob("SAT*LITE"), DATOS_PREC_DIR / "SATELITE")
CMORPH_DIR = SATELITE_DIR / "CMORPH"

ENTRADA_CSV_ESTACION_IMERG = (
    DATOS_LIMPIOS_DIR / "precipitacion_diaria_estacion_imerg.csv"
)
ENTRADA_CSV_CMORPH = CMORPH_DIR / "cmorph_champaign_1998_2025.csv"
SALIDA_CSV_DIARIO = (
    DATOS_LIMPIOS_DIR / "precipitacion_diaria_estacion_imerg_cmorph.csv"
)
SALIDA_CSV_MENSUAL = (
    DATOS_LIMPIOS_DIR / "precipitacion_mensual_estacion_imerg_cmorph.csv"
)

INICIO_COMPARACION = pd.Timestamp("2022-01-01")
FIN_COMPARACION = pd.Timestamp("2024-12-31")
FIN_COMPARACION_EXCLUSIVO = FIN_COMPARACION + pd.Timedelta(days=1)

VALORES_FALTANTES = [-99, -99.0, -9999, -9999.0, -9999.9]


def validar_archivos(rutas: list[Path]) -> None:
    faltantes = [ruta for ruta in rutas if not ruta.exists()]
    if faltantes:
        lista = "\n".join(f"- {ruta}" for ruta in faltantes)
        raise FileNotFoundError(f"No se encontraron estos archivos:\n{lista}")


def leer_diario_estacion_imerg() -> pd.DataFrame:
    validar_archivos([ENTRADA_CSV_ESTACION_IMERG])

    diario = pd.read_csv(ENTRADA_CSV_ESTACION_IMERG, na_values=VALORES_FALTANTES)
    diario.columns = [col.strip() for col in diario.columns]

    columnas_requeridas = {
        "fecha",
        "precip_estacion_mm_dia",
        "precip_imerg_mm_dia",
    }
    faltantes = columnas_requeridas.difference(diario.columns)
    if faltantes:
        raise ValueError(
            "El CSV diario estacion + IMERG no tiene estas columnas "
            f"requeridas: {sorted(faltantes)}"
        )

    diario = diario.rename(
        columns={
            "precip_estacion_mm_dia": "estacion_mm",
            "precip_imerg_mm_dia": "imerg_mm",
        }
    )
    diario["fecha"] = pd.to_datetime(diario["fecha"], errors="coerce")
    diario["estacion_mm"] = pd.to_numeric(diario["estacion_mm"], errors="coerce")
    diario["imerg_mm"] = pd.to_numeric(diario["imerg_mm"], errors="coerce")

    diario = diario[
        (diario["fecha"] >= INICIO_COMPARACION)
        & (diario["fecha"] <= FIN_COMPARACION)
    ]
    return diario[["fecha", "estacion_mm", "imerg_mm"]]


def leer_cmorph() -> pd.DataFrame:
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
        raise ValueError(
            f"CMORPH no tiene estas columnas requeridas: {sorted(faltantes)}"
        )

    cmorph["fecha_hora_utc"] = pd.to_datetime(cmorph["time"], errors="coerce")
    cmorph["precip_mm_per_hr"] = pd.to_numeric(
        cmorph["precip_mm_per_hr"],
        errors="coerce",
    )
    cmorph.loc[cmorph["precip_mm_per_hr"] < 0, "precip_mm_per_hr"] = np.nan

    cmorph = cmorph[
        (cmorph["fecha_hora_utc"] >= INICIO_COMPARACION)
        & (cmorph["fecha_hora_utc"] < FIN_COMPARACION_EXCLUSIVO)
    ].copy()

    # CMORPH viene como tasa en mm/h y el intervalo temporal es de 30 min.
    cmorph["cmorph_mm_30min"] = cmorph["precip_mm_per_hr"] * 0.5

    return cmorph[["fecha_hora_utc", "cmorph_mm_30min"]]


def agregar_cmorph_diario(cmorph: pd.DataFrame) -> pd.DataFrame:
    return (
        cmorph.set_index("fecha_hora_utc")["cmorph_mm_30min"]
        .resample("D")
        .sum(min_count=1)
        .rename("cmorph_mm")
        .reset_index()
        .rename(columns={"fecha_hora_utc": "fecha"})
    )


def construir_tabla_diaria(
    diario_estacion_imerg: pd.DataFrame,
    cmorph_diario: pd.DataFrame,
) -> pd.DataFrame:
    fechas = pd.DataFrame(
        {"fecha": pd.date_range(INICIO_COMPARACION, FIN_COMPARACION, freq="D")}
    )
    diario = fechas.merge(diario_estacion_imerg, on="fecha", how="left")
    diario = diario.merge(cmorph_diario, on="fecha", how="left")
    diario = diario[["fecha", "estacion_mm", "imerg_mm", "cmorph_mm"]]
    diario["fecha"] = diario["fecha"].dt.date
    return diario


def construir_tabla_mensual(diario: pd.DataFrame) -> pd.DataFrame:
    mensual = diario.copy()
    mensual["fecha"] = pd.to_datetime(mensual["fecha"])
    mensual = (
        mensual.set_index("fecha")[["estacion_mm", "imerg_mm", "cmorph_mm"]]
        .resample("MS")
        .sum(min_count=1)
        .reset_index()
    )
    mensual["fecha"] = mensual["fecha"].dt.strftime("%Y-%m")
    return mensual


def porcentaje_faltantes(serie: pd.Series) -> float:
    return serie.isna().mean() * 100


def imprimir_resumen(diario: pd.DataFrame) -> None:
    fechas = pd.to_datetime(diario["fecha"])

    print("Resumen precipitacion diaria estacion + IMERG + CMORPH")
    print(f"- Fecha inicial: {fechas.min().date()}")
    print(f"- Fecha final: {fechas.max().date()}")
    print(f"- Numero de dias comparados: {len(diario):,}")
    print(f"- Total precipitacion estacion: {diario['estacion_mm'].sum(skipna=True):.2f} mm")
    print(f"- Total precipitacion IMERG: {diario['imerg_mm'].sum(skipna=True):.2f} mm")
    print(f"- Total precipitacion CMORPH: {diario['cmorph_mm'].sum(skipna=True):.2f} mm")
    print("- Porcentaje de datos faltantes por fuente:")
    print(f"  estacion: {porcentaje_faltantes(diario['estacion_mm']):.2f}%")
    print(f"  IMERG: {porcentaje_faltantes(diario['imerg_mm']):.2f}%")
    print(f"  CMORPH: {porcentaje_faltantes(diario['cmorph_mm']):.2f}%")
    print(f"- CSV diario guardado: {SALIDA_CSV_DIARIO}")
    print(f"- CSV mensual guardado: {SALIDA_CSV_MENSUAL}")


def main() -> None:
    diario_estacion_imerg = leer_diario_estacion_imerg()
    cmorph = leer_cmorph()
    cmorph_diario = agregar_cmorph_diario(cmorph)
    diario = construir_tabla_diaria(diario_estacion_imerg, cmorph_diario)
    mensual = construir_tabla_mensual(diario)

    DATOS_LIMPIOS_DIR.mkdir(parents=True, exist_ok=True)
    diario.to_csv(SALIDA_CSV_DIARIO, index=False)
    mensual.to_csv(SALIDA_CSV_MENSUAL, index=False)

    imprimir_resumen(diario)


if __name__ == "__main__":
    main()
