from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "04_limpieza_estacion_imerg_cmorph.py",
    "05_graficas_estacion_imerg_cmorph.py",
    "06_anomalias_memoria.py",
    "11_analisis_espectral.py",
    "07_analisis_precipitacion_anual.py",
    "08_analisis_ciclo_diurno.py",
    "09_curvas_idf_empiricas.py",
    "10_cuenca_caudal_teorico.py",
]


def detectar_raiz_proyecto() -> Path:
    actual = Path(__file__).resolve()
    for carpeta in [actual.parent, *actual.parents]:
        if carpeta.name == "TAREA_2_HIDROLOGIA":
            return carpeta
        if (carpeta / "codigo").is_dir() and (carpeta / "DATOS_PREC").is_dir():
            return carpeta
    raise RuntimeError("No se pudo detectar la carpeta raiz TAREA_2_HIDROLOGIA.")


def ejecutar_script(raiz: Path, nombre_script: str) -> None:
    ruta_script = raiz / "codigo" / nombre_script
    if not ruta_script.exists():
        raise FileNotFoundError(f"No se encontro el script requerido: {ruta_script}")

    print("\n" + "=" * 72)
    print(f"Ejecutando: {nombre_script}")
    print("=" * 72)

    resultado = subprocess.run([sys.executable, str(ruta_script)], cwd=raiz)
    if resultado.returncode != 0:
        raise RuntimeError(
            f"El script {nombre_script} fallo con codigo {resultado.returncode}."
        )


def main() -> None:
    raiz = detectar_raiz_proyecto()
    print(f"Raiz del proyecto detectada: {raiz}")
    print(f"Interprete de Python: {sys.executable}")

    for nombre_script in SCRIPTS:
        ejecutar_script(raiz, nombre_script)

    print("\nFlujo completo ejecutado correctamente")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nError durante la ejecucion del flujo completo:")
        print(f"{type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc
