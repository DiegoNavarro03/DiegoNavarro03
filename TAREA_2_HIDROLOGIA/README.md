# Tarea 2 - Hidrologia

Este proyecto analiza la variabilidad temporal de la precipitacion en Champaign, Illinois, comparando una estacion terrestre CRNS con los productos satelitales IMERG y CMORPH para el periodo 2015-2024.

El README esta pensado como guia de reproduccion: explica que carpetas se usan, que datos quedaron en la entrega actual y que comandos ejecutar.

## Estructura actual

- `codigo/`: scripts numerados de procesamiento y analisis.
- `datos_limpios/`: tablas CSV procesadas que quedaron disponibles para reproducir los analisis principales.
- `figuras/`: figuras PNG y HTML ya generadas para el informe.
- `cuenca/`: resultados ya generados de la delimitacion de cuenca y caudal teorico.
- `informe/`: material del informe final.

Nota importante: la carpeta de datos originales `DATOS_PREC/` fue retirada de la version actual. Por eso, los scripts que dependen de datos crudos no pueden ejecutarse desde cero a menos que se vuelva a agregar esa carpeta con la estructura esperada.

## Dependencias

Instalacion recomendada:

```bash
py -m pip install pandas numpy matplotlib plotly kaleido scipy geopandas shapely pyproj fiona
```

Si `geopandas` o `fiona` fallan en Windows por dependencias geoespaciales, usar Conda:

```bash
conda install geopandas
```

## Ejecucion con la entrega actual

Con los archivos que quedaron en `datos_limpios/`, se pueden regenerar los analisis que usan tablas ya procesadas:

```bash
py "TAREA_2_HIDROLOGIA/codigo/05_graficas_estacion_imerg_cmorph.py"
py "TAREA_2_HIDROLOGIA/codigo/06_anomalias_memoria.py"
py "TAREA_2_HIDROLOGIA/codigo/07_analisis_precipitacion_anual.py"
py "TAREA_2_HIDROLOGIA/codigo/11_analisis_espectral.py"
```

El analisis espectral crea automaticamente la carpeta `html/` si no existe y guarda alli la figura interactiva:

```text
TAREA_2_HIDROLOGIA/html/fig21_analisis_espectral_precipitacion.html
```

## Ejecucion completa desde datos originales

Para reproducir absolutamente todo el flujo con el script maestro, primero debe existir la carpeta `TAREA_2_HIDROLOGIA/DATOS_PREC/` con los datos originales de estacion, IMERG y CMORPH.

Cuando esa carpeta este disponible, ejecutar desde la raiz del repositorio:

```bash
py "TAREA_2_HIDROLOGIA/codigo/00_ejecutar_todo.py"
```

El script maestro ejecuta:

1. `04_limpieza_estacion_imerg_cmorph.py`
2. `05_graficas_estacion_imerg_cmorph.py`
3. `06_anomalias_memoria.py`
4. `11_analisis_espectral.py`
5. `07_analisis_precipitacion_anual.py`
6. `08_analisis_ciclo_diurno.py`
7. `09_curvas_idf_empiricas.py`
8. `10_cuenca_caudal_teorico.py`

Los scripts `04`, `08`, `09` y `10` pueden requerir datos crudos o recursos geoespaciales adicionales. Si `DATOS_PREC/` no existe, el flujo maestro no es reproducible completo.

## Datos procesados disponibles

Las tablas principales estan en `datos_limpios/`:

- `precipitacion_diaria_estacion_imerg_cmorph.csv`
- `precipitacion_mensual_estacion_imerg_cmorph.csv`
- `precipitacion_anual_estacion_imerg_cmorph.csv`
- `anomalias_mensuales_estacion_imerg_cmorph.csv`
- `ciclo_diurno_horario_estacion_imerg_cmorph.csv`
- `climatologia_mensual_estacion_imerg_cmorph.csv`
- `idf_maximos_anuales_estacion.csv`
- `idf_intensidades_diseno_estacion.csv`
- `idf_intensidades_diseno_satelites.csv`
- `idf_comparacion_30min_o_mas.csv`
- `analisis_espectral_picos.csv`

## Bonus: analisis espectral de Fourier

Script:

```text
TAREA_2_HIDROLOGIA/codigo/11_analisis_espectral.py
```

Entrada:

```text
TAREA_2_HIDROLOGIA/datos_limpios/precipitacion_diaria_estacion_imerg_cmorph.csv
```

Procedimiento:

- lee la precipitacion diaria de estacion CRNS, IMERG y CMORPH;
- ordena la serie por fecha;
- reindexa el periodo 2015-2024 al calendario diario completo;
- rellena valores faltantes con `0.0 mm` y lo informa en consola;
- calcula el promedio mensual de la precipitacion diaria para evitar saturacion del espectro;
- remueve la media de cada serie;
- aplica FFT con `numpy.fft`;
- convierte frecuencia a periodo en dias;
- grafica potencia espectral contra periodo;
- resalta 365, 180 y 30 dias como referencias hidrologicas.

Salidas:

```text
TAREA_2_HIDROLOGIA/figuras/fig21_analisis_espectral_precipitacion.png
TAREA_2_HIDROLOGIA/html/fig21_analisis_espectral_precipitacion.html
TAREA_2_HIDROLOGIA/datos_limpios/analisis_espectral_picos.csv
```

Resultado principal: al usar promedio mensual, el ciclo anual de 365.25 dias aparece como periodo dominante en estacion CRNS, IMERG y CMORPH.

## Resultados principales

- Periodo de analisis: 2015-01-01 a 2024-12-31.
- Dias del periodo: 3653.
- Dias comunes entre estacion, IMERG y CMORPH: 3314.
- Datos faltantes en estacion CRNS: 9.28%.
- Datos faltantes en IMERG y CMORPH: 0.00%.

Precipitacion total 2015-2024:

- Estacion CRNS: 8737.20 mm.
- IMERG: 11075.89 mm.
- CMORPH: 9402.79 mm.

Comparacion diaria:

- IMERG: r = 0.797, RMSE = 4.77 mm/dia, PBIAS = 14.5%.
- CMORPH: r = 0.701, RMSE = 5.59 mm/dia, PBIAS = -1.7%.

Analisis espectral:

- Periodo dominante estacion CRNS: 365.25 dias.
- Periodo dominante IMERG: 365.25 dias.
- Periodo dominante CMORPH: 365.25 dias.
- El ciclo anual aparece como senal dominante en las tres fuentes.

## Salidas principales

Figuras recomendadas para el informe:

- `fig01_serie_diaria_estacion_imerg_cmorph`
- `fig02_acumulado_mensual_estacion_imerg_cmorph`
- `fig04_distribucion_intensidades_estacion_imerg_cmorph`
- `fig05_dispersion_diaria_estacion_imerg_cmorph`
- `fig06_anomalias_mensuales_estacion_imerg_cmorph`
- `fig08_acf_anomalias_mensuales_estacion_imerg_cmorph`
- `fig12_climatologia_mensual_estacion_imerg_cmorph`
- `fig14_ciclo_diurno_porcentaje_estacion_imerg_cmorph`
- `fig15_ciclo_diurno_mes_hora_estacion_crns`
- `fig16_curvas_idf_estacion_crns`
- `fig17_comparacion_idf_estacion_imerg_cmorph`
- `fig19_caudal_maximo_teorico_cuenca`
- `fig21_analisis_espectral_precipitacion`
- `mapa_cuenca_estacion_champaign`

## Advertencias de reproduccion

- La carpeta `DATOS_PREC/` no esta en la estructura actual; sin ella no se puede correr el flujo completo desde datos crudos.
- `datos_limpios/` se conserva porque los scripts actuales apuntan a esa ruta.
- La carpeta `html/` no necesita existir antes de ejecutar; `11_analisis_espectral.py` la crea automaticamente.
- Algunos HTML historicos siguen dentro de `figuras/` porque fueron generados por scripts anteriores.

## IA utilizada

Se usaron herramientas de IA, incluyendo ChatGPT y Codex, como apoyo para estructurar el flujo de trabajo, depurar codigo, organizar resultados y redactar interpretaciones. Los resultados fueron revisados a partir de las salidas generadas por los scripts.
