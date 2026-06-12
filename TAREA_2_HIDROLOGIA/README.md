# Tarea 2 - Variabilidad temporal de la precipitacion

## Objetivo

El objetivo de esta tarea fue caracterizar la variabilidad temporal de la precipitacion en Champaign, Illinois, comparando una estacion terrestre subhoraria con los productos satelitales IMERG y CMORPH. Ademas, se construyeron curvas IDF empiricas, se analizo una cuenca HydroBASINS asociada a la estacion y se calculo un caudal maximo teorico como limite superior hidrologico.

## Sitio de estudio

- Estacion: CRNS IL Champaign 9 SW.
- Ubicacion: Champaign, Illinois, Estados Unidos.
- Coordenadas aproximadas: 40.05, -88.37.
- Justificacion: se selecciono porque cuenta con datos subhorarios de precipitacion y esta dentro de la cobertura de IMERG y CMORPH.

## Fuentes de datos

- Estacion CRNS: datos subhorarios de precipitacion.
- IMERG GPM: precipitacion satelital con resolucion temporal de 30 minutos.
- CMORPH NOAA: precipitacion satelital con resolucion temporal de 30 minutos.
- HydroBASINS/HydroSHEDS: poligonos de cuencas usados para delimitar la cuenca asociada a la estacion.

## Periodo de analisis

- Periodo oficial: 2015-01-01 a 2024-12-31.
- Numero de anos: 10.
- Numero de dias del periodo: 3653.
- Numero de dias comunes entre estacion, IMERG y CMORPH: 3314.
- Porcentaje de datos faltantes:
  - Estacion CRNS: 9.28%.
  - IMERG: 0.00%.
  - CMORPH: 0.00%.

## Estructura de carpetas

- `codigo/`: scripts de procesamiento, analisis y generacion de figuras.
- `DATOS_PREC/`: datos originales o de entrada organizados por fuente.
- `datos_limpios/`: CSV procesados y listos para analisis.
- `figuras/`: figuras PNG y HTML generadas para el informe.
- `cuenca/`: archivos HydroBASINS, cuenca seleccionada, mapa y caudales teoricos.
- `informe/`: informe final de la tarea.

## Flujo de trabajo

El proyecto se reproduce mediante scripts numerados que deben ejecutarse en orden.

### `04_limpieza_estacion_imerg_cmorph.py`

- Limpia y homogeniza la informacion de estacion, IMERG y CMORPH.
- Genera series comparables en escalas diaria y mensual.

### `05_graficas_estacion_imerg_cmorph.py`

- Genera figuras base de comparacion temporal.
- Incluye serie diaria, acumulados mensuales, distribucion de intensidades y dispersion diaria.

### `06_anomalias_memoria.py`

- Calcula anomalias mensuales.
- Evalua memoria temporal mediante autocorrelacion de la serie mensual y de las anomalias.

### `07_analisis_precipitacion_anual.py`

- Analiza precipitacion anual y diaria.
- Calcula momentos estadisticos, L-momentos, percentiles y distribucion empirica acumulada.

### `08_analisis_ciclo_diurno.py`

- Analiza el ciclo diurno en hora local de Champaign.
- Compara estacion, IMERG y CMORPH.
- Genera ciclo diurno promedio, distribucion porcentual y mapa hora-mes.

### `09_curvas_idf_empiricas.py`

- Construye curvas IDF empiricas.
- Calcula maximos anuales de intensidad para diferentes duraciones.
- Estima intensidades de diseno con Gumbel.
- Compara estacion con IMERG y CMORPH para duraciones mayores o iguales a 30 min.

### `10_cuenca_caudal_teorico.py`

- Identifica la cuenca HydroBASINS que contiene la estacion.
- Calcula area y perimetro.
- Genera mapa de cuenca.
- Calcula caudal medio anual teorico y caudales pico teoricos.

## Instalacion de dependencias

Para instalar las dependencias principales:

```bash
py -m pip install pandas numpy matplotlib plotly kaleido scipy geopandas shapely pyproj fiona
```

En Windows, si `geopandas` o `fiona` fallan por dependencias de GDAL, puede usarse Conda:

```bash
conda install geopandas
```

Otra opcion funcional en algunos entornos de Python recientes es instalar `pyogrio` como motor geoespacial:

```bash
py -m pip install geopandas shapely pyproj pyogrio
```

## Como ejecutar todo el proyecto

Para reproducir todo el flujo:

```bash
py "TAREA_2_HIDROLOGIA/codigo/00_ejecutar_todo.py"
```

Para ejecutar un script individual:

```bash
py "TAREA_2_HIDROLOGIA/codigo/04_limpieza_estacion_imerg_cmorph.py"
```

## Resultados principales

### Precipitacion total 2015-2024

- Estacion CRNS: 8737.20 mm.
- IMERG: 11075.89 mm.
- CMORPH: 9402.79 mm.

### Comparacion diaria

- IMERG: r = 0.797, RMSE = 4.77 mm/dia, PBIAS = 14.5%.
- CMORPH: r = 0.701, RMSE = 5.59 mm/dia, PBIAS = -1.7%.

### Ciclo anual

- Estacion CRNS: mes mas lluvioso promedio mayo, 111.86 mm/mes; mes mas seco promedio febrero, 43.82 mm/mes.
- IMERG: mes mas lluvioso promedio julio, 128.44 mm/mes; mes mas seco promedio febrero, 62.43 mm/mes.
- CMORPH: mes mas lluvioso promedio julio, 119.48 mm/mes; mes mas seco promedio enero, 33.66 mm/mes.

### Ciclo diurno

- Hora pico estacion CRNS: 21:00.
- Hora pico IMERG: 02:00.
- Hora pico CMORPH: 22:00.
- Correlacion ciclo diurno estacion-IMERG: 0.455.
- Correlacion ciclo diurno estacion-CMORPH: 0.296.

### Anomalias y memoria

- IMERG reproduce mejor las anomalias mensuales: correlacion = 0.854, RMSE = 21.12 mm.
- CMORPH: correlacion = 0.790, RMSE = 25.47 mm.
- Persistencia media absoluta rezagos 1-3: serie completa = 0.088, anomalias = 0.104.

### IDF y sumidero

- Duracion critica recomendada para parqueadero de 200 m x 200 m: 15 min.
- Intensidad de diseno recomendada: I = 91.67 mm/h, d = 15 min, T = 10 anos, fuente = estacion CRNS, metodo = Gumbel.
- Escenarios IDF: T = 2 anos: 79.21 mm/h; T = 10 anos: 91.67 mm/h; T = 25 anos: 97.94 mm/h.

### Caudal del parqueadero

- Area = 4 ha.
- Coeficiente de escorrentia: C = 0.9.
- Caudal estimado por metodo racional: aproximadamente 0.92 m3/s.

### Cuenca y caudal maximo teorico

Los valores de esta seccion provienen de `cuenca/atributos_cuenca_champaign.csv` y `cuenca/caudal_maximo_teorico_champaign.csv`.

- HYBAS_ID: 7070552430.
- Nivel HydroBASINS: 7.
- Area calculada: 1768.20 km2.
- Perimetro calculado: 249.48 km.
- Caudal medio anual teorico: 48.96 m3/s.
- Caudal pico teorico estacion 30 min: 31336.67 m3/s.
- Caudal pico teorico estacion 60 min: 19597.70 m3/s.
- Caudal pico teorico estacion 360 min: 5239.15 m3/s.
- Caudal pico teorico estacion 1440 min: 1653.61 m3/s.
- Caudal pico teorico IMERG 30 min: 25388.60 m3/s.
- Caudal pico teorico IMERG 60 min: 15602.03 m3/s.
- Caudal pico teorico IMERG 360 min: 5357.03 m3/s.
- Caudal pico teorico IMERG 1440 min: 1884.87 m3/s.
- Caudal pico teorico CMORPH 30 min: 38831.93 m3/s.
- Caudal pico teorico CMORPH 60 min: 19415.97 m3/s.
- Caudal pico teorico CMORPH 360 min: 9580.28 m3/s.
- Caudal pico teorico CMORPH 1440 min: 2395.07 m3/s.

## Advertencias metodologicas

- La estacion CRNS presenta 9.28% de datos faltantes; por eso algunos anos, especialmente 2018, deben interpretarse con cautela.
- IMERG tiende a sobreestimar el acumulado total frente a la estacion.
- CMORPH se aproxima mejor al acumulado total, pero presenta una cola extrema mas marcada.
- IMERG y CMORPH tienen resolucion temporal de 30 minutos, por lo que no se usaron para construir IDF de 5, 10 ni 15 minutos.
- El valor IDF para T = 25 anos es extrapolacion porque el registro usado tiene 10 anos.
- Los caudales maximos teoricos de cuenca son limites superiores absolutos porque se asumio ET = 0 y deltaS = 0.

## Figuras principales

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
- `mapa_cuenca_estacion_champaign`

Algunas figuras adicionales quedan como respaldo o anexo.

## Herramientas de IA utilizadas

Se usaron herramientas de IA, incluyendo ChatGPT y Codex, como apoyo para:

- estructurar el flujo de trabajo,
- depurar codigo,
- organizar resultados,
- proponer interpretaciones,
- redactar borradores del analisis.

Los resultados fueron revisados por los autores y las interpretaciones se basan en las salidas generadas por los scripts y los datos procesados.

## Reproducibilidad

Para reproducir todo el analisis desde los datos disponibles en el proyecto, ejecutar:

```bash
py "TAREA_2_HIDROLOGIA/codigo/00_ejecutar_todo.py"
```

El flujo genera:

- CSV limpios en `datos_limpios/`.
- Figuras en `figuras/`.
- Resultados de cuenca en `cuenca/`.
- Tablas de IDF y caudal teorico.
