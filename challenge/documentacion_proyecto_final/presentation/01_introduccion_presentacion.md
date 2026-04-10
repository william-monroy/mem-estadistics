# Introduccion

## Parte 1. Contenido para presentacion

### Slide 1. Problema y contexto

- Problema: clasificacion binaria de daño superficial en engranajes solares
- Datos: 10,000 observaciones de entrenamiento y 3,000 de test
- Variables: 200 predictores numericos (`V1 ... V200`) + clase binaria
- Contexto aplicado: mantenimiento predictivo con señales de vibracion
- Metrica oficial de la competencia: `accuracy`

### Slide 2. Objetivo y enfoque general

- Objetivo: maximizar `accuracy` en Kaggle con una metodologia defendible
- Restriccion del curso: usar modelos vistos en clase o en material complementario
- Libertad clave: usar cualquier tecnica de preprocesamiento si queda bien documentada
- Estrategia seguida:
  - benchmarking de modelos base
  - profundizacion en los mas prometedores
  - cambio de enfoque hacia preprocesamiento y feature engineering
  - ensamble final por stacking

## Parte 2. Dialogo de explicacion

### Slide 1. Dialogo

Aqui conviene partir desde el problema real y no desde el algoritmo. Lo que estamos intentando detectar es si un engranaje presenta daño superficial a partir de su vibracion. Eso es importante porque en mantenimiento predictivo el valor practico no esta en clasificar por clasificar, sino en detectar fallas antes de que escalen. El dataset tenia 200 variables numericas por observacion, pero una de las ideas centrales del proyecto fue entender que esas 200 variables no debian verse solo como columnas tabulares, sino probablemente como una secuencia de señal.

Tambien es importante dejar claro desde el inicio que la competencia se evaluaba con `accuracy`. Eso nos dio un objetivo concreto para Kaggle, pero no necesariamente coincide con el mejor criterio para una aplicacion industrial real. Esa tension entre metrica competitiva y utilidad operacional reaparece en resultados y conclusiones.

### Slide 2. Dialogo

El objetivo no fue solo obtener un buen score; tambien fue llegar a una solucion que pudiera defenderse metodologicamente. Por eso no empezamos directamente con una tecnica sofisticada. Primero hicimos benchmarking de modelos base para entender la estructura del problema. Luego concentramos el esfuerzo en las familias que realmente mostraban señal. Y finalmente, cuando vimos que seguir ajustando hiperparametros ya no era suficiente, explotamos la regla mas poderosa del enunciado: la libertad de preprocesamiento. Ese cambio fue el que abrio la puerta al feature engineering de señal y despues al stacking final.

## Parte 3. Respaldo tecnico y material de apoyo

### Datos y hechos que sustentan estas slides

- Formas del dataset:
  - `training.csv`: `10000 x 202`
  - `test.csv`: `3000 x 201`
- Balance de clases:
  - `{0: 5000, 1: 5000}`
- Missing values:
  - `0` en train y `0` en test
- Duplicados exactos:
  - `0` en train

### Archivos de respaldo

- Enunciado del challenge:
  - [README.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/README.md)
- Auditoria de datos y restricciones:
  - [01_contexto_restricciones_y_datos.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final/01_contexto_restricciones_y_datos.md)
- Version refinada del reporte:
  - [01_reporte_principal_formato.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final/refined-focs/refined-docs/01_reporte_principal_formato.md)

### Decisiones metodologicas que conviene poder defender

- No hacia falta limpieza estructural fuerte del dataset.
- El problema se planteo primero como tabular supervisado para acotar familias candidatas.
- La verdadera libertad competitiva estaba en el preprocesamiento, no solo en el clasificador.

### Sugerencia visual

- Slide 1:
  - usar un diagrama simple de pipeline: `vibracion -> features -> clasificacion`
  - incluir un recuadro pequeño con `10,000 train / 3,000 test / 200 variables`
- Slide 2:
  - usar una linea de tiempo corta con 4 etapas: benchmark, seleccion, feature engineering, stacking
