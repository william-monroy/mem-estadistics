# Metodologia: Preprocesamiento

## Parte 1. Contenido para presentacion

### Slide 1. Preprocesamiento base y protocolo de validacion

- Split principal: `80%` train y `20%` validation, estratificado
- Validacion adicional: `5-fold CV` para OOF y comparacion robusta
- Pipeline base:
  - `StandardScaler`
  - `PCA`
  - modelo clasificador
- Se probaron tecnicas avanzadas permitidas por el curso:
  - `LOF`
  - `Tomek Links`
  - `ENN / RENN`
  - transformaciones `Robust`, `Quantile`, `Power`

### Slide 2. Preprocesamiento que cambio el proyecto

- Hipotesis clave: `V1 ... V200` representan una señal de vibracion
- Se construyeron features de tres tipos:
  - temporales
  - frecuenciales con FFT
  - por segmentos
- Se agrego `raw PCA = 32` como informacion complementaria
- Resultado: el mayor salto de accuracy vino de cambiar la representacion, no de seguir tunning modelos tabulares

## Parte 2. Dialogo de explicacion

### Slide 1. Dialogo

Primero hay que explicar que el preprocesamiento no fue un detalle secundario. En todas las familias usamos un split principal de 80/20 estratificado para tener una validacion consistente y, ademas, usamos validacion cruzada para las decisiones realmente importantes. Eso fue clave porque el leaderboard de Kaggle por si solo no era suficiente para decidir. Tambien empezamos con un pipeline base muy convencional, con escalado y PCA, porque necesitabamos una referencia comun entre modelos.

En esa misma etapa probamos tecnicas externas a lo visto estrictamente en clase, pero permitidas por la regla del proyecto: detectores de outliers como LOF, limpieza de instancias como Tomek y ENN/RENN, y transformaciones de escala mas sofisticadas. La razon de probarlas era metodologica: si el problema estaba siendo limitado por la geometria del espacio de entrada, entonces habia que intervenir esa geometria, no solo cambiar de clasificador.

### Slide 2. Dialogo

El verdadero pivote del proyecto fue dejar de ver `V1 ... V200` como una matriz tabular cualquiera y empezar a tratarlas como una señal de vibracion. Eso nos llevo a construir features temporales, que resumen intensidad y forma de la señal; features frecuenciales, que capturan como se distribuye la energia en el espectro; y features por segmentos, que conservan informacion local de distintas partes de la ventana.

Ademas, no quisimos perder completamente la informacion cruda, por eso añadimos una pequeña proyeccion `raw PCA`. El hallazgo importante fue que esta mezcla de descripcion fisica mas una pequeña compresion de la señal original supero claramente a cualquier pipeline tabular que habiamos probado antes.

## Parte 3. Respaldo tecnico y material de apoyo

### Hechos tecnicos que sustentan las slides

- Split principal:
  - `VALID_SIZE = 0.20`
  - fuente: scripts generadores de notebooks `Ultra`
- OOF:
  - en `signal_features` y `final_stacking` se uso `StratifiedKFold(n_splits=5)`
- Tecnicas avanzadas efectivamente probadas:
  - LOF por clase
  - Tomek Links
  - ENN
  - RENN
  - `RobustScaler`
  - `QuantileTransformer`
  - `PowerTransformer`

### Archivos de respaldo

- Script generador con `VALID_SIZE = 0.20`:
  - [generate_colab_advanced_strategies.py](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/scripts/generate_colab_advanced_strategies.py)
- Notebook de KNN cleaning:
  - [Challenge_08_KNN_Cleaning_Colab_Ultra.ipynb](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_knn_cleaning_ultra/Challenge_08_KNN_Cleaning_Colab_Ultra.ipynb)
- Notebook de signal features:
  - [Challenge_10_Signal_Features_Colab_Ultra.ipynb](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_signal_features_ultra/Challenge_10_Signal_Features_Colab_Ultra.ipynb)
- Explicacion detallada:
  - [02_decisiones_metodologicas_detalladas.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final/refined-focs/refined-docs/02_decisiones_metodologicas_detalladas.md)
  - [04_apendice_codigo_y_diseno_tecnico.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final/refined-focs/refined-docs/04_apendice_codigo_y_diseno_tecnico.md)

### Bloques de codigo que conviene tener presentes si preguntan

- `apply_cleaning(...)` para LOF por clase
- `build_feature_bank(...)` para features temporales y frecuenciales
- concatenacion de `feature__raw_pca = 32`

### Sugerencia visual

- Slide 1:
  - tabla corta con `80/20 split`, `5-fold CV`, `PCA`, `LOF/Tomek/ENN`
- Slide 2:
  - un diagrama en capas:
    - señal cruda
    - features temporales
    - FFT y bandas
    - features por segmentos
    - raw PCA adicional
