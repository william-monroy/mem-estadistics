# KNN Cleaning Ultra for Colab

Esta carpeta implementa la linea `KNN + limpieza de instancias`.

El objetivo es mejorar KNN atacando directamente puntos ruidosos o ambiguos, usando tecnicas compatibles con la regla de preprocesamiento libre.

Incluye:

- screening por etapas
- checkpoints
- ZIP de reanudacion
- export de `OOF probabilities` y `test probabilities` para stacking

Flujo recomendado:

1. subir solo el notebook a Google Colab
2. usar runtime CPU
3. ejecutar las celdas iniciales
4. en `Optional: upload the CSV files manually` cambiar `UPLOAD_DATA_FILES = True`
5. subir `training.csv`, `test.csv` y `sample.csv`
6. dejar `SEARCH_PROFILE = "balanced"` si estas en Colab free o `aggressive` si tienes mas margen
7. correr hasta terminar `Stage 2`
8. exportar y descargar el ZIP de reanudacion
9. si la sesion se corta, abrir una nueva, restaurar el ZIP y continuar con `Stage 3` y la fase final

Artefactos finales esperados:

- `summary.json`
- `oof_probabilities.csv`
- `test_probabilities.csv`
- `submissions/challenge_08_knn_cleaning_colab_ultra_submission.csv`
