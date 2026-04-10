# SVM Preprocessing Ultra for Colab

Esta carpeta implementa la linea `SVM + preprocesamiento`.

La idea es dejar fija la region buena de `SVM` y buscar mejoras via:

- escalado
- transformaciones de distribucion
- limpieza suave de instancias

Tambien exporta artefactos de probabilidades para stacking.

Flujo recomendado:

1. subir solo el notebook a Colab
2. usar runtime CPU
3. ejecutar las celdas iniciales
4. en la celda de upload poner `UPLOAD_DATA_FILES = True`
5. subir `training.csv`, `test.csv` y `sample.csv`
6. dejar `SEARCH_PROFILE = "balanced"` en Colab free si quieres una corrida inicial mas segura
7. correr `Stage 1` y `Stage 2`
8. descargar el ZIP exportado antes de cerrar la sesion
9. restaurar ese ZIP si luego quieres completar `Stage 3` y la fase final

Artefactos finales esperados:

- `summary.json`
- `oof_probabilities.csv`
- `test_probabilities.csv`
- `submissions/challenge_09_svm_preprocessing_colab_ultra_submission.csv`
