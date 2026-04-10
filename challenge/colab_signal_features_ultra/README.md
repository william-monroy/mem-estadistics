# Signal Features Ultra for Colab

Esta carpeta implementa la linea `signal feature engineering`.

El supuesto central es que `V1 ... V200` si representan una secuencia ordenada de señal.

La notebook construye:

- features temporales
- features frecuenciales
- features por segmentos
- una variante hibrida con `raw PCA`

y compara `KNN` y `SVM` sobre esas representaciones.

Flujo recomendado:

1. subir solo el notebook a Colab
2. usar runtime CPU
3. ejecutar las celdas iniciales
4. en la celda de upload poner `UPLOAD_DATA_FILES = True`
5. subir `training.csv`, `test.csv` y `sample.csv`
6. ejecutar `Stage 1` para identificar si las features de senal muestran ventaja real
7. continuar con `Stage 2` para confirmar estabilidad
8. exportar y descargar el ZIP al terminar

Artefactos finales esperados:

- `summary.json`
- `oof_probabilities.csv`
- `test_probabilities.csv`
- `submissions/challenge_10_signal_features_colab_ultra_submission.csv`
