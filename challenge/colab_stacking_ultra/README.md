# Stacking Ultra for Colab

Esta carpeta implementa la fase final de blending/stacking.

No entrena modelos base desde cero. Consume los artefactos exportados por:

- KNN cleaning
- SVM preprocessing
- signal features

y busca la mejor combinacion con:

- promedio simple
- `LogisticRegression` como meta-modelo

Flujo recomendado:

1. ejecutar antes al menos dos notebooks base y descargar sus ZIP exportados
2. subir solo este notebook a Colab
3. ejecutar las celdas iniciales
4. en `Optional: restore resume bundles` cambiar `RESTORE_RESUME_BUNDLES = True`
5. subir dos o mas ZIPs de reanudacion provenientes de los modelos base
6. verificar que el notebook descubra `oof_probabilities.csv` y `test_probabilities.csv`
7. correr la busqueda de blending y generar la submission final

El stacking no sirve si los modelos base no dejaron estos artefactos:

- `oof_probabilities.csv`
- `test_probabilities.csv`
