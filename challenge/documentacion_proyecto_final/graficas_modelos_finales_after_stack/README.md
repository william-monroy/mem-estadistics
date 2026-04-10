# Graficas Modelos Finales After Stack

Version visual: `dark mode`

- fondo negro
- texto blanco
- paletas en tonos morados cuando aplica

Esta carpeta contiene graficas nuevas generadas a partir de:

- `challenge/results-after-stack/challenge_12_final_stacking_colab_resume/output/...`
- los public scores observados manualmente en Kaggle

Modelos cubiertos:

- `signal_features`
- `knn_cleaning`
- `final_stacking`

## Estructura

- `comparison/00_model_score_comparison.png`
- `signal_features/`
- `knn_cleaning/`
- `final_stacking/`

## Notas metodologicas

- Para `signal_features` y `knn_cleaning`, la confusion matrix y la distribucion de probabilidades se construyeron con `oof_probabilities.csv`.
- Para `final_stacking`, se uso `meta_oof_probabilities.csv`.
- Para `signal_features` y `knn_cleaning`, `pred` usa el threshold persistido en el artefacto OOF.
- Para `final_stacking`, `pred` ya esta persistido con el mejor threshold encontrado (`0.49`).
- La figura `06_signal_group_visualization_pca2d.png` es una visualizacion aproximada del mejor modelo de `signal_features` en 2D usando el espacio de features ingenierizadas y una proyeccion PCA.
- La figura `06_knn_group_visualization_pca2d.png` es una visualizacion aproximada del comportamiento de KNN en 2D usando PCA. No representa exactamente la frontera del modelo final en 64 dimensiones.
- La figura `06_stacking_meta_space_visualization.png` es una visualizacion exacta del modelo final de stacking en el plano de probabilidades de sus dos modelos base.
- Los public scores usados en estas graficas son:
  - `signal_features = 0.95000`
  - `knn_cleaning = 0.83888`
  - `final_stacking = 0.96277`

## Resumen numerico

```csv
model,validation_accuracy,oof_accuracy,public_score,meta_oof_accuracy
signal_features,0.9445,0.9403,0.95,
knn_cleaning,0.817,0.8145,0.83888,
final_stacking,,,0.96277,0.9476
```
