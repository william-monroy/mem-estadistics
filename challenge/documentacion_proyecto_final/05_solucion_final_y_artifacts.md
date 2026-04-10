# 05. Solucion Final y Mapa de Artefactos

## 5.1 Solucion final adoptada

La solucion final del proyecto no fue un modelo individual, sino una combinacion enfocada de dos modelos base:

1. `challenge_08_knn_cleaning_colab_ultra`
2. `challenge_10_signal_features_colab_ultra`

La eleccion final se baso en tres hechos:

- `signal_features` era, con diferencia, el mejor modelo individual persistido
- `knn_cleaning` aun siendo mas debil, aportaba diversidad real
- la discrepancia entre ambos hacia plausible que un blend superara al mejor modelo individual

Esa evidencia quedo documentada en:

- [../results-pre-stack/output/challenge_12_final_stacking_colab/checkpoints/base_model_summary.csv](../results-pre-stack/output/challenge_12_final_stacking_colab/checkpoints/base_model_summary.csv)
- [../results-pre-stack/output/challenge_12_final_stacking_colab/checkpoints/pairwise_disagreement.csv](../results-pre-stack/output/challenge_12_final_stacking_colab/checkpoints/pairwise_disagreement.csv)

Valores clave:

- `signal_features`:
  - `OOF accuracy = 0.9403`
- `knn_cleaning`:
  - `OOF accuracy = 0.8145`
- tasa de desacuerdo entre ambos:
  - `0.2086`

## 5.2 Notebook final utilizada para el submission

Notebook final:

- [../colab_final_stacking/Challenge_12_Final_Stacking_Colab.ipynb](../colab_final_stacking/Challenge_12_Final_Stacking_Colab.ipynb)

Resumen final:

- [../results-pre-stack/output/challenge_12_final_stacking_colab/summary.json](../results-pre-stack/output/challenge_12_final_stacking_colab/summary.json)

Busqueda final:

- [../results-pre-stack/output/challenge_12_final_stacking_colab/checkpoints/final_stacking_search_results.csv](../results-pre-stack/output/challenge_12_final_stacking_colab/checkpoints/final_stacking_search_results.csv)

Submission final generada:

- [../results-pre-stack/submissions/challenge_12_final_stacking_colab_submission.csv](../results-pre-stack/submissions/challenge_12_final_stacking_colab_submission.csv)

## 5.3 Mejor combinacion encontrada

La mejor configuracion persistida fue:

```json
{
  "meta_family": "weighted_average",
  "weights": [0.425, 0.575],
  "best_threshold": 0.49,
  "best_meta_oof_accuracy": 0.9476,
  "best_base_models": [
    "challenge_08_knn_cleaning_colab_ultra",
    "challenge_10_signal_features_colab_ultra"
  ]
}
```

Interpretacion:

- el modelo de `signal_features` recibio mas peso porque era claramente mas fuerte
- aun asi, no recibio el `100%` del peso, porque `knn_cleaning` aportaba informacion complementaria
- bajar el umbral final a `0.49` fue mejor que dejarlo en `0.50`

## 5.4 Por que el promedio ponderado gano al meta-modelo logit

En la busqueda final se compararon:

- `weighted averages`
- `LogisticRegression` como meta-modelo

El top de resultados mostro:

- mejor ponderado:
  - `0.9476`
- mejor logit:
  - `0.9473`

La diferencia es pequena, pero suficiente para justificar una decision a favor del promedio ponderado porque:

- es mas simple
- es mas interpretable
- es mas facil de defender en una presentacion
- tiene menos riesgo de introducir sobreajuste adicional

## 5.5 Mapa de artefactos que conviene citar en el informe

### Artefactos del problema y datos

- Enunciado:
  - [../README.md](../README.md)
- Dataset:
  - [../data/training.csv](../data/training.csv)
  - [../data/test.csv](../data/test.csv)
  - [../data/sample.csv](../data/sample.csv)

### Artefactos del benchmarking inicial

- [../output/model_summary.csv](../output/model_summary.csv)
- [../output/challenge_01_knn/summary.json](../output/challenge_01_knn/summary.json)
- [../output/challenge_02_logistic_regression/summary.json](../output/challenge_02_logistic_regression/summary.json)
- [../output/challenge_03_decision_tree/summary.json](../output/challenge_03_decision_tree/summary.json)
- [../output/challenge_04_random_forest/summary.json](../output/challenge_04_random_forest/summary.json)
- [../output/challenge_05_gradient_boosting/summary.json](../output/challenge_05_gradient_boosting/summary.json)

### Artefactos de los modelos avanzados

- KNN cleaned:
  - [../results-pre-stack/knn_cleaning_ultra/output/challenge_08_knn_cleaning_colab_ultra/summary.json](../results-pre-stack/knn_cleaning_ultra/output/challenge_08_knn_cleaning_colab_ultra/summary.json)
  - [../results-pre-stack/knn_cleaning_ultra/output/challenge_08_knn_cleaning_colab_ultra/oof_probabilities.csv](../results-pre-stack/knn_cleaning_ultra/output/challenge_08_knn_cleaning_colab_ultra/oof_probabilities.csv)
  - [../results-pre-stack/knn_cleaning_ultra/output/challenge_08_knn_cleaning_colab_ultra/test_probabilities.csv](../results-pre-stack/knn_cleaning_ultra/output/challenge_08_knn_cleaning_colab_ultra/test_probabilities.csv)

- Signal features:
  - [../results-pre-stack/signal_features_ultra/output/challenge_10_signal_features_colab_ultra/summary.json](../results-pre-stack/signal_features_ultra/output/challenge_10_signal_features_colab_ultra/summary.json)
  - [../results-pre-stack/signal_features_ultra/output/challenge_10_signal_features_colab_ultra/oof_probabilities.csv](../results-pre-stack/signal_features_ultra/output/challenge_10_signal_features_colab_ultra/oof_probabilities.csv)
  - [../results-pre-stack/signal_features_ultra/output/challenge_10_signal_features_colab_ultra/test_probabilities.csv](../results-pre-stack/signal_features_ultra/output/challenge_10_signal_features_colab_ultra/test_probabilities.csv)

- Final stacking:
  - [../results-pre-stack/output/challenge_12_final_stacking_colab/summary.json](../results-pre-stack/output/challenge_12_final_stacking_colab/summary.json)
  - [../results-pre-stack/output/challenge_12_final_stacking_colab/meta_oof_probabilities.csv](../results-pre-stack/output/challenge_12_final_stacking_colab/meta_oof_probabilities.csv)
  - [../results-pre-stack/output/challenge_12_final_stacking_colab/meta_test_probabilities.csv](../results-pre-stack/output/challenge_12_final_stacking_colab/meta_test_probabilities.csv)
  - [../results-pre-stack/submissions/challenge_12_final_stacking_colab_submission.csv](../results-pre-stack/submissions/challenge_12_final_stacking_colab_submission.csv)

## 5.6 Texto corto sugerido para una conclusion ejecutiva

Se comenzo tratando el problema como clasificacion tabular clasica y se probaron modelos lineales, vecinos cercanos, arboles y ensembles. Los resultados mostraron que KNN y SVM capturaban mejor la estructura no lineal de los datos que los arboles. A partir de ahi, el proyecto cambio de foco: en vez de seguir ampliando solo la busqueda de hiperparametros, se exploto la libertad de preprocesamiento permitida por el curso. El mayor salto llego al reinterpretar `V1 ... V200` como una senal y construir features temporales, frecuenciales y por segmentos. Finalmente, la mejor solucion no fue un modelo unico, sino un stacking simple y robusto que combino el modelo de `signal features` con un KNN complementario. Esa solucion alcanzo el accuracy objetivo esperado en Kaggle.

## 5.7 Campos que conviene completar manualmente antes de la entrega final

El repositorio ya deja la mayor parte del trabajo documentado, pero hay tres datos que conviene completar manualmente en la version final del informe o presentacion:

1. `Public leaderboard final del stacking`:
   - completar con el valor exacto observado en Kaggle
2. `Posicion relativa final del equipo`:
   - si se quiere citar el ranking en el leaderboard
3. `Private leaderboard final`:
   - una vez que el concurso cierre y este disponible
