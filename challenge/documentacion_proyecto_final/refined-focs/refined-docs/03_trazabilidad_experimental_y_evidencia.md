# Trazabilidad Experimental y Evidencia

Este documento ordena el proyecto como una secuencia de etapas con evidencia cuantitativa. La idea no es solo enumerar notebooks, sino mostrar como cada resultado cambio la decision siguiente.

## 1. Estado inicial del problema

Artefactos de referencia:

- [README.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/README.md)
- [01_contexto_restricciones_y_datos.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final/01_contexto_restricciones_y_datos.md)

Datos auditados:

| Variable | Resultado |
| --- | --- |
| `training.csv` | `10000 x 202` |
| `test.csv` | `3000 x 201` |
| distribucion de `class` | `{0: 5000, 1: 5000}` |
| missing en train | `0` |
| missing en test | `0` |
| duplicados exactos en train | `0` |

**Decision que habilito esta evidencia**  
No habia que invertir el grueso del esfuerzo en limpieza estructural. El proyecto podia centrarse en representacion y modelado.

## 2. Fase de benchmarking inicial

Artefacto central:

- [model_summary.csv](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/output/model_summary.csv)

Resultados:

| Modelo | Mejor CV local | Accuracy de validacion |
| --- | ---: | ---: |
| K-nearest neighbors | `0.754625` | `0.7845` |
| Random Forest | `0.719374` | `0.7280` |
| Gradient Boosting | `0.634500` | `0.6305` |
| Decision Tree | `0.595000` | `0.6110` |
| Logistic Regression | `0.562375` | `0.5690` |

**Lectura**

- KNN fue el mejor baseline.
- Random Forest quedo segundo, pero con una brecha visible.
- Las familias lineales y los arboles individuales quedaron fuera de la ruta principal.

**Decision siguiente**

- profundizar en KNN;
- mantener Random Forest como linea secundaria;
- dejar de invertir tiempo fuerte en Logistic, Decision Tree y Gradient Boosting.

## 3. Fase KNN: de baseline a tuned

### 3.1 KNN baseline

Artefacto:

- [challenge_01_knn/summary.json](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/output/challenge_01_knn/summary.json)

Configuracion persistida:

```json
{
  "model__n_neighbors": 5,
  "model__weights": "distance",
  "pca__n_components": 50
}
```

Resultado:

- `validation_accuracy = 0.7845`

### 3.2 KNN ready

Artefacto:

- [Challenge_01_KNN_Ready.ipynb](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/Challenge_01_KNN_Ready.ipynb)

Region fijada:

```python
("scaler", StandardScaler()),
("pca", PCA(n_components=48, random_state=RANDOM_STATE)),
("model", KNeighborsClassifier(n_neighbors=4, weights="distance", metric="manhattan"))
```

**Lectura**

Aqui aparecio por primera vez el patron que luego se repetiria:

- `PCA` intermedio
- `k` bajo
- `weights = distance`
- `metric = manhattan`

### 3.3 KNN tuned

Artefactos:

- [Challenge_01_KNN_Tuned.ipynb](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/Challenge_01_KNN_Tuned.ipynb)
- [challenge_01_knn_tuned/summary.json](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/output/challenge_01_knn_tuned/summary.json)

Resultado:

- `validation_accuracy = 0.822`

Mejor region fina observada:

- `metric = manhattan`
- `n_neighbors = 4`
- `weights = distance`
- `pca__n_components` alrededor de `44-48`

**Decision siguiente**

KNN merecia una linea avanzada propia. Ya no era solo un baseline; era una familia seria.

## 4. Evidencia temprana de Kaggle

Public scores observados durante la exploracion:

| Submission | Public score |
| --- | ---: |
| `challenge_01_knn_submission.csv` | `0.80611` |
| `challenge_01_knn_tuned_submission.csv` | `0.83777` |
| `svm_colab_submission.csv` | `0.84166` |
| `random_forest_colab_ultra_submission.csv` | `0.74722` |

**Lectura**

- KNN tuned generalizaba bien hacia Kaggle.
- SVM se volvia la nueva referencia a vencer.
- Random Forest quedaba atras incluso en leaderboard publico.

## 5. Fase Colab Ultra: Random Forest

Artefactos:

- [Challenge_04_RandomForest_Colab_Ultra.ipynb](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_random_forest_ultra/Challenge_04_RandomForest_Colab_Ultra.ipynb)
- [summary.json](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results/challenge_04_random_forest_colab_ultra_resume/output/challenge_04_random_forest_colab_ultra/summary.json)

Mejores parametros:

```json
{
  "criterion": "gini",
  "max_depth": 40,
  "max_features": 0.35,
  "max_samples": null,
  "min_samples_leaf": 1,
  "min_samples_split": 2
}
```

Resultados:

- `best_cv_accuracy = 0.739875`
- `validation_accuracy = 0.757`

**Decision siguiente**

La linea se consideró cerrada. Aunque funcional, no justificaba seguir compitiendo contra KNN y SVM.

## 6. Fase Colab Ultra: SVM

Artefactos:

- [Challenge_06_SVM_Colab_Ultra.ipynb](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_svm_ultra/Challenge_06_SVM_Colab_Ultra.ipynb)
- [summary.json](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results/challenge_06_svm_colab_ultra_resume/output/challenge_06_svm_colab_ultra/summary.json)

Mejor configuracion:

```json
{
  "model__C": 3.0,
  "model__gamma": 0.01,
  "pca__n_components": 128
}
```

Resultados:

- `best_search_stage_score = 0.8028333333333334`
- `validation_accuracy = 0.841`
- `public Kaggle = 0.84166`

**Decision siguiente**

SVM quedo validado como familia fuerte, pero la pregunta se desplazó desde "seguir moviendo hiperparametros" hacia "mejorar la representacion del espacio de entrada".

## 7. Fase avanzada: KNN cleaning

Artefactos:

- [Challenge_08_KNN_Cleaning_Colab_Ultra.ipynb](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_knn_cleaning_ultra/Challenge_08_KNN_Cleaning_Colab_Ultra.ipynb)
- [summary.json](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results-pre-stack/knn_cleaning_ultra/output/challenge_08_knn_cleaning_colab_ultra/summary.json)

Mejor configuracion persistida:

```json
{
  "clean__method": "none",
  "model__metric": "manhattan",
  "model__n_neighbors": 2,
  "model__weights": "distance",
  "pca__n_components": 64,
  "scale__method": "standard"
}
```

Resultados:

- `validation_accuracy = 0.817`
- `oof_accuracy = 0.8145`

**Lectura**

La libreta fue util, pero no gano por accuracy individual. Su valor principal fue aportar una segunda fuente de error distinta frente al modelo de features de señal.

## 8. Fase avanzada: signal features

Artefactos:

- [Challenge_10_Signal_Features_Colab_Ultra.ipynb](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_signal_features_ultra/Challenge_10_Signal_Features_Colab_Ultra.ipynb)
- [summary.json](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results-pre-stack/signal_features_ultra/output/challenge_10_signal_features_colab_ultra/summary.json)

Mejor configuracion:

```json
{
  "feature__raw_pca": 32,
  "feature__set": "all",
  "model__family": "svm",
  "model__C": 6.0,
  "model__gamma": 0.01,
  "scale__method": "standard"
}
```

Resultados:

- `validation_accuracy = 0.9445`
- `oof_accuracy = 0.9403`

**Lectura**

Este fue el salto decisivo del proyecto. La representacion del problema cambio mas de lo que habia logrado cualquier ajuste puro de hiperparametros.

**Decision siguiente**

Signal features paso a ser el nuevo modelo lider y a la vez el candidato principal para combinarse en un ensamble final.

## 9. Fase exploratoria incompleta: SVM preprocessing

### 9.1 Lo que si quedo consolidado en Stage 2

Artefacto:

- [stage2_cv_summary.csv](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results-pre-stack/svm_preprocessing/output/challenge_09_svm_preprocessing_colab_ultra/checkpoints/stage2_cv_summary.csv)

Top 3 de `Stage 2`:

| Orden | Cleaning | Scale | PCA | C | gamma | CV mean |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `lof_0.03` | `robust` | `120` | `4.0` | `0.02` | `0.789376` |
| 2 | `lof_0.03` | `power_yeo` | `136` | `3.0` | `0.01` | `0.789125` |
| 3 | `none` | `power_yeo` | `144` | `4.0` | `0.01` | `0.788501` |

### 9.2 Lo que se aprendio del Stage 3 parcial

Artefactos:

- [stage3_local_candidates.json](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results-pre-stack/svm_preprocessing/output/challenge_09_svm_preprocessing_colab_ultra/checkpoints/stage3_local_candidates.json)
- [stage3_local_cv_fold_results(1).csv](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results-pre-stack/svm_preprocessing/output/challenge_09_svm_preprocessing_colab_ultra/checkpoints/stage3_local_cv_fold_results(1).csv)

Hechos relevantes:

- `2832` candidatos en `Stage 3`
- `14160` fits potenciales con `5` folds
- solo se lograron `1235` evaluaciones parciales antes de detener la corrida
- todas esas evaluaciones correspondian a `fold_idx = 0`

Top parcial observado en ese fold:

| Candidate | Seed | Cleaning | Scale | PCA | C | gamma | Fold accuracy |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `stage3_1017` | `seed_1` | `lof_0.03` | `power_yeo` | `152` | `3.0` | `0.01` | `0.820000` |
| `stage3_756` | `seed_0` | `none` | `quantile_normal` | `120` | `4.0` | `0.01` | `0.819375` |
| `stage3_1221` | `seed_1` | `lof_0.03` | `standard` | `152` | `3.0` | `0.01` | `0.819375` |

**Lectura**

La region prometedora se entendio mejor, pero no habia CV completo suficiente para convertir esto en decision final confiable. Por eso SVM preprocessing quedo como evidencia exploratoria, no como submission ganador.

## 10. Fase final: stacking

Artefactos:

- [Challenge_12_Final_Stacking_Colab.ipynb](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_final_stacking/Challenge_12_Final_Stacking_Colab.ipynb)
- [summary.json](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results-pre-stack/output/challenge_12_final_stacking_colab/summary.json)
- [final_stacking_search_results.csv](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results-pre-stack/output/challenge_12_final_stacking_colab/checkpoints/final_stacking_search_results.csv)

Modelos base usados:

- `challenge_08_knn_cleaning_colab_ultra`
- `challenge_10_signal_features_colab_ultra`

Evidencia de diversidad:

- `disagreement_rate = 0.2086`

Top del search final:

| Orden | Meta family | Parametros | Threshold | Meta OOF |
| --- | --- | --- | ---: | ---: |
| 1 | `weighted_average` | `weights = [0.425, 0.575]` | `0.49` | `0.9476` |
| 2 | `weighted_average` | `weights = [0.375, 0.625]` | `0.52` | `0.9474` |
| 3 | `weighted_average` | `weights = [0.35, 0.65]` | `0.54` | `0.9474` |
| 4 | `logreg` | `C = 1.0` | `0.65` | `0.9473` |

**Lectura**

- el mejor modelo final fue un ensamble simple, no un meta-modelo complejo;
- el peso de `signal_features` tenia que ser mayor, pero no exclusivo;
- `KNN cleaning` aportaba suficiente diversidad como para mejorar sobre el uso de `signal_features` sola.

## 11. Cierre del proyecto

Submission final generado:

- [challenge_12_final_stacking_colab_submission.csv](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results-pre-stack/submissions/challenge_12_final_stacking_colab_submission.csv)

Hecho cualitativo confirmado por la ejecucion final:

- el stacking final alcanzo el objetivo de accuracy buscado en Kaggle

Campo que debe completarse manualmente en el informe:

- valor exacto del `public leaderboard` final
- valor exacto del `private leaderboard` al cierre

## 12. Resumen ejecutivo de la trayectoria experimental

La secuencia final fue:

1. benchmark base para encontrar familias candidatas;
2. consolidacion de KNN y SVM como lineas fuertes;
3. descarte de Random Forest como ruta principal por baja rentabilidad experimental;
4. explotacion de la libertad de preprocesamiento;
5. descubrimiento de que la mejor mejora venia de representar mejor la señal;
6. ensamble final con solo dos modelos realmente competitivos y complementarios.

Esa trazabilidad es importante porque permite defender que el submission ganador no fue un hallazgo accidental, sino el resultado de una secuencia de decisiones consistentes con la evidencia observada.
