# 02. Bitacora Experimental y Toma de Decisiones

## 2.1 Fase 1: benchmarking inicial con familias clasicas

La primera fase tuvo un objetivo simple: responder rapidamente que familias merecian mas presupuesto de computo.

Se construyeron notebooks base siguiendo la estructura del curso:

- `KNN`: [../Challenge_01_KNN.ipynb](../Challenge_01_KNN.ipynb)
- `Logistic Regression`: [../Challenge_02_LogisticRegression.ipynb](../Challenge_02_LogisticRegression.ipynb)
- `Decision Tree`: [../Challenge_03_DecisionTree.ipynb](../Challenge_03_DecisionTree.ipynb)
- `Random Forest`: [../Challenge_04_RandomForest.ipynb](../Challenge_04_RandomForest.ipynb)
- `Gradient Boosting`: [../Challenge_05_GradientBoosting.ipynb](../Challenge_05_GradientBoosting.ipynb)

La comparacion consolidada quedo en [../output/model_summary.csv](../output/model_summary.csv).

Resultados locales de esa primera ronda:

| Modelo | Validation accuracy | Mejor CV local aproximado |
| --- | ---: | ---: |
| K-nearest neighbors | `0.7845` | `0.754625` |
| Random forest | `0.7280` | `0.719374` |
| Gradient boosting | `0.6305` | `0.634500` |
| Decision tree | `0.6110` | `0.595000` |
| Logistic regression | `0.5690` | `0.562375` |

### Decision tomada

La conclusion fue inmediata:

- `Logistic Regression` quedo descartado como modelo final.
- `Decision Tree` no tenia capacidad suficiente.
- `Gradient Boosting` no mostraba una mejora que justificara mas inversion.
- `Random Forest` era competitivo, pero no lider.
- `KNN` paso a ser la primera familia prioritaria.

## 2.2 Fase 2: notebooks Ready para reducir costo de iteracion

Como varias busquedas iniciales tardaban demasiado, se creo una segunda capa de notebooks `Ready` con mejores parametros ya fijados, para correr manualmente sin repetir el `GridSearchCV` completo.

Ejemplos:

- [../Challenge_01_KNN_Ready.ipynb](../Challenge_01_KNN_Ready.ipynb)
- [../Challenge_04_RandomForest_Ready.ipynb](../Challenge_04_RandomForest_Ready.ipynb)
- [../Challenge_05_GradientBoosting_Ready.ipynb](../Challenge_05_GradientBoosting_Ready.ipynb)

Esto sirvio para separar dos tareas:

- busqueda de hiperparametros
- entrenamiento final reproducible

En esta fase ya se vio que la region KNN buena se mantenia alrededor de:

- `PCA` en la zona `44-50`
- `k` bajo
- `weights = distance`
- `metric = manhattan`

## 2.3 Fase 3: tuning mas agresivo sobre las familias que seguian vivas

Despues de las corridas `Ready`, el proyecto se concentro en:

- `KNN`
- `Random Forest`
- `Gradient Boosting`

Se buscaba si algun ajuste mas agresivo podia acercar el accuracy a un rango ya competitivo para Kaggle.

### Lo que ocurrio en la practica

- `KNN` si respondio bien al tuning.
- `Random Forest` mejoro solo de forma limitada.
- `Gradient Boosting` no mostro un salto importante.

El mejor KNN afinado quedo reflejado por dos artefactos:

- [../Challenge_01_KNN_Tuned.ipynb](../Challenge_01_KNN_Tuned.ipynb)
- [../output/challenge_01_knn_tuned/summary.json](../output/challenge_01_knn_tuned/summary.json)

El resumen local mas importante fue:

- `validation_accuracy = 0.822`

Durante esa fase tambien se consolidaron dos observaciones:

1. La metrica Manhattan y pesos por distancia eran recurrentemente competitivos para KNN.
2. Los ensembles de arboles no estaban capturando la estructura del problema tan bien como las familias basadas en vecindad o margenes no lineales.

## 2.4 Fase 4: migracion a notebooks "Colab Ultra" resumibles

Los tiempos de entrenamiento empezaron a volverse el principal cuello de botella. Para resolver esto se construyeron variantes de Colab con:

- ejecucion por etapas
- checkpoints
- ZIPs de reanudacion
- estructura compatible con sesiones temporales

Carpetas relevantes:

- [../colab_random_forest_ultra](../colab_random_forest_ultra)
- [../colab_knn_ultra](../colab_knn_ultra)
- [../colab_svm_ultra](../colab_svm_ultra)
- [../colab_bagging_ultra](../colab_bagging_ultra)

En este punto se hizo una observacion decisiva:

### `Random Forest` no estaba cerrando la brecha

La corrida avanzada de Random Forest quedo resumida en:

- [../results/challenge_04_random_forest_colab_ultra_resume/output/challenge_04_random_forest_colab_ultra/summary.json](../results/challenge_04_random_forest_colab_ultra_resume/output/challenge_04_random_forest_colab_ultra/summary.json)

Resultado:

- `validation_accuracy = 0.757`
- `best_cv_accuracy = 0.739875`

La conclusion fue que seguir empujando arboles no iba a llevar al objetivo.

### `SVM` aparecio como linea realmente competitiva

La variante avanzada de SVM quedo en:

- [../results/challenge_06_svm_colab_ultra_resume/output/challenge_06_svm_colab_ultra/summary.json](../results/challenge_06_svm_colab_ultra_resume/output/challenge_06_svm_colab_ultra/summary.json)

Resultado:

- `validation_accuracy = 0.841`
- mejor region:
  - `C = 3.0`
  - `gamma = 0.01`
  - `PCA = 128`

En Kaggle, durante esa etapa, se observaron public scores manuales:

| Submission | Public score observado |
| --- | ---: |
| `svm_colab_submission.csv` | `0.84166` |
| `challenge_01_knn_tuned_submission.csv` | `0.83777` |
| `challenge_01_knn_submission.csv` | `0.80611` |
| `random_forest_colab_ultra_submission.csv` | `0.74722` |

### Decision tomada

La lectura fue clara:

- `SVM` y `KNN` eran las familias vivas.
- `Random Forest` quedaba relegado a rol secundario.
- El problema parecia premiar:
  - geometria local
  - fronteras no lineales
  - representaciones mejores de la senal

## 2.5 Fase 5: cambio de estrategia, del tuning al preprocessing

Despues de varias iteraciones, el proyecto dejo de perseguir mejoras solo mediante hiperparametros y se replanteo una pregunta mas profunda:

> Si la regla del curso permite cualquier tecnica de preprocesamiento bien documentada, por que seguir tratando la base como tabular generica?

De ahi salieron tres nuevas lineas:

1. `KNN + cleaning`
2. `SVM + preprocessing`
3. `signal feature engineering`

Estas lineas se materializaron en:

- [../colab_knn_cleaning_ultra](../colab_knn_cleaning_ultra)
- [../colab_svm_preprocessing_ultra](../colab_svm_preprocessing_ultra)
- [../colab_signal_features_ultra](../colab_signal_features_ultra)

## 2.6 Fase 6: evidencia fuerte a favor de `signal feature engineering`

La notebook que cambio la trayectoria del proyecto fue:

- [../colab_signal_features_ultra/Challenge_10_Signal_Features_Colab_Ultra.ipynb](../colab_signal_features_ultra/Challenge_10_Signal_Features_Colab_Ultra.ipynb)

Su resumen quedo en:

- [../results-pre-stack/signal_features_ultra/output/challenge_10_signal_features_colab_ultra/summary.json](../results-pre-stack/signal_features_ultra/output/challenge_10_signal_features_colab_ultra/summary.json)

Resultado:

- `validation_accuracy = 0.9445`
- `oof_accuracy = 0.9403`
- mejor configuracion:
  - feature set `all`
  - `raw PCA = 32`
  - familia `svm`
  - `C = 6.0`
  - `gamma = 0.01`
  - `scale = standard`

Esta fue la primera evidencia fuerte de que el orden y la estructura de la senal importaban mucho mas de lo que sugeria la representacion tabular original.

## 2.7 Fase 7: KNN cleaning como modelo complementario, no dominante

La linea de limpieza de instancias nacio porque pruebas exploratorias previas sugerian que remover puntos ruidosos podia ayudar a KNN.

Resultado final:

- [../results-pre-stack/knn_cleaning_ultra/output/challenge_08_knn_cleaning_colab_ultra/summary.json](../results-pre-stack/knn_cleaning_ultra/output/challenge_08_knn_cleaning_colab_ultra/summary.json)

Metricas:

- `validation_accuracy = 0.817`
- `oof_accuracy = 0.8145`

Algo importante ocurrio aqui:

- la hipotesis de limpieza estaba bien planteada
- pero el mejor candidato final de esa notebook no uso limpieza explicita

Mejores parametros persistidos:

- `clean__method = none`
- `metric = manhattan`
- `n_neighbors = 2`
- `weights = distance`
- `pca__n_components = 64`
- `scale = standard`

### Interpretacion

Esta linea no gano sola, pero fue muy util por dos razones:

1. mantuvo a KNN como modelo competitivo
2. aporto diversidad real frente al modelo de `signal features`

## 2.8 Fase 8: SVM preprocessing se volvio demasiado costoso para cerrarlo completo

La notebook:

- [../colab_svm_preprocessing_ultra/Challenge_09_SVM_Preprocessing_Colab_Ultra.ipynb](../colab_svm_preprocessing_ultra/Challenge_09_SVM_Preprocessing_Colab_Ultra.ipynb)

si alcanzo a completar:

- `Stage 1`
- `Stage 2`

pero `Stage 3` crecio demasiado:

- `2832` candidatos
- `5` folds por candidato
- `14160` fits

En ese punto el proyecto cambio de criterio:

- en vez de forzar el cierre completo de SVM preprocessing
- se decidio pasar a stacking con los dos modelos ya fuertes y completos

La evidencia parcial de SVM preprocessing siguio siendo util para aprendizaje, pero no fue necesaria para el submission final.

## 2.9 Fase 9: stacking final enfocado

Con `KNN cleaning` y `signal features` ya completos, se construyo un stacking final especifico:

- [../colab_final_stacking/Challenge_12_Final_Stacking_Colab.ipynb](../colab_final_stacking/Challenge_12_Final_Stacking_Colab.ipynb)

Este stacking no reentrena modelos base. Solo consume:

- `oof_probabilities.csv`
- `test_probabilities.csv`

de las corridas previas.

La salida final quedo en:

- [../results-pre-stack/output/challenge_12_final_stacking_colab/summary.json](../results-pre-stack/output/challenge_12_final_stacking_colab/summary.json)

Mejor solucion encontrada:

- base models:
  - `challenge_08_knn_cleaning_colab_ultra`
  - `challenge_10_signal_features_colab_ultra`
- meta-familia:
  - `weighted_average`
- pesos:
  - `0.425`
  - `0.575`
- threshold:
  - `0.49`
- `best_meta_oof_accuracy = 0.9476`

## 2.10 Decision final

La ruta ganadora del proyecto no fue:

- el mejor arbol
- el mejor ensemble de arboles
- el SVM mas pesado

La ruta ganadora fue:

1. descubrir con benchmarking que `KNN` y `SVM` eran las familias prometedoras
2. reconocer que el verdadero salto estaba en la representacion de la senal
3. construir un modelo muy fuerte con `signal feature engineering`
4. combinarlo con un modelo complementario de KNN
5. cerrar con un stacking simple y robusto
