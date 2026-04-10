# 03. Modelos, Parametros y Resultados

## 3.1 Tabla consolidada de modelos principales

### Modelos base y tuned tempranos

| Modelo | Artefacto principal | Parametros destacados | Resultado local | Lectura |
| --- | --- | --- | ---: | --- |
| Logistic regression | [../output/challenge_02_logistic_regression/summary.json](../output/challenge_02_logistic_regression/summary.json) | `C=1.0`, `solver=lbfgs` | `0.569` | Insuficiente. Frontera lineal demasiado rigida. |
| Decision tree | [../output/challenge_03_decision_tree/summary.json](../output/challenge_03_decision_tree/summary.json) | `criterion=gini`, `max_depth=12`, `min_samples_leaf=5` | `0.611` | No generaliza lo bastante. |
| Gradient boosting | [../output/challenge_05_gradient_boosting/summary.json](../output/challenge_05_gradient_boosting/summary.json) | `learning_rate=0.05`, `max_depth=2`, `n_estimators=100`, `subsample=0.7` | `0.6305` | Mejor que arbol simple, pero lejos del objetivo. |
| Random forest | [../output/challenge_04_random_forest/summary.json](../output/challenge_04_random_forest/summary.json) | `n_estimators=100`, `max_depth=None`, `min_samples_leaf=2`, `max_features=sqrt` | `0.728` | Competitivo como baseline, pero no lider. |
| KNN baseline | [../output/challenge_01_knn/summary.json](../output/challenge_01_knn/summary.json) | `k=5`, `weights=distance`, `PCA=50` | `0.7845` | Primera familia realmente prometedora. |
| KNN ready | [../Challenge_01_KNN_Ready.ipynb](../Challenge_01_KNN_Ready.ipynb) | `PCA=48`, `k=4`, `weights=distance`, `metric=manhattan` | notebook fijo para corrida | Consolidacion de la region buena de KNN. |
| KNN tuned | [../output/challenge_01_knn_tuned/summary.json](../output/challenge_01_knn_tuned/summary.json) | `metric=manhattan`, `k=4`, `weights=distance`, `PCA=44` en el mejor fine search del notebook | `0.822` | Mejor version de la etapa tabular clasica. |

### Modelos Colab Ultra y etapa previa al stacking

| Modelo | Artefacto principal | Parametros destacados | Resultado local | Resultado Kaggle conocido | Decision |
| --- | --- | --- | ---: | ---: | --- |
| Random Forest Ultra | [../results/challenge_04_random_forest_colab_ultra_resume/output/challenge_04_random_forest_colab_ultra/summary.json](../results/challenge_04_random_forest_colab_ultra_resume/output/challenge_04_random_forest_colab_ultra/summary.json) | `criterion=gini`, `max_depth=40`, `max_features=0.35`, `min_samples_leaf=1`, `min_samples_split=2`, `n_estimators` final `960` | `0.757` | `0.74722` | Despriorizado. Mejoro, pero no cambia el ranking. |
| SVM Ultra | [../results/challenge_06_svm_colab_ultra_resume/output/challenge_06_svm_colab_ultra/summary.json](../results/challenge_06_svm_colab_ultra_resume/output/challenge_06_svm_colab_ultra/summary.json) | `C=3.0`, `gamma=0.01`, `PCA=128` | `0.841` | `0.84166` | Confirmo que SVM si captura mejor la estructura no lineal. |
| KNN Cleaning Ultra | [../results-pre-stack/knn_cleaning_ultra/output/challenge_08_knn_cleaning_colab_ultra/summary.json](../results-pre-stack/knn_cleaning_ultra/output/challenge_08_knn_cleaning_colab_ultra/summary.json) | `clean=none`, `metric=manhattan`, `k=2`, `weights=distance`, `PCA=64`, `scale=standard` | `0.817`, `OOF=0.8145` | no versionado aqui | Modelo complementario. |
| Signal Features Ultra | [../results-pre-stack/signal_features_ultra/output/challenge_10_signal_features_colab_ultra/summary.json](../results-pre-stack/signal_features_ultra/output/challenge_10_signal_features_colab_ultra/summary.json) | `feature_set=all`, `raw_PCA=32`, `family=svm`, `C=6.0`, `gamma=0.01`, `scale=standard` | `0.9445`, `OOF=0.9403` | confirmado por el usuario como parte del tramo final exitoso | Modelo fuerte del proyecto. |
| Final Stacking | [../results-pre-stack/output/challenge_12_final_stacking_colab/summary.json](../results-pre-stack/output/challenge_12_final_stacking_colab/summary.json) | `weighted_average`, pesos `[0.425, 0.575]`, `threshold=0.49` | `OOF=0.9476` | objetivo alcanzado segun verificacion manual | Solucion final. |

## 3.2 Por que se eligio cada familia de modelos

### Logistic Regression

**Motivacion**

- establecer una referencia lineal
- responder rapido si el problema era separable con una frontera simple

**Resultado**

- el score quedo demasiado bajo
- sirvio mas como prueba negativa que como candidato real

**Decision**

- no invertir mas tiempo en modelos puramente lineales sobre features crudas

### Decision Tree

**Motivacion**

- introducir no linealidad con una familia interpretable y vista en clase

**Resultado**

- mejoro frente a la regresion logistica
- pero siguio muy por debajo de KNN

**Decision**

- mantenerlo solo como benchmark pedagogico

### Random Forest

**Motivacion**

- ensemble robusto sobre datos tabulares
- clasico candidato fuerte cuando hay muchas variables numericas

**Resultado**

- funciono bien como baseline
- no logro competir con KNN/SVM, incluso con busqueda profunda en Colab

**Decision**

- no seguir usando tiempo de computo en esta familia como ruta principal

### Gradient Boosting

**Motivacion**

- explorar boosting como alternativa a Random Forest

**Resultado**

- no encontro una frontera suficientemente fuerte

**Decision**

- descartar como eje central del proyecto

### KNN

**Motivacion**

- la estructura local de la vibracion podia ser mas importante que reglas globales arboladas
- KNN estaba explicitamente alineado con el material del curso

**Region de hiperparametros que se repitio como buena**

- `metric = manhattan`
- `weights = distance`
- `k` bajo
- `PCA` moderado, alrededor de `44-50`

**Interpretacion**

- las observaciones parecen organizarse mejor por proximidad local que por particiones jerarquicas

**Decision**

- KNN paso a ser familia prioritaria desde muy temprano

### SVM con kernel RBF

**Motivacion**

- una vez visto que los arboles no lideraban y que KNN si capturaba senales utiles, tenia sentido probar una familia no lineal basada en margenes

**Resultado**

- `SVM Ultra` alcanzo `validation_accuracy = 0.841`
- publico observado en Kaggle: `0.84166`

**Interpretacion**

- la no linealidad del problema si podia capturarse con una representacion global mas suave que KNN

**Decision**

- promover SVM como segunda gran familia del proyecto

## 3.3 Parametros y decisiones de las lineas finales

### KNN Cleaning Ultra

Artefacto:

- [../colab_knn_cleaning_ultra/Challenge_08_KNN_Cleaning_Colab_Ultra.ipynb](../colab_knn_cleaning_ultra/Challenge_08_KNN_Cleaning_Colab_Ultra.ipynb)

Espacio probado:

- `clean__method`:
  - `none`
  - `lof_0.03`
  - `lof_0.05`
  - `tomek`
  - `enn3`
  - `enn5`
  - `renn3`
  - `renn5`
- `scale__method`:
  - `standard`
  - `robust`
- `pca__n_components`:
  - alrededor de `32-64`
- `model__n_neighbors`:
  - valores bajos
- `model__metric`:
  - `manhattan`
  - `euclidean`

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

**Lectura**

La linea fue valiosa, pero por una razon menos obvia:

- no gano por la limpieza en si
- gano por refinar aun mas la version local y complementaria de KNN

### Signal Features Ultra

Artefacto:

- [../colab_signal_features_ultra/Challenge_10_Signal_Features_Colab_Ultra.ipynb](../colab_signal_features_ultra/Challenge_10_Signal_Features_Colab_Ultra.ipynb)

Features generadas:

- estadisticas temporales basicas:
  - media
  - desviacion
  - RMS
  - energia
  - `peak-to-peak`
  - curtosis
  - asimetria
- estadisticas diferenciales y autocorrelacion simple
- estadisticas por segmentos
- features frecuenciales:
  - modulo de FFT
  - energia por bandas
  - `spectral centroid`
  - frecuencia dominante
  - entropia espectral
- opcion de concatenar `raw PCA`

Mejor configuracion:

```json
{
  "feature__raw_pca": 32,
  "feature__set": "all",
  "model__C": 6.0,
  "model__family": "svm",
  "model__gamma": 0.01,
  "model__metric": null,
  "model__n_neighbors": null,
  "scale__method": "standard"
}
```

**Lectura**

Esta fue la confirmacion experimental de que el mayor salto no estaba en el clasificador aislado, sino en la representacion de la senal.

### SVM Preprocessing Ultra

Artefacto:

- [../colab_svm_preprocessing_ultra/Challenge_09_SVM_Preprocessing_Colab_Ultra.ipynb](../colab_svm_preprocessing_ultra/Challenge_09_SVM_Preprocessing_Colab_Ultra.ipynb)

El `Stage 2` completo dejo esta region fuerte:

| Orden | Cleaning | Scale | PCA | C | gamma | CV mean |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `lof_0.03` | `robust` | `120` | `4.0` | `0.02` | `0.789376` |
| 2 | `lof_0.03` | `power_yeo` | `136` | `3.0` | `0.01` | `0.789125` |
| 3 | `none` | `power_yeo` | `144` | `4.0` | `0.01` | `0.788501` |

El `Stage 3` parcial sugirio otra tendencia:

- `gamma = 0.01` seguia dominando
- `lof_0.03` aparecia mejor que `lof_0.05`
- `PCA = 152` empezo a verse prometedor
- `C = 3.0` y `C = 6.0` aparecian repetidamente arriba

Pero:

- ningun candidato completo llego a `5/5` folds
- por eso esta evidencia fue util como guia, no como criterio final

## 3.4 Por que la solucion final no fue un meta-modelo complejo

En el stacking final se compararon dos familias de combinacion:

1. `weighted_average`
2. `LogisticRegression` como meta-modelo

Top de la busqueda final:

- mejor ponderacion:
  - pesos `[0.425, 0.575]`
  - threshold `0.49`
  - `meta_oof_accuracy = 0.9476`
- mejor `LogisticRegression`:
  - `C = 1.0`
  - threshold `0.65`
  - `meta_oof_accuracy = 0.9473`

### Decision

Se escogio el promedio ponderado porque:

- fue ligeramente mejor
- es mas simple
- es mas facil de explicar
- reduce riesgo de sobreajuste del meta-modelo

## 3.5 Lectura global de resultados

La historia estadistica del proyecto fue esta:

1. Modelos lineales y arboles simples no bastaban.
2. Los ensembles de arboles mejoraron, pero no llegaron a la zona objetivo.
3. KNN y SVM mostraron que la estructura del problema era claramente no lineal.
4. El cambio decisivo fue tratar las `V1 ... V200` como senal y no solo como columnas independientes.
5. Una vez logrado eso, el stacking final solo tuvo que combinar un modelo muy fuerte con otro complementario.
