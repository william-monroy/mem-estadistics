# Metodologia: Modelo Estadistico

## Parte 1. Contenido para presentacion

### Slide 1. Seleccion de modelos

- Benchmark inicial de 5 familias:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - KNN
- Mejor baseline: `KNN = 0.7845`
- Luego se profundizo en las lineas mas prometedoras:
  - `SVM`
  - `KNN`
  - `signal_features + SVM`

### Slide 2. Modelo final elegido

- Mejor modelo individual:
  - `signal_features + SVM`
  - `validation = 0.9445`
  - `OOF = 0.9403`
- Modelo final de competencia:
  - stacking enfocado entre `signal_features` y `knn_cleaning`
  - mejor combinacion:
    - pesos `[0.425, 0.575]`
    - threshold `0.49`
    - `meta OOF = 0.9476`

## Parte 2. Dialogo de explicacion

### Slide 1. Dialogo

El benchmark inicial fue necesario para tomar una decision disciplinada sobre en que familias valia la pena invertir tiempo. Logistic Regression y Decision Tree quedaron rapidamente descartados. Gradient Boosting tampoco mostró una señal competitiva. Random Forest si funcionó, pero no al nivel suficiente como para justificar todo el presupuesto experimental que requería. KNN fue el primer modelo que realmente se separó del resto, y eso nos dijo que la geometria local del problema era importante.

Despues incorporamos SVM porque queriamos confirmar si una frontera no lineal mas fuerte podia superar a KNN. Efectivamente SVM funcionó bien, pero el mayor aprendizaje no fue que "SVM era mejor", sino que todavia estabamos limitados por la representacion de entrada. Eso fue lo que abrió el paso a `signal_features`.

### Slide 2. Dialogo

El mejor modelo individual del proyecto fue un SVM entrenado sobre features de señal. Esa fue la primera vez que pasamos claramente la barrera de 0.94 en validacion. Pero aun con ese resultado, seguimos encontrando valor en una segunda fuente de error distinta: `knn_cleaning`. Aunque ese modelo era mucho mas debil por si solo, sus errores no coincidían completamente con los del modelo principal.

Por eso el modelo final no fue el mejor individual, sino un stacking muy enfocado entre ambos. Comparamos promedio ponderado contra una regresion logistica como meta-modelo, y el promedio ponderado ganó por poco. Lo elegimos porque era ligeramente mejor, mas simple y mas facil de defender en una presentacion academica.

## Parte 3. Respaldo tecnico y material de apoyo

### Resultados usados para justificar la seleccion

Benchmark inicial:

| Modelo | Accuracy de validacion |
| --- | ---: |
| KNN | `0.7845` |
| Random Forest | `0.7280` |
| Gradient Boosting | `0.6305` |
| Decision Tree | `0.6110` |
| Logistic Regression | `0.5690` |

Modelo individual fuerte:

| Modelo | Accuracy |
| --- | ---: |
| `signal_features` validation | `0.9445` |
| `signal_features` OOF | `0.9403` |

Modelo final:

| Metrica | Valor |
| --- | ---: |
| `best_meta_oof_accuracy` | `0.9476` |
| threshold | `0.49` |
| pesos | `[0.425, 0.575]` |

### Parametros finales importantes

`signal_features`:

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

`final_stacking`:

```json
{
  "meta_family": "weighted_average",
  "weights": [0.425, 0.575],
  "best_threshold": 0.49
}
```

### Archivos de respaldo

- Benchmark inicial:
  - [model_summary.csv](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/output/model_summary.csv)
- Signal features:
  - [summary.json](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results-pre-stack/signal_features_ultra/output/challenge_10_signal_features_colab_ultra/summary.json)
- Final stacking:
  - [summary.json](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results-pre-stack/output/challenge_12_final_stacking_colab/summary.json)
  - [final_stacking_search_results.csv](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results-pre-stack/output/challenge_12_final_stacking_colab/checkpoints/final_stacking_search_results.csv)

### Decisiones que conviene justificar si preguntan

- por que no se siguio empujando `Random Forest`
- por que `KNN` siguio siendo util aunque no ganara por si solo
- por que el stacking uso solo dos modelos base y no todos los historicos
- por que el promedio ponderado gano al meta-modelo logit

### Sugerencia visual

- Slide 1:
  - grafico de barras con los 5 modelos base
- Slide 2:
  - esquema simple:
    - `signal_features`
    - `knn_cleaning`
    - `weighted average + threshold`
