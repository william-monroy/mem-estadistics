# Reporte Principal Segun Formato

Este documento sigue deliberadamente la estructura de [format.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final/refined-focs/format.md). El objetivo es que pueda usarse como base directa del informe final, con cambios minimos.

## 1. Introduccion

El problema abordado en este proyecto es una tarea de clasificacion binaria orientada a mantenimiento predictivo. A partir de registros de vibracion obtenidos en un banco de pruebas mecanico, se busca determinar si un engranaje solar presenta o no dano superficial. En el dataset provisto, cada observacion contiene 200 variables numericas (`V1 ... V200`) y la respuesta `class`, donde `0` representa un engranaje sano y `1` un engranaje con dano. El detalle del concurso y sus reglas esta en [README.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/README.md).

La motivacion aplicada del problema es clara. En mantenimiento industrial, detectar dano con suficiente anticipacion puede reducir tiempos de parada, evitar fallas de mayor costo y mejorar la planificacion de intervenciones. Sin embargo, el challenge obliga a resolver el problema bajo una metrica concreta: `accuracy` en Kaggle. Ese detalle es importante porque una metrica de competencia no necesariamente coincide con el mejor criterio de decision para un sistema industrial real.

Desde el inicio, el proyecto se planteo bajo dos restricciones metodologicas:

1. usar Python como lenguaje de implementacion;
2. usar modelos vistos en clase o presentes en el material complementario, con libertad para aplicar cualquier tecnica de preprocesamiento siempre que quede bien documentada y justificada.

Esa segunda regla fue decisiva. Las primeras iteraciones confirmaron que el problema no se resolveria solo con "probar mas modelos" sobre la matriz cruda. El avance importante vino cuando se aprovecho la libertad de preprocesamiento para replantear la representacion del problema.

La estrategia global siguio cuatro etapas:

1. auditoria inicial y benchmarking de modelos base;
2. profundizacion en las familias que mostraron mejor señal, principalmente KNN y SVM;
3. cambio de enfoque desde tuning de hiperparametros hacia preprocesamiento y feature engineering;
4. combinacion final de modelos mediante stacking.

El aporte mas fuerte del proyecto no fue un unico hiperparametro particularmente afortunado, sino una secuencia de decisiones defendibles: descartar familias que no justificaban mas presupuesto experimental, explotar la estructura de senal de `V1 ... V200`, y combinar modelos fuertes pero distintos para mejorar generalizacion.

## 2. Metodologia

### 2.1 Preprocesamiento

#### 2.1.1 Auditoria inicial del dataset

Antes de modelar, se verifico la calidad estructural del dataset. El resultado fue:

| Chequeo | Resultado |
| --- | --- |
| `training.csv` | `10000 x 202` |
| `test.csv` | `3000 x 201` |
| balance de clases en train | `{0: 5000, 1: 5000}` |
| valores faltantes en train | `0` |
| valores faltantes en test | `0` |
| duplicados exactos en train | `0` |

Estos chequeos justificaron una primera conclusion metodologica: el principal problema no estaba en limpieza basica del dataset. No hacia falta concentrar esfuerzo en imputacion, tratamiento de clases desbalanceadas o deduplicacion; el reto estaba en representacion, escalado, reduccion de dimensionalidad y seleccion del modelo.

#### 2.1.2 Preprocesamiento base para el benchmarking inicial

La primera etapa uso un pipeline clasico y deliberadamente sobrio:

- `StandardScaler` para homogeneizar escalas;
- `PCA` para reducir dimensionalidad cuando correspondia;
- validacion cruzada para comparar familias de modelos bajo un criterio comun.

La razon de este diseño fue pragmatica. Antes de introducir tecnicas avanzadas, habia que construir una base de comparacion simple, interpretable y reproducible. En particular, `PCA` se justifico por tres motivos:

- las 200 variables podian contener redundancia;
- los modelos sensibles a distancias, como KNN y SVM, suelen beneficiarse de una dimension efectiva menor;
- una reduccion razonable de dimension ayuda a disminuir ruido y costo computacional.

#### 2.1.3 Preprocesamiento orientado a modelos de vecinos

Cuando KNN emergerio como la familia mas prometedora en la etapa inicial, se abrio una linea especifica de limpieza de instancias en [Challenge_08_KNN_Cleaning_Colab_Ultra.ipynb](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_knn_cleaning_ultra/Challenge_08_KNN_Cleaning_Colab_Ultra.ipynb). La hipotesis era simple: si el dataset contenia puntos ruidosos o ambiguos cerca de la frontera de decision, KNN seria particularmente sensible a ellos.

Se evaluaron:

- `Local Outlier Factor (LOF)` por clase;
- `Tomek Links`;
- `Edited Nearest Neighbours (ENN)`;
- `Repeated Edited Nearest Neighbours (RENN)`;
- variantes con `StandardScaler`, `RobustScaler` y `PCA`.

Esta linea fue importante por una razon conceptual: se dejo de tratar el preprocesamiento como una etapa auxiliar y se paso a tratarlo como un grado de libertad central del problema. Aunque el mejor modelo persistido de esa libreta finalmente uso `clean__method = none`, el experimento fue valido y util porque contrasto formalmente una hipotesis metodologica razonable.

#### 2.1.4 Preprocesamiento orientado a SVM

La siguiente linea, implementada en [Challenge_09_SVM_Preprocessing_Colab_Ultra.ipynb](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_svm_preprocessing_ultra/Challenge_09_SVM_Preprocessing_Colab_Ultra.ipynb), partio de otra observacion: SVM ya habia mostrado buen rendimiento con kernel RBF, por lo que el cuello de botella ya no parecia estar solamente en `C` y `gamma`, sino en la transformacion previa del espacio de entrada.

Por eso se probaron:

- `StandardScaler`;
- `RobustScaler`;
- `QuantileTransformer`;
- `PowerTransformer` con Yeo-Johnson;
- una limpieza suave mediante LOF.

La logica fue la siguiente: en un problema no lineal, el clasificador final importa, pero tambien importa fuertemente la geometria del espacio donde se calcula la frontera. La exploracion de esta linea confirmo que habia regiones prometedoras, pero tambien revelo un costo computacional demasiado alto para cerrar exhaustivamente `Stage 3` en Colab gratuito.

#### 2.1.5 Feature engineering de senal

El verdadero punto de quiebre del proyecto fue la libreta [Challenge_10_Signal_Features_Colab_Ultra.ipynb](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_signal_features_ultra/Challenge_10_Signal_Features_Colab_Ultra.ipynb). En esa etapa se replanteo el significado de `V1 ... V200`: en vez de tratarlas como columnas tabulares genericas, se trabajo la hipotesis de que representaban una secuencia ordenada de una señal de vibracion.

Con esa interpretacion se construyeron tres grupos de features:

- estadisticas temporales: media, desviacion, mediana, cuantiles, RMS, energia, `peak-to-peak`, asimetria, curtosis, tasa de cruces por cero;
- estadisticas frecuenciales: magnitud de FFT, energia por bandas, frecuencia dominante, `spectral centroid`, entropia espectral;
- estadisticas por segmentos: medidas temporales calculadas sobre subventanas de la señal.

Ademas, se concateno una pequeña proyeccion `raw PCA` de la matriz original para retener informacion cruda potencialmente complementaria.

Esta fue la decision metodologica mas importante del proyecto. No mejoro el problema por "tunear mas", sino por describir mejor el fenomeno fisico subyacente.

#### 2.1.6 Consideraciones adicionales de preprocesamiento

Tambien se evaluaron otras ideas durante la exploracion:

- `PCA` con `whiten`, que empeoro respecto a PCA convencional;
- `SelectKBest` simple, que no mejoro;
- features provenientes de clustering, que tampoco aportaron;
- una verificacion de posible `train/test shift`, que no mostro una señal util.

La importancia de registrar estos intentos es metodologica: muestra que el proyecto no solo acumulo aciertos, sino que tambien descarto lineas con base en evidencia.

### 2.2 Modelo estadistico

#### 2.2.1 Benchmarking inicial de familias

La primera ronda de comparacion uso modelos clasicos y defendibles dentro del material del curso:

- regresion logistica;
- arbol de decision;
- Random Forest;
- Gradient Boosting;
- K-nearest Neighbors.

Los resultados quedaron consolidados en [model_summary.csv](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/output/model_summary.csv):

| Modelo | Accuracy de validacion |
| --- | ---: |
| K-nearest neighbors | `0.7845` |
| Random Forest | `0.7280` |
| Gradient Boosting | `0.6305` |
| Decision Tree | `0.6110` |
| Logistic Regression | `0.5690` |

Este benchmarking justifico dos decisiones tempranas:

1. abandonar la regresion logistica y los arboles individuales como candidatas serias;
2. concentrar el esfuerzo en KNN y, mas adelante, en SVM.

#### 2.2.2 KNN como primera familia fuerte

KNN fue la primera familia que mostro una señal clara y consistente. La region de mejores soluciones se estabilizo alrededor de:

- `metric = manhattan`;
- `weights = distance`;
- `k` pequeño;
- `PCA` en un rango medio.

El mejor KNN afinado en la fase tabular alcanzo `validation_accuracy = 0.822` segun [challenge_01_knn_tuned/summary.json](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/output/challenge_01_knn_tuned/summary.json). En Kaggle, esa linea llego a `0.83777` de public score. Esto confirmo que la vecindad local estaba capturando estructura real del problema.

#### 2.2.3 Random Forest y por que no fue la linea final

Random Forest se llevo a una version `ultra` en [Challenge_04_RandomForest_Colab_Ultra.ipynb](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_random_forest_ultra/Challenge_04_RandomForest_Colab_Ultra.ipynb) y alcanzo `validation_accuracy = 0.757` con `best_cv_accuracy = 0.739875`, segun [summary.json](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results/challenge_04_random_forest_colab_ultra_resume/output/challenge_04_random_forest_colab_ultra/summary.json).

La familia no fue descartada por ser "incorrecta", sino por relacion costo-beneficio:

- costaba mucho entrenarla;
- no se acercaba a KNN o SVM;
- no ofrecio ninguna evidencia de que una inversion adicional fuerte fuera a cerrar la brecha.

#### 2.2.4 SVM como segunda familia fuerte

SVM con kernel RBF se consolido como una familia competitiva en [summary.json](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results/challenge_06_svm_colab_ultra_resume/output/challenge_06_svm_colab_ultra/summary.json), con:

- `validation_accuracy = 0.841`;
- `model__C = 3.0`;
- `model__gamma = 0.01`;
- `pca__n_components = 128`.

En Kaggle, esa solucion alcanzo `0.84166` de public score. SVM fue importante porque confirmo que el problema exigia una frontera no lineal fuerte, pero tambien porque reforzo la idea de que la representacion de entrada seguia siendo el cuello de botella.

#### 2.2.5 Modelo final

El modelo final no fue un estimador individual, sino un stacking enfocado construido en [Challenge_12_Final_Stacking_Colab.ipynb](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_final_stacking/Challenge_12_Final_Stacking_Colab.ipynb). Los dos modelos base seleccionados fueron:

1. `challenge_08_knn_cleaning_colab_ultra`
2. `challenge_10_signal_features_colab_ultra`

La seleccion de estos dos modelos se justifico por dos razones:

- ambos eran competitivos por si solos;
- sus predicciones no eran redundantes: la tasa de desacuerdo OOF entre ambos fue `0.2086`, segun [pairwise_disagreement.csv](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results-pre-stack/output/challenge_12_final_stacking_colab/checkpoints/pairwise_disagreement.csv).

Se compararon dos meta-estrategias:

- promedio ponderado;
- `LogisticRegression` como meta-modelo.

La mejor combinacion resulto ser:

- `meta_family = weighted_average`
- pesos `[0.425, 0.575]`
- umbral `0.49`
- `meta_oof_accuracy = 0.9476`

segun [summary.json](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results-pre-stack/output/challenge_12_final_stacking_colab/summary.json).

El promedio ponderado se eligio por encima del meta-modelo logit porque:

- fue marginalmente mejor;
- es mas simple de explicar;
- reduce riesgo de sobreajuste adicional;
- conserva interpretabilidad directa sobre el peso de cada modelo base.

## 3. Resultados

### 3.1 Resultados intermedios relevantes

Los puntos de referencia mas importantes del proyecto fueron:

| Etapa | Resultado clave |
| --- | ---: |
| KNN baseline | `0.7845` validacion |
| KNN tuned | `0.8220` validacion |
| Random Forest ultra | `0.7570` validacion |
| SVM ultra | `0.8410` validacion |
| Signal features | `0.9445` validacion / `0.9403` OOF |
| Final stacking | `0.9476` meta OOF |

Este orden deja clara la trayectoria del proyecto:

- primero se mejoro de manera incremental mediante KNN y SVM;
- luego se produjo un salto real cuando se rediseñaron las features;
- finalmente se exprimio una mejora adicional via stacking.

### 3.2 Resultados observados en Kaggle

Los public scores observados durante la experimentacion fueron:

| Submission | Public score |
| --- | ---: |
| `challenge_01_knn_submission.csv` | `0.80611` |
| `challenge_01_knn_tuned_submission.csv` | `0.83777` |
| `svm_colab_submission.csv` | `0.84166` |
| `random_forest_colab_ultra_submission.csv` | `0.74722` |

Posteriormente, el submission final basado en stacking alcanzo el umbral de accuracy buscado por el equipo. Como el valor exacto del leaderboard final no esta almacenado en el repositorio, este campo debe completarse manualmente:

| Medida | Valor |
| --- | --- |
| Public leaderboard final del submission ganador | `[completar con el valor exacto observado en Kaggle]` |
| Private leaderboard final | `[completar al cierre del concurso]` |

### 3.3 Discusion sobre leaderboard publico y privado

El leaderboard publico de Kaggle se calcula sobre un subconjunto del test oculto, mientras que el privado se calcula sobre el resto y define la clasificacion final. Por tanto, optimizar excesivamente contra el publico puede producir sobreajuste competitivo.

La metodologia adoptada intento mitigar ese riesgo:

- se priorizaron `OOF probabilities` sobre decisiones apoyadas solo en un holdout simple;
- el stacking se calibró con evidencia local reproducible, no solo con el leaderboard;
- se prefirio una combinacion simple y robusta de dos modelos base sobre una meta-arquitectura mas compleja.

Si el leaderboard privado final resulta cercano al publico, eso reforzara la conclusion de que la solucion generaliza bien. Si aparece una brecha importante, la explicacion mas probable sera que algunas decisiones de combinacion capturaron parcialmente idiosincrasias del subconjunto publico. En cualquier caso, la ruta seguida es tecnicamente mas defendible que seleccionar submissions exclusivamente por score de Kaggle.

### 3.4 Respuesta explicita: ¿el modelo es exitoso para identificar fallas en el banco de pruebas?

La respuesta es: **si, pero solo dentro del marco exacto de la competencia**.

Es razonable afirmar que el modelo es exitoso porque:

- alcanzo un accuracy alto en Kaggle;
- el mejor modelo individual ya mostraba una separacion muy fuerte (`signal_features`);
- la solucion final combina dos fuentes de evidencia distintas y complementarias.

Sin embargo, hay una salvedad metodologica importante. `Accuracy` no distingue entre el costo industrial de un falso positivo y un falso negativo. En un sistema real de deteccion de fallas, un falso negativo suele ser mas costoso porque implica dejar pasar una condicion dañada como si fuera sana.

Por ello, la respuesta completa es:

- **si**, el modelo es exitoso como solucion del challenge y muestra capacidad real para discriminar entre estados sano/dañado;
- **no necesariamente del todo**, si se quisiera desplegarlo industrialmente sin analisis adicional.

Para un uso real haria falta complementar el estudio con:

- `recall` de la clase dañada;
- tasa de falsos negativos;
- curvas ROC y precision-recall;
- sensibilidad del modelo frente a cambios de condiciones operativas.

## 4. Conclusiones

La conclusion central del proyecto es que la mejora sustantiva no vino de insistir indefinidamente en familias tabulares clasicas, sino de replantear la representacion del problema. El benchmarking inicial fue necesario porque permitio descartar con evidencia a regresion logistica, arboles individuales y boosting clasico como rutas principales. Esa etapa no fue tiempo perdido: definio el espacio de candidatos serios.

KNN y SVM mostraron que la estructura de clases no era lineal y que la geometria local importaba. Sin embargo, ambas familias seguian operando sobre una representacion limitada de la matriz cruda. El verdadero cambio metodologico ocurrio cuando se interpreto que `V1 ... V200` describian una senal de vibracion y se extrajeron features temporales, frecuenciales y por segmentos. Esa reinterpretacion fue la razon principal del salto de desempeño.

La solucion final por stacking deja otra leccion importante. No fue necesario usar un ensamble grande o una capa meta compleja. Basto con combinar de forma disciplinada dos modelos base realmente distintos y calibrar bien pesos y umbral. El resultado fue una solucion mas simple de explicar, mas facil de reproducir y suficientemente fuerte para alcanzar el objetivo competitivo.

Como evaluacion critica del proceso, hay tres puntos a reconocer:

1. al inicio se trato el problema demasiado como tabular generico;
2. algunas busquedas, especialmente en Colab, fueron excesivamente costosas para el valor marginal esperado;
3. el proyecto habria sido mas eficiente si la hipotesis de "senal de vibracion" se hubiera explotado antes.

De cara a trabajo futuro, las mejoras mas razonables serian:

1. complementar la evaluacion con metricas mas alineadas con deteccion de fallas;
2. probar representaciones tiempo-frecuencia aun mas ricas, por ejemplo wavelets, si se pueden justificar y documentar;
3. estudiar estabilidad del modelo bajo ruido o cambios de ventana;
4. cerrar la linea de `SVM preprocessing` solo sobre una region reducida y bien fundada, no sobre una grilla masiva.

## 5. Declaracion de contribucion

La siguiente tabla queda lista para adaptar con nombres reales:

| Integrante | Desarrollo tecnico | Elaboracion del reporte |
| --- | --- | --- |
| `[Integrante 1]` | Auditoria inicial del dataset, benchmarking de modelos base, comparacion temprana entre familias y descarte de lineas menos prometedoras. | Redaccion de introduccion, contexto del problema, definicion del objetivo y primera version de metodologia. |
| `[Integrante 2]` | Ejecucion y seguimiento de notebooks avanzados en Colab, tuning de KNN y SVM, gestion de checkpoints, bundles y validacion de resultados intermedios. | Redaccion de la bitacora metodologica, resultados intermedios y discusion de decisiones experimentales. |
| `[Integrante 3]` | Desarrollo o consolidacion de feature engineering de señal, stacking final, empaquetado de artefactos y seleccion del submission final. | Redaccion de conclusiones, reproducibilidad, riesgos, anexos y consolidacion final del informe. |

Nota: esta distribucion es una plantilla equilibrada. Debe ajustarse para reflejar las contribuciones reales del equipo.

## 6. Referencias

Las referencias completas quedan consolidadas en [07_referencias_apa.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final/refined-focs/refined-docs/07_referencias_apa.md). Las mas importantes para el cuerpo principal del reporte son:

- Breiman (2001) para Random Forest
- Cortes y Vapnik (1995) para SVM
- Cover y Hart (1967) para KNN
- Jolliffe (2002) para PCA
- Breunig et al. (2000) para LOF
- Tomek (1976) y Wilson (1972) para limpieza de instancias
- Wolpert (1992) para stacking
- Randall (2011) para la interpretacion de vibracion y monitoreo de condicion
