# 01. Contexto, Restricciones y Datos

## 1.1 Problema de negocio e ingenieria

El challenge trabaja con un problema clasico de mantenimiento predictivo industrial: detectar fallas en un engranaje solar a partir de senales de vibracion.

Segun el enunciado del proyecto en [../README.md](../README.md):

- los datos provienen de un banco de prueba mecanico con un motor, cajas planetarias, caja paralela y freno magnetico
- las vibraciones se capturan con un sensor de aceleracion
- la tarea es identificar si el engranaje solar presenta dano superficial
- la salida esperada es una clasificacion binaria:
  - `0`: no danado
  - `1`: danado

Esto hace que el problema sea una clasificacion supervisada binaria sobre datos tabulares derivados de senales de vibracion.

## 1.2 Regla academica que condiciono todo el proyecto

La regla operativa del curso, tal como se trabajo durante el proyecto, fue:

- el lenguaje debe ser Python
- deben usarse modelos vistos en clase hasta la fecha de cierre del concurso
- tambien se pueden usar metodos presentes en el material complementario aunque no hayan sido tratados formalmente en clase
- se permite cualquier tecnica de preprocesamiento, incluso si no fue vista en clase, siempre que quede bien documentada y referenciada

Esta regla tuvo dos consecuencias importantes:

1. El espacio de modelos permitidos estuvo acotado por los notebooks de actividades y el material del curso.
2. El mayor margen creativo quedo del lado del `preprocessing` y del `feature engineering`.

## 1.3 Materiales del curso usados como marco metodologico

Los materiales del curso se usaron de dos formas distintas:

### Como base para la estructura de trabajo

Las actividades de referencia marcaron el estilo de desarrollo paso a paso:

- [../reference-activities/Actividad_1_ref.ipynb](../reference-activities/Actividad_1_ref.ipynb)
- [../reference-activities/Activity_2_ref.ipynb](../reference-activities/Activity_2_ref.ipynb)
- [../reference-activities/Actividad_3_ref.ipynb](../reference-activities/Actividad_3_ref.ipynb)
- [../reference-activities/Actividad_4_ref.ipynb](../reference-activities/Actividad_4_ref.ipynb)

De ahi se tomo la idea de mantener siempre un flujo claro:

1. carga de datos
2. auditoria y limpieza
3. preprocesamiento
4. entrenamiento
5. evaluacion
6. interpretacion
7. generacion de submission

### Como base para justificar familias de modelos

Los archivos mas relevantes fueron:

- [../reference-material/Introduction to Classification.md](../reference-material/Introduction%20to%20Classification.md)
- [../reference-material/Classification Trees.md](../reference-material/Classification%20Trees.md)
- [../reference-material/Ensemble Methods.md](../reference-material/Ensemble%20Methods.md)
- [../reference-material/Data preprocessing.md](../reference-material/Data%20preprocessing.md)
- [../reference-material/Principal Component Analysis.md](../reference-material/Principal%20Component%20Analysis.md)
- [../reference-material/Model Evaluation and Inference.md](../reference-material/Model%20Evaluation%20and%20Inference.md)
- [../reference-material/Clustering Methods.md](../reference-material/Clustering%20Methods.md)
- [../reference-material/Additional Topics.md](../reference-material/Additional%20Topics.md)

De este conjunto salieron:

- los modelos de la primera ronda
- la justificacion de PCA y escalado
- la exploracion de ensembles
- la libertad para replantear la representacion del problema via preprocesamiento

## 1.4 Auditoria inicial de datos

La primera auditoria cuantitativa del dataset arrojo:

| Elemento | Resultado |
| --- | --- |
| `training.csv` | `10000 x 202` |
| `test.csv` | `3000 x 201` |
| Numero de features crudas | `200` |
| Variable objetivo | `class` |
| Balance de clases en train | `{0: 5000, 1: 5000}` |
| Missing values en train | `0` |
| Missing values en test | `0` |
| Duplicados exactos en train | `0` |

Interpretacion:

- el dataset esta balanceado, por lo que `accuracy` no esta sesgada por desbalance severo
- no hay una fase de imputacion que domine el problema
- el trabajo real no esta en "arreglar" datos rotos, sino en modelar bien la geometria de las observaciones

## 1.5 Lectura estadistica inicial del problema

Antes de entrenar modelos, habia tres hipotesis razonables:

### Hipotesis A: el problema podia resolverse como tabular clasico

La primera lectura natural era tratar `V1 ... V200` como 200 predictores numericos y comparar familias estandar:

- modelos lineales
- vecinos cercanos
- arboles
- ensembles

Esta hipotesis justifico la ronda inicial de benchmarking.

### Hipotesis B: las features podian conservar orden temporal o estructural

El enunciado habla explicitamente de senales de vibracion. Eso sugeria que `V1 ... V200` podian no ser solo variables tabulares independientes, sino una representacion ordenada de una senal.

Esta intuicion no se exploto al principio, pero luego se convertiria en el cambio mas importante del proyecto.

### Hipotesis C: la metrica oficial obliga a priorizar generalizacion, no solo ajuste local

Kaggle usa `accuracy` sobre un test oculto dividido en leaderboard publico y privado. El riesgo evidente era:

- conseguir un buen score local o publico
- pero fallar al generalizar al conjunto privado

Por eso el proyecto fue migrando desde validaciones simples a:

- `cross-validation`
- `OOF probabilities`
- stacking con artefactos reproducibles

## 1.6 Principios operativos definidos al inicio

Desde el comienzo se adoptaron cuatro principios de trabajo:

1. Cada experimento importante debia dejar un notebook o carpeta trazable.
2. Las figuras debian guardarse fuera del notebook para poder analizarlas despues.
3. Los modelos pesados debian poder reanudarse si se cortaba la sesion.
4. Las decisiones debian estar guiadas por evidencia, no por preferencias de modelo.

Estos principios explican por que el repositorio termino con:

- notebooks base
- notebooks `Ready`
- variantes `Colab Ultra`
- variantes centradas en preprocesamiento
- una ultima fase de `final stacking`
