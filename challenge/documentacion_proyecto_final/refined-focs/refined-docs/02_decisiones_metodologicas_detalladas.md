# Decisiones Metodologicas Detalladas

Este documento no busca repetir resultados, sino explicar el por que de las decisiones. Su valor esta en hacer visible la logica detras del proceso completo: que hipotesis se abrieron, cuales se cerraron y por que el proyecto termino convergiendo hacia `signal feature engineering + stacking`.

## 1. Por que el proyecto no arranco directamente con un pipeline complejo

La primera decision importante fue **no asumir** que el mejor modelo debia ser complejo desde el inicio. Esa decision tuvo tres fundamentos:

- el dataset no presentaba problemas basicos de calidad;
- la variable respuesta estaba balanceada;
- la metrica del concurso era simple y facilmente interpretable.

En este contexto, la forma correcta de arrancar no era construir una solucion sofisticada de una sola vez, sino establecer un benchmark creible. Ese benchmark permitia contestar preguntas basicas:

- ¿el problema parece lineal o no lineal?
- ¿la geometria de distancia parece importante?
- ¿los arboles capturan o no la estructura de separacion?
- ¿vale la pena profundizar en preprocesamiento antes de saber que familias compiten?

La decision fue correcta porque evito sobreingenieria prematura y permitio descartar rapido las familias con peor retorno experimental.

## 2. Por que se hizo una auditoria simple pero explicita de los datos

En muchos proyectos la auditoria inicial se omite del relato porque "no encontro nada". Eso es un error metodologico. Aqui se documento formalmente que:

- no habia missing values;
- no habia duplicados exactos;
- las clases estaban perfectamente balanceadas;
- train y test tenian la estructura esperada.

Esto importaba por dos razones:

1. justificaba no gastar el grueso del esfuerzo en limpieza estructural;
2. permitia defender ante cualquier evaluador que la posterior complejidad del pipeline no era una reaccion a datos rotos, sino a la dificultad real del patron de clasificacion.

## 3. Por que KNN fue el primer modelo realmente serio

KNN no fue elegido por intuicion aislada, sino por evidencia. En el benchmark inicial fue el mejor modelo y, mas importante aun, sus mejores configuraciones se estabilizaron alrededor de un patron interpretable:

- `metric = manhattan`
- `weights = distance`
- `k` pequeño
- `PCA` moderado

Esa regularidad importaba mucho. Un resultado bueno aislado puede ser suerte; una zona de buenos resultados recurrente indica estructura real.

### 3.1 Por que `metric = manhattan` fue una pista importante

Cuando Manhattan supera consistentemente a Euclidean, la lectura razonable es que el problema depende mas de una acumulacion de diferencias parciales que de un gran desplazamiento cuadratico en pocas coordenadas. En datos de vibracion esto tiene sentido: el daño puede alterar la señal en muchos puntos de manera distribuida, en vez de explotar una sola coordenada.

### 3.2 Por que `weights = distance` era coherente con el dominio

Si la frontera de decision es local y sensible a pequeñas variaciones, no tiene sentido que un vecino apenas cercano valga exactamente lo mismo que uno practicamente superpuesto. El peso por distancia hace que la prediccion responda mas a la vecindad fina y menos a la periferia del conjunto local.

### 3.3 Por que KNN no fue el punto final aunque rindiera bien

KNN fue excelente como señal diagnostica del problema, pero tenia dos limites:

- sufre si el espacio de entrada contiene ruido o dimensiones irrelevantes;
- por si solo no aprovecha conocimiento del dominio, solo geometria de los datos.

Eso explica por que fue una gran familia intermedia, pero no la solucion final por si sola.

## 4. Por que los modelos arbolados dejaron de ser prioridad

`Random Forest`, `Decision Tree` y `Gradient Boosting` se probaron porque eran candidatos razonables y estaban bien alineados con el material del curso. No se los descarto arbitrariamente.

La razon del descarte progresivo fue esta:

- `Decision Tree` fue demasiado debil;
- `Gradient Boosting` no se acerco a KNN ni a SVM;
- `Random Forest` mejoro algo, pero a un costo computacional que no justificaba seguir empujando esa linea.

La leccion metodologica aqui es importante: no toda familia popular merece el mismo presupuesto experimental. Una vez que `Random Forest ultra` quedo alrededor de `0.757` en validacion y `0.74722` en Kaggle, seguir invirtiendo dias de computo en ella ya no era una decision racional.

## 5. Por que SVM fue una familia clave aunque no quedara como submission final individual

SVM cumplio dos funciones distintas.

### 5.1 Confirmacion de no linealidad fuerte

Cuando SVM con kernel RBF supera con claridad a los modelos lineales y arbolados, la conclusion es fuerte: la frontera de decision relevante es no lineal, pero no necesariamente del tipo que un ensemble de particiones ortogonales captura bien.

### 5.2 Evidencia de que el problema aun estaba "mal representado"

SVM no solo rindio bien; tambien mostro una region relativamente estable:

- `C` moderado
- `gamma` alrededor de `0.01`
- `PCA` alto

Eso sugeria que el clasificador estaba haciendo buen trabajo, pero sobre una representacion todavia insuficiente. En otras palabras, SVM fue un indicador de que el siguiente salto probablemente no vendria de seguir moviendo `C` y `gamma`, sino de transformar mejor los datos.

## 6. Por que se decidio explotar al maximo la libertad de preprocesamiento

La regla del curso permitia usar cualquier tecnica de preprocesamiento, incluso si no se habia visto formalmente en clase, siempre que se documentara y referenciara bien. Esa libertad cambio el problema de diseño.

En vez de pensar solo "que modelo final puedo usar", la pregunta correcta paso a ser:

> ¿que transformaciones hacen que el patron de daño sea mas separable con modelos estadisticos permitidos?

Ese cambio de pregunta fue posiblemente el mayor avance conceptual del proyecto.

## 7. Por que la limpieza de instancias fue una hipotesis metodologica correcta

La linea `KNN cleaning` surgio porque los metodos de vecinos son vulnerables a:

- ruido local;
- puntos de frontera;
- outliers que deforman la vecindad.

Las tecnicas probadas no fueron caprichosas:

- `LOF` buscaba outliers de densidad local;
- `Tomek Links` buscaba pares ambiguos en la frontera;
- `ENN` y `RENN` intentaban editar el training set para dejar una frontera mas limpia.

Aunque el mejor modelo persistido de esa libreta no termino usando limpieza, la decision de explorarla fue correcta. Sirvio para distinguir intuicion de evidencia y para confirmar que no toda idea razonable produce mejora real en el dataset final.

## 8. Por que la reinterpretacion como señal fue el verdadero pivote

Hasta cierta etapa, el proyecto se comporto como si el dataset fuera un problema tabular cualquiera. Esa era una simplificacion util al principio, pero insuficiente al mediano plazo.

La pregunta que destrabo el proyecto fue:

> ¿y si `V1 ... V200` no son solo 200 features, sino 200 muestras ordenadas de una señal de vibracion?

Esa pregunta cambio todo. Si la interpretacion es correcta, entonces la informacion relevante no esta solo en el valor absoluto de cada columna, sino en:

- energia global;
- forma estadistica;
- distribucion frecuencial;
- cambios locales a lo largo de la ventana.

Eso justifica por que las features temporales, frecuenciales y por segmentos tuvieron sentido fisico y no solo estadistico.

## 9. Por que `signal_features` supero tanto a las lineas tabulares

El salto de `signal_features` no fue un accidente numerico menor. Pasar a `0.9445` de validacion y `0.9403` OOF significo que el pipeline encontro una representacion con mucha mayor separabilidad.

La explicacion razonable es esta:

- las features temporales resumen intensidad, dispersion e impulsividad;
- las features de frecuencia capturan periodicidades y redistribucion de energia;
- las features por segmentos capturan localizacion parcial de eventos;
- el `raw PCA` adicional preserva parte de la informacion cruda que las estadisticas no resumen por completo.

Ese resultado es coherente con el dominio. En vibraciones, el daño superficial no suele expresarse como una sola coordenada anomala, sino como un cambio mas estructural en la forma y el contenido espectral de la señal.

## 10. Por que no se uso un stacking masivo con todos los modelos historicos

Una mala practica comun en Kaggle es meter todos los modelos disponibles al ensamble solo porque "mas modelos" parece mejor. Aqui no se hizo eso.

La decision fue usar solo modelos que cumplieran dos condiciones:

1. ser competitivos por si solos;
2. cometer errores suficientemente distintos.

Por eso el ensamble final uso:

- un modelo dominante: `signal_features`
- un modelo complementario: `knn_cleaning`

No se incluyeron modelos debiles como `Random Forest ultra` porque un meta-modelo tambien puede sobrecargarse con ruido de base learners de baja calidad.

## 11. Por que el promedio ponderado gano al meta-modelo logit

El stacking final comparo un promedio ponderado contra una `LogisticRegression` como meta-modelo. La pregunta correcta aqui no es "cual es mas sofisticado", sino "cual extrae mas valor adicional sin introducir complejidad innecesaria".

El promedio ponderado gano ligeramente con:

- pesos `0.425 / 0.575`
- threshold `0.49`
- `meta_oof_accuracy = 0.9476`

Esto sugiere que la relacion entre ambos modelos base ya era bastante estable y monotona. No hacia falta una capa de aprendizaje adicional muy flexible; bastaba con calibrar bien la mezcla.

## 12. Por que se ajusto el threshold y no se fijo en `0.5`

Muchas soluciones convierten probabilidades a clase con umbral fijo `0.5` sin discutirlo. Aqui no se asumio eso porque:

- la metrica objetivo era `accuracy`;
- el threshold que maximiza accuracy no tiene por que ser `0.5`;
- una leve descalibracion de probabilidades puede volver suboptimo ese valor.

El mejor threshold final fue `0.49`. El cambio parece pequeño, pero el hecho de que aparezca sistematicamente como mejor prueba que dejarlo fijo en `0.5` habria sido una suposicion innecesaria.

## 13. Por que se interrumpio `SVM preprocessing Stage 3`

Abandonar una linea experimental tambien es una decision metodologica. `Stage 3` de `SVM preprocessing` genero:

- `2832` candidatos;
- `5` folds por candidato;
- `14160` fits potenciales.

Con tiempos observados alrededor de `9.24 s` por fit, completar toda la fase en Colab gratuito no era una inversion razonable. Mas aun, la informacion ya disponible decia que:

- `signal_features` estaba muy por delante;
- ya existian dos modelos suficientemente buenos para stacking;
- el valor marginal esperado de cerrar exhaustivamente SVM era bajo frente al costo computacional.

La decision correcta fue detener esa linea y usarla solo como evidencia exploratoria.

## 14. Por que se invirtio tiempo en infraestructura operativa

La aparicion de notebooks `Ultra`, checkpoints, bundles ZIP, orquestacion AWS y hasta una variante con CDK no fue decorativa. Fue una respuesta a una restriccion practica:

- el espacio de busqueda era grande;
- Colab tenia sesiones temporales;
- perder una corrida larga era demasiado costoso.

Documentar esto importa porque muestra madurez de proyecto. No solo se eligieron modelos; tambien se diseñaron mecanismos para que las decisiones pudieran sostenerse operativamente.

## 15. Sintesis de la cadena de decisiones

La logica del proyecto puede resumirse asi:

1. validar que el dataset este sano;
2. benchmarkear modelos base para reducir el espacio de familias;
3. concentrar el trabajo en KNN y SVM porque la evidencia lo justificaba;
4. explotar la libertad de preprocesamiento en vez de seguir empujando tuning ciego;
5. reinterpretar la matriz como señal y construir features acordes;
6. combinar solo modelos realmente competitivos y diferentes;
7. preferir la solucion final mas simple que alcanzara el mejor rendimiento.

Esa secuencia es defendible tanto estadisticamente como desde ingenieria experimental.
