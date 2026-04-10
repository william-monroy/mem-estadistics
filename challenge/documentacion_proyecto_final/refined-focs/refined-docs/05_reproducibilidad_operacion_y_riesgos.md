# Reproducibilidad, Operacion y Riesgos

Este documento resume la parte menos visible pero mas importante del proyecto: como se hizo posible experimentar de manera robusta pese a limites de tiempo, sesiones temporales y ejecuciones largas.

## 1. Por que la operacion fue parte del metodo

En un proyecto pequeno, la historia podria contarse como:

1. cargar datos
2. entrenar modelos
3. elegir el mejor
4. enviar submission

Aqui eso no era realista. Hubo restricciones concretas:

- ejecucion local limitada para busquedas largas;
- sesiones efimeras de Google Colab;
- notebooks con stages que podian durar horas;
- necesidad de no perder progreso parcial;
- presion por comparar muchas lineas sin rehacer todo desde cero.

Por eso la reproducibilidad no fue un tema secundario; fue una condicion para poder llegar al submission final.

## 2. Evolucion de la infraestructura experimental

### 2.1 Notebooks base

Primera etapa:

- notebooks simples;
- foco en benchmarking, claridad y comparabilidad.

Ventaja:

- lectura directa;
- facil debug;
- baja complejidad conceptual.

Limite:

- no soportaban bien busquedas largas ni reanudacion.

### 2.2 Notebooks `Ready`

Segunda etapa:

- congelar los mejores hiperparametros hallados;
- dejar libretas listas para ejecutar sin `GridSearchCV` pesado.

Ventaja:

- ahorraban tiempo cuando ya existia una region buena.

Limite:

- seguian siendo insuficientes para lineas donde el espacio de busqueda aun era grande.

### 2.3 Notebooks `Colab Ultra`

Tercera etapa:

- ejecucion por stages;
- checkpoints por etapa o por fold;
- exportacion de bundles ZIP;
- capacidad de reanudar trabajo en nuevas sesiones.

Ventaja:

- hizo viable experimentar bajo limites de Colab free.

Limite:

- algunos stages, como `SVM preprocessing Stage 3`, siguieron siendo demasiado costosos.

## 3. Problemas operativos reales que hubo que resolver

### 3.1 Reinicios de Google Colab

Colab free no garantiza sesiones largas ni persistencia de disco local. Eso obligo a adoptar una politica sistematica:

- exportar bundles al terminar cada etapa importante;
- descargar manualmente los archivos criticos cuando la sesion estaba en riesgo;
- reabrir sesiones nuevas reusando checkpoints.

### 3.2 Corridas demasiado largas

El mejor ejemplo fue `SVM preprocessing Stage 3`:

- `2832` candidatos;
- `5` folds por candidato;
- `14160` fits potenciales;
- alrededor de `9.24 s` por fit observados.

Eso equivale a una carga impractica para Colab gratuito. La leccion fue clara:

- una exploracion puede ser metodologicamente correcta;
- pero si el costo computacional es desproporcionado frente al valor esperado, debe detenerse.

### 3.3 Progreso parcial no resumible por defecto

Herramientas como `GridSearchCV` no siempre facilitan rescatar progreso parcial si el proceso se interrumpe. Por eso las variantes `Ultra` se diseñaron para escribir resultados parciales a disco en archivos CSV y JSON, en vez de depender solo de objetos en memoria.

## 4. Artefactos que hicieron viable la reanudacion

El flujo final se apoyo en varios tipos de artefactos:

- `summary.json`
- `oof_probabilities.csv`
- `test_probabilities.csv`
- `stage*_cv_summary.csv`
- `stage*_cv_fold_results.csv`
- bundles ZIP de reanudacion

Su funcion fue distinta:

- `summary.json` sintetiza el resultado final de una corrida;
- `oof_probabilities.csv` y `test_probabilities.csv` desacoplan entrenamiento base y stacking;
- los CSV de folds permiten inspeccionar progreso parcial;
- los ZIP permiten transportar estados entre sesiones de Colab.

## 5. Flujo reproducible recomendado para los modelos base

### 5.1 KNN cleaning

Ruta:

- [colab_knn_cleaning_ultra](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_knn_cleaning_ultra)

Flujo:

1. subir la notebook a Colab;
2. subir `training.csv`, `test.csv`, `sample.csv`;
3. ejecutar los stages definidos;
4. exportar el ZIP de resultados;
5. descargar el ZIP antes de cerrar sesion.

### 5.2 Signal features

Ruta:

- [colab_signal_features_ultra](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_signal_features_ultra)

Flujo:

1. subir notebook;
2. cargar datos;
3. correr screening y CV;
4. exportar artefactos finales;
5. conservar `summary.json`, `oof_probabilities.csv`, `test_probabilities.csv`.

### 5.3 SVM preprocessing

Ruta:

- [colab_svm_preprocessing_ultra](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_svm_preprocessing_ultra)

Estado metodologico:

- util como exploracion;
- no necesario para reproducir el submission ganador;
- si se reabre, debe hacerse sobre una region reducida, no sobre el `Stage 3` completo.

## 6. Flujo reproducible del stacking final

Ruta:

- [colab_final_stacking](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_final_stacking)

Artefactos clave ya preparados:

- [knn_cleaning_ultra_for_final_stacking.zip](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_final_stacking/upload_bundles/knn_cleaning_ultra_for_final_stacking.zip)
- [signal_features_ultra_for_final_stacking.zip](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_final_stacking/upload_bundles/signal_features_ultra_for_final_stacking.zip)

Flujo:

1. empaquetar resultados base si fuera necesario con [package_results_pre_stack.py](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_final_stacking/package_results_pre_stack.py)
2. subir [Challenge_12_Final_Stacking_Colab.ipynb](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_final_stacking/Challenge_12_Final_Stacking_Colab.ipynb) a Colab
3. subir los ZIPs de modelos base
4. ejecutar la busqueda final de combinaciones
5. descargar el submission ganador

## 7. AWS y CDK: por que se exploraron aunque no fueran imprescindibles al final

Durante el proyecto se prepararon dos rutas adicionales:

- [aws_ultra_suite](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_ultra_suite)
- [aws_cdk_runner](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_cdk_runner)

Estas carpetas respondian a una necesidad real:

- algunas corridas eran demasiado largas para Colab;
- se necesitaba evaluar opciones mas estables;
- era razonable separar logica de modelado e infraestructura.

Aunque la solucion final no dependio de AWS, la exploracion deja dos beneficios:

1. demuestra que el proyecto penso la escalabilidad de la experimentacion;
2. deja una ruta util si hubiera que repetir corridas largas o rehacer busquedas en otra etapa.

## 8. Riesgos metodologicos y como se mitigaron

### Riesgo 1: sobreajuste al leaderboard publico

Mitigacion:

- no elegir submissions solo por score de Kaggle;
- priorizar OOF y artefactos persistidos;
- usar el leaderboard publico como validacion complementaria, no como unica verdad.

### Riesgo 2: sobreajuste del stacking

Mitigacion:

- usar solo dos modelos base realmente competitivos;
- comparar promedio ponderado con meta-modelo logit;
- preferir la solucion mas simple que alcanzaba el mejor resultado.

### Riesgo 3: interpretar demasiado fuerte un SVM preprocessing incompleto

Mitigacion:

- documentar que `Stage 3` quedo parcial;
- usar esa evidencia solo como indicio exploratorio;
- no basar el submission ganador en ese stage.

### Riesgo 4: depender demasiado de accuracy

Mitigacion:

- explicitar en el informe que `accuracy` no agota el problema industrial;
- recomendar como trabajo futuro el analisis de falsos negativos y recall de la clase dañada.

## 9. Lo que esta capa operativa agrega a la defensa del proyecto

Desde fuera, podria parecer que la infraestructura extra fue accesorio. No lo fue. Aporta tres argumentos fuertes para la defensa del trabajo:

- hubo criterio estadistico para elegir modelos y criterios de evaluacion;
- hubo criterio de ingenieria para no perder trabajo y sostener corridas largas;
- hubo criterio de priorizacion para detener lineas costosas cuando la evidencia ya no justificaba seguir.

Esa combinacion fortalece el proyecto porque muestra que no solo se encontro un buen modelo; tambien se construyo un proceso robusto para llegar a el.
