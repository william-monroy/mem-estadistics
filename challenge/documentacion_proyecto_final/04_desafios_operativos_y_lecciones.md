# 04. Desafios Operativos, Problemas y Lecciones

## 4.1 El principal problema no fue estadistico, fue operacional

La dificultad del proyecto no estuvo solo en elegir modelos. Tambien estuvo en poder ejecutar los experimentos dentro de restricciones reales:

- notebook local
- Google Colab free
- sesiones temporales
- CPU limitada
- riesgo de perder trabajo si no se guardaban resultados parciales

Por eso el proyecto termino requiriendo ingenieria de ejecucion, no solo modelado.

## 4.2 Primer gran cuello de botella: busquedas exhaustivas demasiado caras

Las primeras busquedas profundas mostraron una tension clara:

- queriamos explorar mas parametros
- pero cada ampliacion del espacio disparaba el tiempo de entrenamiento

Esto obligo a pasar por tres capas:

1. notebooks base con busqueda moderada
2. notebooks `Ready` con parametros ya fijados
3. notebooks `Colab Ultra` por etapas y con checkpoint

## 4.3 Por que hubo que construir notebooks resumibles

El trabajo en Colab free revelo muy pronto dos problemas:

1. las sesiones se reinician
2. el almacenamiento local del runtime es temporal

Si una busqueda larga no guardaba resultados intermedios, se perdia por completo.

De ahi surgieron decisiones de ingenieria que hoy son parte central del repositorio:

- guardar resultados por etapa
- guardar resultados por fold
- exportar bundles ZIP de reanudacion
- aislar `output/`, `submissions/` y `checkpoints/`

Estas decisiones quedaron reflejadas, por ejemplo, en:

- [../colab_random_forest_ultra](../colab_random_forest_ultra)
- [../colab_knn_ultra](../colab_knn_ultra)
- [../colab_svm_ultra](../colab_svm_ultra)
- [../colab_final_stacking](../colab_final_stacking)

## 4.4 Random Forest fue una leccion de techo practico

`Random Forest` nunca estuvo "mal". De hecho fue un baseline razonable.

El problema fue otro:

- incluso al volverlo mas sofisticado
- incluso al hacerlo resumible
- incluso al dejarlo correr por mucho tiempo

no produjo un salto proporcional al costo de computo.

Metricas relevantes:

- baseline local:
  - `validation_accuracy = 0.728`
- ultra local:
  - `validation_accuracy = 0.757`
- public Kaggle observado:
  - `0.74722`

### Leccion

No toda mejora local justifica el costo operacional. Cuando una familia muestra techo claro, insistir en ella ya no es investigacion productiva.

## 4.5 SVM preprocessing: ejemplo claro de explosion combinatoria

La linea `SVM preprocessing` fue metodologicamente correcta, pero operacionalmente demasiado cara en su `Stage 3`.

Estado documentado en:

- [../results-pre-stack/svm_preprocessing/output/challenge_09_svm_preprocessing_colab_ultra/checkpoints/stage2_cv_summary.csv](../results-pre-stack/svm_preprocessing/output/challenge_09_svm_preprocessing_colab_ultra/checkpoints/stage2_cv_summary.csv)
- [../results-pre-stack/svm_preprocessing/output/challenge_09_svm_preprocessing_colab_ultra/checkpoints/stage3_local_candidates.json](../results-pre-stack/svm_preprocessing/output/challenge_09_svm_preprocessing_colab_ultra/checkpoints/stage3_local_candidates.json)
- [../results-pre-stack/svm_preprocessing/output/challenge_09_svm_preprocessing_colab_ultra/checkpoints/stage3_local_cv_fold_results(1).csv](../results-pre-stack/svm_preprocessing/output/challenge_09_svm_preprocessing_colab_ultra/checkpoints/stage3_local_cv_fold_results(1).csv)

Magnitud del problema:

- candidatos de `Stage 3`: `2832`
- folds por candidato: `5`
- fits totales requeridos: `14160`

Progreso real cuando se detuvo:

- archivo parcial mas avanzado:
  - `1235` evaluaciones
- pero todas sobre:
  - `fold_idx = 0`
- candidatos completos:
  - `0`

Estimacion de costo con los tiempos observados:

- tiempo medio por fit: ~`9.24 s`
- tiempo serial restante: ~`36.2 h`
- incluso repartido en `3` cuentas de Colab free: ~`12.1 h`

### Leccion

Cuando el costo de terminar una busqueda ya supera el valor esperado de la informacion que podria aportar, hay que cambiar de estrategia.

Eso fue exactamente lo que se hizo:

- se dejo de esperar a `SVM preprocessing`
- se paso directamente al stacking con los modelos ya completos

## 4.6 KNN cleaning dejo una leccion metodologica importante

La hipotesis inicial era que limpiar outliers o puntos ruidosos mejoraria claramente a KNN.

Sin embargo, el mejor candidato final de la notebook resulto ser:

- `clean__method = none`

Esto no invalida la linea. Al contrario, muestra una buena practica de proyecto:

- una hipotesis razonable debe ponerse a prueba
- si el mejor resultado final contradice la intuicion inicial, se documenta tal cual

### Leccion

La documentacion debe registrar tanto aciertos como refutaciones parciales.

## 4.7 El gran giro del proyecto fue conceptual: tratar la base como senal

Durante varias fases se trato el problema como tabular clasico. Eso era razonable al comienzo.

Pero la ganancia mas fuerte aparecio cuando se acepto una idea mas fiel al dominio:

- `V1 ... V200` probablemente retienen estructura de senal

La notebook de `signal feature engineering` funciono porque:

- introdujo una representacion mas alineada con el fenomeno fisico
- no se limito a cambiar el clasificador
- cambio el espacio donde el clasificador opera

### Leccion

Cuando el origen de los datos tiene estructura fisica clara, ignorarla suele ser mas costoso que cualquier error de tuning fino.

## 4.8 Por que se construyo infraestructura para AWS y CDK

El proyecto tambien exploro una ruta de ejecucion en AWS:

- [../aws_ultra_suite](../aws_ultra_suite)
- [../aws_cdk_runner](../aws_cdk_runner)

Esa ruta surgio por una necesidad real:

- algunas busquedas eran demasiado largas para Colab free
- habia riesgo de perder sesiones
- se necesitaba una alternativa automatizable para corridas largas

Aunque la solucion final no dependio de AWS para alcanzar el resultado, esa exploracion dejo dos aprendizajes:

1. separar la logica de entrenamiento de la infraestructura vale la pena
2. automatizar `deploy -> run -> collect -> destroy` reduce errores humanos

## 4.9 Importancia de `OOF probabilities`

Una leccion central del tramo final fue que las predicciones de clase no bastan para combinar modelos bien.

Para hacer stacking serio se necesitaba:

- `oof_probabilities.csv`
- `test_probabilities.csv`

Eso obligo a modificar el diseno de notebooks posteriores para que exportaran esos artefactos de forma sistematica.

Sin esa disciplina:

- el stacking final no habria sido posible
- o habria sido mucho menos confiable

## 4.10 Lecciones generales del proyecto

1. Empezar con benchmarking amplio fue correcto.
2. Las familias de arboles fueron utiles para aprender, pero no para cerrar el challenge.
3. Los mayores saltos no vinieron de busquedas mas grandes, sino de cambiar la representacion.
4. El costo computacional condiciona la metodologia.
5. Checkpoints y reanudacion no son extras; son parte del experimento.
6. Un stacking simple y bien justificado puede ser mejor que un meta-modelo mas complejo.
