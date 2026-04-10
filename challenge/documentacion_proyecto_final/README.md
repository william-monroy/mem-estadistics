# Documentacion del Proyecto de Clasificacion de Fallas

Esta carpeta resume el proceso completo seguido para resolver el challenge de clasificacion de fallas en engranajes a partir de senales de vibracion.

El objetivo de esta documentacion es convertir el trabajo tecnico realizado en una base util para:

- redactar el informe final
- preparar la presentacion del proyecto
- justificar tecnicamente cada decision importante
- explicar por que ciertas lineas se abandonaron y otras se promovieron

## Navegacion

1. [01_contexto_restricciones_y_datos.md](./01_contexto_restricciones_y_datos.md)
   Contexto del problema, reglas del curso, auditoria de datos e hipotesis iniciales.

2. [02_bitacora_experimental.md](./02_bitacora_experimental.md)
   Relato cronologico de la estrategia seguida desde los primeros modelos hasta el stacking final.

3. [03_modelos_parametros_y_resultados.md](./03_modelos_parametros_y_resultados.md)
   Catalogo detallado de modelos, hiperparametros, desempeno y decisiones.

4. [04_desafios_operativos_y_lecciones.md](./04_desafios_operativos_y_lecciones.md)
   Problemas encontrados durante el proyecto, cuellos de botella y soluciones de ingenieria.

5. [05_solucion_final_y_artifacts.md](./05_solucion_final_y_artifacts.md)
   Descripcion de la solucion final, artefactos relevantes y guia para reutilizar el pipeline.

## Resumen ejecutivo

- El problema consiste en clasificar una condicion de dano en un engranaje solar a partir de senales de vibracion.
- La metrica oficial de Kaggle es `accuracy`.
- El conjunto de entrenamiento esta balanceado y no presenta valores faltantes ni duplicados exactos.
- La primera ronda de modelos mostro muy rapido que las familias lineales y los arboles individuales no eran suficientes.
- `KNN` y despues `SVM` fueron las primeras familias realmente competitivas.
- El cambio de mayor impacto no vino de seguir afinando hiperparametros sobre features crudas, sino de cambiar la representacion del problema:
  - primero con preprocesamiento orientado a vecindad
  - despues con `signal feature engineering`
- La mejor solucion local antes del submission final fue un `stacking` enfocado que combino:
  - `challenge_08_knn_cleaning_colab_ultra`
  - `challenge_10_signal_features_colab_ultra`
- El blend final ganador fue un promedio ponderado con pesos `0.425 / 0.575` y umbral `0.49`, con `meta_oof_accuracy = 0.9476`.
- Segun la verificacion manual realizada en Kaggle al final del proceso, esta solucion si alcanzo el objetivo de accuracy esperado.

## Archivos base del proyecto

- Enunciado del challenge: [../README.md](../README.md)
- Actividades de referencia:
  - [../reference-activities/Actividad_1_ref.ipynb](../reference-activities/Actividad_1_ref.ipynb)
  - [../reference-activities/Activity_2_ref.ipynb](../reference-activities/Activity_2_ref.ipynb)
  - [../reference-activities/Actividad_3_ref.ipynb](../reference-activities/Actividad_3_ref.ipynb)
  - [../reference-activities/Actividad_4_ref.ipynb](../reference-activities/Actividad_4_ref.ipynb)
- Material de referencia principal:
  - [../reference-material/Introduction to Classification.md](../reference-material/Introduction%20to%20Classification.md)
  - [../reference-material/Classification Trees.md](../reference-material/Classification%20Trees.md)
  - [../reference-material/Ensemble Methods.md](../reference-material/Ensemble%20Methods.md)
  - [../reference-material/Data preprocessing.md](../reference-material/Data%20preprocessing.md)
  - [../reference-material/Principal Component Analysis.md](../reference-material/Principal%20Component%20Analysis.md)
  - [../reference-material/Model Evaluation and Inference.md](../reference-material/Model%20Evaluation%20and%20Inference.md)

## Nota importante sobre el score final en Kaggle

El repositorio si conserva los artefactos locales finales del stacking, pero no versiona automaticamente el valor exacto del `public leaderboard`.

Por tanto, en el informe final conviene dejar una linea para completar manualmente:

`Public leaderboard final del stacking: [completar con el valor exacto observado en Kaggle]`
