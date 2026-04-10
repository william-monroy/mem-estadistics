# Colab Leaderboard Push Ultra

Esta carpeta contiene la ultima libreta orientada especificamente a subir unas decimas adicionales en el `public leaderboard`.

## Hipotesis de trabajo

El submission final actual ya hizo dos cosas bien:

- uso el modelo base mas fuerte disponible: `signal_features`;
- capturo diversidad real con `knn_cleaning`.

Para intentar cerrar la brecha restante con el top 1 y top 2, esta variante ya no hace una busqueda amplia. Hace algo mas enfocado:

1. reentrena solo candidatos `elite` muy cercanos al mejor `signal_features`;
2. añade una cuota pequeña de diversidad via `raw SVM` y `KNN refresh`;
3. hace un ensamble mas fino que el del `challenge_12`;
4. opcionalmente genera submissions agresivos con pseudo-labeling de alta confianza.

## Archivos principales

- `Challenge_13_Leaderboard_Push_Colab.ipynb`
- `requirements.txt`
- `package_current_models.py`
- `upload_bundles/current_artifacts_for_lb_push.zip`

## Que conviene subir a Colab

Minimo:

- `Challenge_13_Leaderboard_Push_Colab.ipynb`
- `training.csv`
- `test.csv`
- `sample.csv`

Muy recomendado:

- `upload_bundles/current_artifacts_for_lb_push.zip`

Ese ZIP ya contiene los artefactos actuales mas fuertes del proyecto:

- `challenge_10_signal_features_colab_ultra`
- `challenge_08_knn_cleaning_colab_ultra`
- `challenge_12_final_stacking_colab`

## Flujo recomendado

1. abrir una sesion CPU en Google Colab
2. subir la notebook
3. en la celda de configuracion poner:

```python
UPLOAD_DATA_FILES = True
UPLOAD_RESULT_BUNDLES = True
SEARCH_PROFILE = "aggressive"
RUN_STAGE_4_AGGRESSIVE_PSEUDOLABEL = True
```

4. subir los CSV de datos
5. subir `current_artifacts_for_lb_push.zip`
6. ejecutar toda la notebook

## Que submissions genera

La notebook intenta dejar hasta cuatro archivos:

- `challenge_13_public_lb_push_safe_submission.csv`
- `challenge_13_public_lb_push_aggressive_alpha15.csv`
- `challenge_13_public_lb_push_aggressive_alpha25.csv`
- `challenge_13_public_lb_push_pseudo_only.csv`

## Orden sugerido para Kaggle

1. `safe_submission`
2. `aggressive_alpha15`
3. `aggressive_alpha25`
4. `pseudo_only` solo si aun quieres probar una variante mas arriesgada

La idea es empezar por la opcion mejor validada y luego usar las variantes agresivas solo como intento final de leaderboard.

## Nota metodologica

El modo `safe` esta apoyado por OOF.  
El modo `aggressive` usa pseudo-labeling y por tanto es mas especulativo. Eso puede ayudar al `public leaderboard`, pero no necesariamente al `private`.
