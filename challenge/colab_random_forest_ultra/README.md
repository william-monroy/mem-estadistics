# Random Forest Ultra for Colab

Esta carpeta ya no asume que tu proyecto completo existe previamente en Google Drive.

El flujo correcto ahora es este:

1. Subes solamente [Challenge_04_RandomForest_Colab_Ultra.ipynb](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_random_forest_ultra/Challenge_04_RandomForest_Colab_Ultra.ipynb) a Google Colab.
2. Inicias la sesión de Colab.
3. El notebook crea un workspace vacío dentro de `/content/challenge_random_forest_ultra_workspace`.
4. Desde la propia sesión de Colab subes manualmente:
   - `training.csv`
   - `test.csv`
   - `sample.csv`
   - o un ZIP de reanudación exportado previamente

## Contenido de esta carpeta

- [Challenge_04_RandomForest_Colab_Ultra.ipynb](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_random_forest_ultra/Challenge_04_RandomForest_Colab_Ultra.ipynb): notebook principal.
- [README.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_random_forest_ultra/README.md): esta guía.
- [requirements.txt](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_random_forest_ultra/requirements.txt): referencia de dependencias.
- [workspace_template](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_random_forest_ultra/workspace_template): estructura de carpetas esperada dentro de Colab.

## Qué hace el notebook

- crea automáticamente un workspace en `/content/challenge_random_forest_ultra_workspace`
- permite subir los CSV manualmente desde el navegador usando `files.upload()`
- permite restaurar un ZIP de reanudación
- deja Google Drive como persistencia opcional, no obligatoria
- guarda checkpoints por etapas
- puede reanudar la búsqueda si restauras los checkpoints

## Estructura del workspace dentro de Colab

El notebook trabaja con esta estructura:

```text
/content/challenge_random_forest_ultra_workspace/
├── data/
│   ├── training.csv
│   ├── test.csv
│   └── sample.csv
├── output/
│   └── challenge_04_random_forest_colab_ultra/
│       └── checkpoints/
├── submissions/
└── exports/
```

## Cómo usarlo paso a paso en Colab

1. Abre el notebook en Google Colab.
2. Usa runtime de CPU.
3. Si tienes `High-RAM`, mejor.
4. Ejecuta las celdas `0`, `1` y `2`.
5. En la celda `Optional: upload the three CSV files manually from the browser` cambia:
   - `UPLOAD_DATA_FILES = True`
6. Ejecuta esa celda y selecciona:
   - `training.csv`
   - `test.csv`
   - `sample.csv`
7. Si en vez de eso vas a continuar una corrida anterior:
   - deja `UPLOAD_DATA_FILES = False`
   - en la celda `Optional: restore a previous resume ZIP` cambia `RESTORE_RESUME_BUNDLE = True`
   - sube el ZIP exportado previamente
8. Ejecuta la celda de validación de archivos.
9. Revisa el perfil de búsqueda:
   - `balanced` para una sesión más conservadora
   - `aggressive` para exprimir más Colab
10. Ejecuta el resto del notebook.

## Persistencia y reanudación

Hay dos modos.

### Modo base: manual

No usa Google Drive.

- los checkpoints se van guardando dentro del workspace temporal de Colab
- al final de cada etapa se genera un bundle ZIP de reanudación en `exports/`
- si quieres poder continuar luego, descarga ese ZIP antes de cerrar o perder la sesión

### Modo opcional: Google Drive

Si quieres persistencia más segura:

1. En la celda de configuración cambia `ENABLE_GOOGLE_DRIVE_PERSISTENCE = True`
2. Ejecuta luego la celda `Optional: enable Google Drive persistence`

Con eso el notebook copiará checkpoints y artefactos importantes a Drive. Ya no depende de que descargues manualmente el ZIP después de cada etapa.

## Qué cambia respecto a la versión anterior

- ya no asume que `challenge/data/` existe en Drive antes de empezar
- ya no parte de una carpeta prearmada en Colab
- el usuario controla la subida manual de archivos dentro de la sesión
- el workspace se crea solo
- la reanudación ahora funciona tanto por ZIP manual como por Drive opcional

## Límite importante

Este notebook está mucho mejor adaptado al flujo real de Colab, pero eso no garantiza que Random Forest vaya a superar 0.90 en este dataset. La mejora principal aquí es operativa:

- búsqueda más inteligente
- menos desperdicio de tiempo
- checkpoints útiles
- reanudación realista
