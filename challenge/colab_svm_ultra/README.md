# SVM Ultra for Colab

Esta carpeta está pensada para ejecutar una búsqueda profunda de `SVC` con kernel `RBF` en Google Colab usando el mismo enfoque operativo que ya venimos usando:

- ejecución por etapas
- checkpoints
- reanudación con ZIP
- Google Drive opcional

## Contenido

- [Challenge_06_SVM_Colab_Ultra.ipynb](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_svm_ultra/Challenge_06_SVM_Colab_Ultra.ipynb): notebook principal para Colab.
- [README.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_svm_ultra/README.md): esta guía.
- [requirements.txt](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_svm_ultra/requirements.txt): referencia de dependencias.
- [workspace_template](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_svm_ultra/workspace_template): estructura esperada del workspace.

## Flujo esperado en Colab

Esta variante no asume que tu proyecto ya existe en Google Drive.

El flujo correcto es:

1. Subes solo el notebook a Google Colab.
2. Inicias una sesión nueva.
3. El notebook crea un workspace vacío en `/content/challenge_svm_ultra_workspace`.
4. Desde Colab subes manualmente:
   - `training.csv`
   - `test.csv`
   - `sample.csv`
   - o un ZIP de reanudación exportado previamente

## Estrategia de búsqueda

La búsqueda está dividida en cuatro etapas:

1. `Stage 1`: screening amplio sobre un holdout reducido.
2. `Stage 2`: shortlist con `3-fold CV`.
3. `Stage 3`: refinamiento local alrededor de los mejores seeds.
4. `Stage 4`: desempate por estabilidad usando múltiples holdouts repetidos.

Después se entrena el mejor SVM sobre el split de entrenamiento, se evalúa contra el holdout principal y se genera la submission.

## Por qué esta versión está optimizada para SVM

- concentra la búsqueda en `C`, `gamma` y `PCA`
- evita una grilla cartesiana gigantesca que en Colab sería poco práctica
- reutiliza `StandardScaler + PCA` por fold para no recalcular de más
- guarda resultados por candidato y por fold para que la reanudación sea real

## Cómo usarlo paso a paso en Colab

1. Abre el notebook en Google Colab.
2. Usa runtime de CPU.
3. Si tienes `High-RAM`, mejor.
4. Ejecuta las celdas iniciales.
5. En la celda de upload cambia:
   - `UPLOAD_DATA_FILES = True`
6. Sube:
   - `training.csv`
   - `test.csv`
   - `sample.csv`
7. Si en vez de arrancar desde cero vas a continuar una corrida anterior:
   - deja `UPLOAD_DATA_FILES = False`
   - en la celda de restore cambia `RESTORE_RESUME_BUNDLE = True`
   - sube el ZIP de reanudación
8. Ejecuta la celda de validación.
9. Revisa el perfil:
   - `balanced`: más conservador
   - `aggressive`: más profundo
10. Ejecuta el resto del notebook.

## Persistencia y reanudación

Hay dos modos.

### Modo base: manual

- los checkpoints viven en el runtime temporal de Colab
- al final de cada etapa se genera un ZIP de reanudación
- si quieres continuar luego, descarga ese ZIP antes de perder la sesión

### Modo opcional: Google Drive

Si quieres persistencia adicional:

1. cambia `ENABLE_GOOGLE_DRIVE_PERSISTENCE = True`
2. ejecuta la celda `Optional: enable Google Drive persistence`

## Estructura del workspace dentro de Colab

```text
/content/challenge_svm_ultra_workspace/
├── data/
│   ├── training.csv
│   ├── test.csv
│   └── sample.csv
├── output/
│   └── challenge_06_svm_colab_ultra/
│       └── checkpoints/
├── submissions/
└── exports/
```

## Nota importante

Esta notebook asume que `SVM` está permitido por tus materiales del curso. Si al revisar tus PDFs confirmas que `SVM` no forma parte de los modelos autorizados, no la uses para la entrega final.

## Límite importante

SVM-RBF es una apuesta razonable para intentar mejorar lo que vimos con árboles, pero tampoco puede garantizar `0.90+`. La utilidad real de esta carpeta es que te deja explorar esa familia de modelos de forma seria y reanudable, sin volver a perder horas de trabajo por un reset de Colab.
