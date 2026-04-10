# AWS Ultra Suite

Esta carpeta convierte los notebooks `ultra` en una suite no interactiva orientada a AWS EC2.

El objetivo es claro:

- correr `KNN`, `SVM`, `Bagging` y `Random Forest` sin abrir notebooks
- aprovechar mejor una instancia CPU de AWS
- guardar checkpoints y poder reanudar
- evitar interacción manual durante la ejecución

## Contenido

- [aws_bootstrap.sh](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_ultra_suite/aws_bootstrap.sh): instala dependencias y lanza toda la suite.
- [config.example.json](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_ultra_suite/config.example.json): configuración base.
- [run_all_ultra_models.py](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_ultra_suite/run_all_ultra_models.py): orquestador secuencial.
- [run_single_model.py](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_ultra_suite/run_single_model.py): ejecutor por modelo.
- [model_knn.py](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_ultra_suite/model_knn.py): runner AWS para KNN.
- [model_svm.py](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_ultra_suite/model_svm.py): runner AWS para SVM.
- [model_bagging.py](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_ultra_suite/model_bagging.py): runner AWS para Bagging.
- [model_random_forest.py](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_ultra_suite/model_random_forest.py): runner AWS para Random Forest.
- [common.py](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_ultra_suite/common.py): utilidades compartidas.
- [requirements.txt](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_ultra_suite/requirements.txt): dependencias.

## Mejoras aplicadas respecto a Colab

### Operativas

- ya no dependes de notebooks ni de upload manual
- cada modelo corre en un proceso separado
- el orquestador deja logs por modelo
- la suite genera un resumen global en CSV y JSON
- puedes reanudar porque cada modelo mantiene checkpoints propios

### Específicas para AWS

- `KNN`, `Random Forest` y `Bagging` usan automáticamente casi todos los cores disponibles
- `SVM` paraleliza evaluación por candidato y fold usando `joblib`
- `SVM` usa arrays memmappeados en disco temporal para reducir copias en RAM
- la configuración se resuelve desde un JSON central
- hay soporte opcional para leer datos desde `S3` y subir resultados a `S3`

## Instancias recomendadas

Mi recomendación práctica es:

- `c7a.8xlarge`
  - mejor balance entre costo y tiempo
  - `32 vCPU`, `64 GiB`
- `c7a.4xlarge`
  - opción más barata
  - `16 vCPU`, `32 GiB`
- `c8a.8xlarge`
  - opción más rápida si priorizas terminar antes

No usaría GPU para esta suite.

## Perfil recomendado

La suite soporta:

- `balanced`
- `aws_fast`
- `aws_max`

Si vas a correr todo:

- usa `aws_fast` como punto de partida

Si quieres exprimir más la instancia y aceptas tiempos mayores:

- usa `aws_max`

## Flujo recomendado en EC2

### Opción más simple

1. Copia este repo a la instancia.
2. Verifica que los datos existan en:
   - `challenge/data/training.csv`
   - `challenge/data/test.csv`
   - `challenge/data/sample.csv`
3. Ejecuta:

```bash
bash challenge/aws_ultra_suite/aws_bootstrap.sh challenge/aws_ultra_suite/config.example.json aws_fast
```

Con eso:

- se crea un entorno virtual
- se instalan dependencias
- se ejecutan los cuatro modelos en orden
- se guardan logs, checkpoints, submissions y resúmenes

### Opción con S3

Si quieres cero dependencia del disco local para datos y persistencia extra:

1. copia `config.example.json` a otro archivo, por ejemplo `config.ec2.json`
2. ajusta:
   - `s3_data_uri`
   - `s3_output_uri`
   - `sync_outputs_every_stage`
3. ejecuta:

```bash
bash challenge/aws_ultra_suite/aws_bootstrap.sh challenge/aws_ultra_suite/config.ec2.json aws_fast
```

## Orden de ejecución

El orquestador corre por defecto en este orden:

1. `knn`
2. `svm`
3. `bagging`
4. `random_forest`

Ese orden no es casual:

- primero corre los modelos con mejor potencial práctico
- deja los árboles más al final porque, con lo que ya vimos, parecen menos prometedores para `90+`

## Archivos de salida

Por defecto todo cae en:

```text
challenge/aws_ultra_suite/workspace/
├── output/
├── submissions/
├── exports/
├── logs/
├── tmp/
├── aws_ultra_model_summary.csv
└── aws_ultra_run_summary.json
```

Qué revisar al terminar:

- `workspace/logs/*.log`
- `workspace/output/*/summary.json`
- `workspace/submissions/*.csv`
- `workspace/aws_ultra_model_summary.csv`

## Reanudación

Cada modelo guarda checkpoints propios dentro de su carpeta de `output`.

Si la instancia se interrumpe o paras el proceso:

1. vuelves a lanzar el mismo comando
2. el modelo retoma desde los CSV y JSON ya guardados

No necesitas volver a empezar desde cero.

## Configuración importante

En [config.example.json](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_ultra_suite/config.example.json) puedes controlar:

- `models`
- `profile`
- `continue_on_error`
- `reserve_cpus`
- `max_cpu_fraction`
- `svm_parallel_jobs`
- `svm_cache_mb`
- `knn_n_jobs`
- `ensemble_n_jobs`
- `s3_data_uri`
- `s3_output_uri`

## Recomendaciones técnicas

### Si quieres reducir costo

- corre primero solo:
  - `knn`
  - `svm`
- si esos dos no mejoran lo suficiente, no gastes más en `bagging` y `random_forest`

### Si quieres máxima persistencia

- usa `s3_output_uri`
- y deja `sync_outputs_every_stage = true`

### Si quieres exprimir CPU sin matar la máquina

- deja `reserve_cpus = 2`
- no pongas `max_cpu_fraction = 1.0`

## Restricción importante

La suite incluye `SVM` porque era la familia adicional más razonable para intentar superar a los árboles.

Pero si al revisar tus materiales del curso confirmas que `SVM` no está permitido, exclúyelo del `config`:

```json
"models": ["knn", "bagging", "random_forest"]
```
