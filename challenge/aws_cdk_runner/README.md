# AWS CDK Runner

Esta carpeta envuelve la suite [aws_ultra_suite](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_ultra_suite) en un stack mínimo de AWS CDK para que puedas:

- desplegar una instancia EC2 preparada desde tu PC local
- empaquetar automáticamente el código y los CSV del challenge como assets
- ejecutar la suite sin interacción manual dentro de AWS
- persistir resultados en S3 durante la corrida
- descargar outputs al final
- destruir casi todo con un solo `cdk destroy`

## Qué crea

El stack crea:

- una `EC2` CPU-oriented con Amazon Linux 2023
- una `VPC` mínima con solo subred pública
- un `Security Group` sin reglas de entrada
- un `IAM Role` con `SSM` y permisos de lectura/escritura sobre S3
- un bucket `S3` temporal para resultados
- assets de CDK para:
  - [challenge/aws_ultra_suite](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_ultra_suite)
  - [challenge/data](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/data) si `include_data_asset = true`

## Qué no hace

- no usa SageMaker
- no abre puertos SSH
- no requiere que subas archivos manualmente a la instancia
- no conserva el bucket tras `destroy`; si quieres resultados, descárgalos antes

## Flujo operativo

### 1. Preparar config

Duplica [deploy_config.example.json](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_cdk_runner/deploy_config.example.json) como `deploy_config.json` dentro de esta carpeta y ajusta al menos:

- `aws_region`
- `instance_type`
- `profile`
- `models`

Parámetros importantes:

- `include_data_asset = true`
  - empaqueta `challenge/data` dentro del deploy
  - es la opción más simple
- `s3_data_uri`
  - úsalo solo si no quieres subir los CSV como asset
- `terminate_on_success = true`
  - la instancia se apaga sola al terminar y, como el comportamiento está configurado a `terminate`, desaparece
- `terminate_on_failure = false`
  - si falla, la instancia queda viva para inspección por `SSM`

## 2. Desplegar desde local

Desde esta carpeta:

```bash
./scripts/deploy.sh
```

O indicando un config concreto:

```bash
./scripts/deploy.sh ./deploy_config.json
```

Ese script hace esto:

1. crea `.venv`
2. instala dependencias Python del CDK
3. instala el CLI de CDK vía `npm`
4. ejecuta `cdk bootstrap`
5. ejecuta `cdk deploy`

## 3. Monitorear

Puedes revisar el estado del stack y los outputs con:

```bash
./scripts/status.sh
```

La salida te mostrará:

- `StackStatus`
- `InstanceId`
- bucket y prefijo de resultados
- estado actual de la EC2
- los últimos objetos escritos en S3

Si necesitas entrar a la instancia:

```bash
aws ssm start-session --target <InstanceId>
```

Ese comando también sale como output del stack.

## 4. Descargar resultados

Antes de destruir el stack, descarga los resultados:

```bash
./scripts/download_results.sh
```

Por defecto caerán en:

```text
challenge/aws_cdk_runner/downloads/<StackName>-<timestamp>/
```

## 5. Destruir

Cuando ya no necesites nada en AWS:

```bash
./scripts/destroy.sh
```

Eso eliminará:

- EC2
- VPC
- Security Group
- IAM Role/Profile del stack
- bucket de resultados y su contenido

## Importante sobre el cleanup

`cdk destroy` limpia el stack, pero no elimina automáticamente el stack de bootstrap `CDKToolkit` que usa CDK para manejar assets. Eso es normal y está fuera de este stack.

## Config recomendada

Punto de partida razonable:

```json
{
  "instance_type": "c7a.8xlarge",
  "profile": "aws_fast",
  "models": ["knn", "svm", "bagging", "random_forest"],
  "sync_outputs_every_stage": true,
  "include_data_asset": true,
  "terminate_on_success": true,
  "terminate_on_failure": false
}
```

## Estructura

- [app.py](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_cdk_runner/app.py): entrypoint del CDK app.
- [ultra_runner_cdk/config.py](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_cdk_runner/ultra_runner_cdk/config.py): carga y valida el config.
- [ultra_runner_cdk/stack.py](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_cdk_runner/ultra_runner_cdk/stack.py): define la infraestructura.
- [ultra_runner_cdk/user_data.py](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_cdk_runner/ultra_runner_cdk/user_data.py): genera el `UserData`.
- [scripts/deploy.sh](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_cdk_runner/scripts/deploy.sh): despliegue local.
- [scripts/status.sh](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_cdk_runner/scripts/status.sh): estado del stack y de la instancia.
- [scripts/download_results.sh](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_cdk_runner/scripts/download_results.sh): descarga de resultados desde S3.
- [scripts/destroy.sh](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/aws_cdk_runner/scripts/destroy.sh): destrucción del stack.
