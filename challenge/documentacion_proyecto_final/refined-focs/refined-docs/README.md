# Refined Docs

Esta carpeta contiene una segunda capa de documentacion, mas cercana a un dossier final de proyecto que a una simple bitacora tecnica. Su objetivo no es repetir lo ya escrito en [documentacion_proyecto_final](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final), sino reorganizarlo segun el formato pedido en [format.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final/refined-focs/format.md) y dejar trazabilidad suficiente para defender cada decision.

## Criterio de esta carpeta

Estos documentos fueron escritos usando cuatro fuentes de verdad:

- el enunciado y las reglas del challenge en [README.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/README.md)
- la documentacion base ya construida en [documentacion_proyecto_final](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final)
- los artefactos reales del proyecto dentro de `challenge/output`, `challenge/results` y `challenge/results-pre-stack`
- la trazabilidad metodologica acumulada durante todo el proceso de exploracion, descarte, rediseño y ensamble final

La idea central es simple: no basta con decir que un modelo gano; hay que explicar por que se eligio, por que se descartaron alternativas, que evidencia llevo a cambiar de rumbo y que trade-offs se aceptaron.

## Estructura recomendada

1. [01_reporte_principal_formato.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final/refined-focs/refined-docs/01_reporte_principal_formato.md)  
   Version principal del reporte, organizada exactamente en las secciones exigidas por `format.md`.

2. [02_decisiones_metodologicas_detalladas.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final/refined-focs/refined-docs/02_decisiones_metodologicas_detalladas.md)  
   Explica la logica de cada decision importante: por que se probaron ciertos modelos, por que otros se abandonaron y por que el proyecto termino pivotando hacia feature engineering y stacking.

3. [03_trazabilidad_experimental_y_evidencia.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final/refined-focs/refined-docs/03_trazabilidad_experimental_y_evidencia.md)  
   Cronologia cuantitativa del proyecto, con notebooks, artefactos, hiperparametros y resultados que justifican los cambios de estrategia.

4. [04_apendice_codigo_y_diseno_tecnico.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final/refined-focs/refined-docs/04_apendice_codigo_y_diseno_tecnico.md)  
   Fragmentos pequenos de codigo y justificacion de diseño. Sirve para defender implementacion, reproducibilidad y decisiones de ingenieria.

5. [05_reproducibilidad_operacion_y_riesgos.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final/refined-focs/refined-docs/05_reproducibilidad_operacion_y_riesgos.md)  
   Resume los problemas practicos del proyecto: Colab, checkpoints, bundles, ejecuciones largas, AWS/CDK y riesgos metodologicos.

6. [06_plantilla_declaracion_contribucion.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final/refined-focs/refined-docs/06_plantilla_declaracion_contribucion.md)  
   Plantilla lista para completar con tres integrantes.

7. [07_referencias_apa.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final/refined-focs/refined-docs/07_referencias_apa.md)  
   Lista consolidada de referencias en estilo APA, incluyendo material del curso y literatura externa para las tecnicas de preprocesamiento o ensamble.

## Como usar esta carpeta

Si el objetivo es redactar el informe final:

1. usar [01_reporte_principal_formato.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final/refined-focs/refined-docs/01_reporte_principal_formato.md) como esqueleto principal
2. enriquecerlo con la argumentacion de [02_decisiones_metodologicas_detalladas.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final/refined-focs/refined-docs/02_decisiones_metodologicas_detalladas.md)
3. sustentar afirmaciones numericas con [03_trazabilidad_experimental_y_evidencia.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final/refined-focs/refined-docs/03_trazabilidad_experimental_y_evidencia.md)
4. tomar ejemplos de codigo y pipeline desde [04_apendice_codigo_y_diseno_tecnico.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final/refined-focs/refined-docs/04_apendice_codigo_y_diseno_tecnico.md)
5. adaptar [06_plantilla_declaracion_contribucion.md](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/documentacion_proyecto_final/refined-focs/refined-docs/06_plantilla_declaracion_contribucion.md) con los nombres reales del equipo

## Nota importante

El repositorio conserva los artefactos experimentales y el submission final, pero no registra automaticamente el valor exacto del leaderboard final observado en Kaggle. Por eso en varios archivos se deja un marcador de texto para completar manualmente:

- `public leaderboard final`: completar con el valor exacto visto en Kaggle
- `private leaderboard final`: completar cuando cierre el concurso

El resto de los numeros citados en esta carpeta si proviene de artefactos versionados dentro del repo.
