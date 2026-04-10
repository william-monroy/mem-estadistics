# Resultados

## Parte 1. Contenido para presentacion

### Slide 1. Resultados del modelo final

- Mejor modelo individual:
  - `signal_features`
  - `validation = 0.9445`
  - `OOF = 0.9403`
- Modelo final:
  - `final_stacking`
  - `meta OOF = 0.9476`
  - `threshold = 0.49`
- Matriz de confusion OOF del stacking final:
  - TN = `4680`
  - FP = `320`
  - FN = `204`
  - TP = `4796`

### Slide 2. Resultado en Kaggle e interpretacion

- Public leaderboard observado:
  - `0.96277`
  - posicion observada: `3`
- `Private leaderboard`:
  - pendiente al cierre del concurso
- El modelo si es exitoso para la competencia
- Pero `accuracy` no refleja completamente el costo real de falsos negativos en mantenimiento predictivo

## Parte 2. Dialogo de explicacion

### Slide 1. Dialogo

En esta slide conviene separar claramente el mejor modelo individual del modelo final. El mejor individual fue `signal_features`, que ya nos llevaba a una validacion de 0.9445 y un OOF de 0.9403. Eso ya era una mejora muy grande frente a todas las fases anteriores. Pero el ensamble final todavia logro exprimir una mejora adicional hasta 0.9476 de meta OOF, lo cual justificó usarlo como submission principal.

La matriz de confusion tambien ayuda a mostrar que no estamos escondiendo el comportamiento por clase. El stacking final redujo los errores totales, y en particular los falsos negativos quedaron en 204 sobre 10,000 observaciones OOF. No es una evaluacion industrial completa, pero para una competencia medida con accuracy es un resultado fuerte y consistente.

### Slide 2. Dialogo

El dato mas importante de competencia es que el submission final llego a `0.96277` en el public leaderboard y nos posicionó en el tercer lugar observado. Eso confirma que la mejora local no se quedó encerrada en validacion, sino que generalizó al test oculto de Kaggle.

Ahora bien, la segunda mitad de la slide debe ser mas critica. El hecho de que la metrica sea alta no significa automaticamente que el sistema ya este listo para deployment industrial. En un sistema real, el costo de dejar pasar una falla puede ser mayor que el de una falsa alarma. Por eso la respuesta correcta es que el modelo es exitoso dentro del marco de la competencia, pero una aplicacion operativa real exigiria analizar recall, falsos negativos y sensibilidad del modelo bajo distintas condiciones.

## Parte 3. Respaldo tecnico y material de apoyo

### Numeros exactos usados en las slides

`signal_features`:

- `validation_accuracy = 0.9445`
- `oof_accuracy = 0.9403`
- matriz de confusion de validacion:
  - `[[947, 53], [58, 942]]`

`final_stacking`:

- `best_meta_oof_accuracy = 0.9476`
- `best_threshold = 0.49`
- matriz de confusion OOF calculada desde `meta_oof_probabilities.csv`:
  - `[[4680, 320], [204, 4796]]`

`Kaggle`:

- `public leaderboard = 0.96277`
- `rank observado = 3`
- `private leaderboard = [pendiente]`

### Archivos de respaldo

- Signal features summary:
  - [summary.json](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results-pre-stack/signal_features_ultra/output/challenge_10_signal_features_colab_ultra/summary.json)
- Final stacking summary:
  - [summary.json](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results-pre-stack/output/challenge_12_final_stacking_colab/summary.json)
- Meta OOF probabilities:
  - [meta_oof_probabilities.csv](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results-pre-stack/output/challenge_12_final_stacking_colab/meta_oof_probabilities.csv)
- Final submission:
  - [challenge_12_final_stacking_colab_submission.csv](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/results-pre-stack/submissions/challenge_12_final_stacking_colab_submission.csv)

### Preguntas que probablemente aparezcan

- ¿por que mostrar OOF y no solo public leaderboard?  
  Porque el leaderboard publico usa solo una parte del test oculto y puede inducir sobreajuste competitivo.

- ¿por que `0.96277` y `0.9476` no son el mismo numero?  
  Porque uno es `public leaderboard` en test oculto de Kaggle y el otro es `meta OOF accuracy` sobre entrenamiento.

- ¿por que decir que el modelo es exitoso pero con salvedades?  
  Porque la metrica de la competencia no mide directamente el costo industrial de los errores.

### Sugerencia visual

- Slide 1:
  - tabla comparativa corta `signal_features` vs `final_stacking`
  - al lado, una confusion matrix tipo heatmap del stacking final
- Slide 2:
  - captura pequeña del leaderboard o una tabla con `public = 0.96277`
  - debajo, una nota breve: `private pendiente`
