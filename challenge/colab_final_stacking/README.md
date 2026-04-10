# Final Stacking for Colab

Esta carpeta contiene la variante final de stacking para Google Colab.

A diferencia de la notebook generica de stacking, esta version esta enfocada en tu situacion real:

- `signal_features` como base learner fuerte
- `knn_cleaning` como modelo de diversidad
- `svm_preprocessing` como extra opcional

## Contenido

- `Challenge_12_Final_Stacking_Colab.ipynb`
- `package_results_pre_stack.py`
- `requirements.txt`
- `workspace_template/README.md`

## Flujo recomendado

1. Desde tu maquina local, empaqueta tus corridas base:

   ```bash
   python3 challenge/colab_final_stacking/package_results_pre_stack.py
   ```

2. Eso generara ZIPs dentro de `challenge/colab_final_stacking/upload_bundles/`.

3. Sube `Challenge_12_Final_Stacking_Colab.ipynb` a Google Colab.

4. En la celda `Optional: upload one or more model-result ZIP bundles` cambia:

   ```python
   UPLOAD_RESULT_BUNDLES = True
   ```

5. Sube al menos dos ZIPs:
   - `knn_cleaning_ultra_for_final_stacking.zip`
   - `signal_features_ultra_for_final_stacking.zip`

6. Ejecuta el resto de la notebook.

7. Si despues completas SVM, genera su ZIP y vuelve a correr el stacking incluyendo ese bundle.

## Que busca internamente

- `weighted average` con busqueda de pesos
- busqueda de `threshold`
- `LogisticRegression` como meta-modelo

## Artefactos finales

- `output/challenge_12_final_stacking_colab/summary.json`
- `output/challenge_12_final_stacking_colab/meta_oof_probabilities.csv`
- `output/challenge_12_final_stacking_colab/meta_test_probabilities.csv`
- `submissions/challenge_12_final_stacking_colab_submission.csv`
