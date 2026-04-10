# Apendice de Codigo y Diseno Tecnico

Este apendice incluye fragmentos pequeños de codigo que ayudan a explicar como se implementaron las ideas metodologicas principales. No intenta copiar notebooks completas; se centra en bloques cuya logica fue decisiva.

## 1. Auditoria inicial del dataset

El proyecto partio de una auditoria simple, pero importante. Un patron representativo del codigo base era:

```python
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(train_df.shape)
print(test_df.shape)
print(train_df["class"].value_counts().to_dict())
print(train_df.isna().sum().sum(), test_df.isna().sum().sum())
print(train_df.duplicated().sum())
```

### Por que este bloque importa

Parece trivial, pero resolvia preguntas metodologicas esenciales:

- si habia missing, el pipeline debia incluir imputacion;
- si habia desbalance, habia que controlar sampling o metricas;
- si habia duplicados, habia que evitar leakage o sesgo.

El hecho de que estos chequeos salieran "limpios" justifico todo el enfoque posterior.

## 2. Limpieza de instancias para KNN

Fuente:

- [generate_colab_advanced_strategies.py](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/scripts/generate_colab_advanced_strategies.py)

Fragmento representativo:

```python
def apply_cleaning(X_input: np.ndarray, y_input: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray, dict]:
    if method == "none":
        return X_input, y_input, {"removed_rows": 0}

    if method.startswith("lof_"):
        contamination = float(method.split("_")[1])
        scaled = StandardScaler().fit_transform(X_input)
        keep_mask = np.ones(len(X_input), dtype=bool)
        for cls in np.unique(y_input):
            idx = np.where(y_input == cls)[0]
            if len(idx) < 12:
                continue
            n_neighbors = min(25, len(idx) - 1)
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            pred = lof.fit_predict(scaled[idx])
            keep_mask[idx] = pred == 1
        return X_input[keep_mask], y_input[keep_mask], {"removed_rows": int((~keep_mask).sum())}
```

### Por que el diseño fue correcto

1. `LOF` se aplico por clase, no globalmente.  
   Eso evita que puntos validos de una clase queden penalizados solo por estar rodeados por la otra.

2. Se escalo antes de medir vecindad.  
   Un detector basado en distancias sin escalado previo habria sido dificil de defender.

3. Se devolvio `removed_rows`.  
   No bastaba con saber si la accuracy subia; habia que entender cuan agresiva era cada limpieza.

## 3. Construccion del banco de features de señal

Fuente:

- [generate_colab_advanced_strategies.py](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/scripts/generate_colab_advanced_strategies.py)

Fragmento representativo:

```python
def build_feature_bank(X_input: np.ndarray) -> dict[str, np.ndarray]:
    X = X_input.astype(np.float64)
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    rms = np.sqrt(np.mean(X ** 2, axis=1))
    peak_to_peak = X.max(axis=1) - X.min(axis=1)

    centered = X - mean[:, None]
    skew = np.mean(centered ** 3, axis=1) / ((std ** 3) + 1e-12)
    kurt = np.mean(centered ** 4, axis=1) / ((std ** 4) + 1e-12)

    fft_mag = np.abs(np.fft.rfft(X, axis=1))
    fft_power = fft_mag ** 2
    spectral_entropy = -np.sum(
        (fft_power / (fft_power.sum(axis=1, keepdims=True) + 1e-12))
        * np.log(fft_power / (fft_power.sum(axis=1, keepdims=True) + 1e-12) + 1e-12),
        axis=1,
    )

    return {
        "basic": basic.astype(np.float32),
        "fft": np.hstack([basic, fft_features]).astype(np.float32),
        "all": np.hstack([basic, fft_features, segment_features]).astype(np.float32),
    }
```

### Por que este bloque cambio el proyecto

Este codigo materializa el cambio conceptual mas fuerte:

- antes: el problema se trataba como `200` columnas tabulares;
- despues: se describia como una señal con estructura temporal y frecuencial.

El diseño por bancos (`basic`, `fft`, `all`) tambien fue deliberado. No se asumio que "mas features" implicaba automaticamente mejor resultado; se construyeron niveles crecientes para medir el valor de cada grupo.

## 4. Fusion con `raw PCA`

Fragmento representativo:

```python
if params["feature__raw_pca"] > 0:
    raw_scaler = StandardScaler()
    X_fit_raw_scaled = raw_scaler.fit_transform(X_fit)
    X_eval_raw_scaled = raw_scaler.transform(X_eval)
    raw_pca = PCA(
        n_components=min(params["feature__raw_pca"], X_fit_raw_scaled.shape[0], X_fit_raw_scaled.shape[1]),
        random_state=RANDOM_STATE,
    )
    X_fit_raw = raw_pca.fit_transform(X_fit_raw_scaled).astype(np.float32)
    X_eval_raw = raw_pca.transform(X_eval_raw_scaled).astype(np.float32)
    X_fit_base = np.hstack([X_fit_base, X_fit_raw]).astype(np.float32)
    X_eval_base = np.hstack([X_eval_base, X_eval_raw]).astype(np.float32)
```

### Justificacion

Esta fusion resuelve una tension metodologica interesante:

- las features ingenierizadas resumen bien el dominio;
- pero la matriz cruda puede seguir conteniendo señal complementaria no capturada por esas estadisticas.

En vez de elegir una u otra, se uso una pequeña proyeccion `raw PCA = 32` para mezclar ambas fuentes. El mejor modelo final de `signal_features` justamente aprovecho esa combinacion.

## 5. Descubrimiento automatico de artefactos para stacking

Fuente:

- [generate_colab_final_stacking.py](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/scripts/generate_colab_final_stacking.py)

Fragmento representativo:

```python
def discover_probability_artifacts() -> list[dict]:
    records = []
    for oof_path in sorted(ARTIFACT_SEARCH_ROOT.rglob("oof_probabilities.csv")):
        model_root = oof_path.parent
        test_path = model_root / "test_probabilities.csv"
        summary_path = model_root / "summary.json"
        if not test_path.exists():
            continue
        ...
```

### Por que este diseño importo

El stacking final no se diseñó como una notebook que reentrena todo otra vez. Se diseño como un consumidor de artefactos persistidos:

- `oof_probabilities.csv`
- `test_probabilities.csv`
- `summary.json`

Eso redujo recomputo, facilito reanudar y desacoplo completamente el entrenamiento base del ensamble final.

## 6. Busqueda de pesos y threshold

Fragmento representativo:

```python
def evaluate_weighted_candidate(meta_df, subset, weights, threshold_grid):
    prob_1 = weighted_probabilities(meta_df, subset, weights)
    threshold, accuracy = find_best_threshold(
        meta_df["y_true"].to_numpy(dtype=np.int8),
        prob_1,
        threshold_grid,
    )
    return {
        "meta_family": "weighted_average",
        "base_models": subset,
        "weights": weights,
        "best_threshold": threshold,
        "meta_oof_accuracy": accuracy,
    }
```

### Que problema resuelve

En muchos ensambles se fijan sin discutir:

- pesos iguales;
- threshold `0.5`.

Aqui ambas decisiones se dejaron abiertas. Eso fue importante porque la mejor mezcla final no fue `0.5 / 0.5`, sino `0.425 / 0.575`, y el mejor threshold tampoco fue `0.5`, sino `0.49`.

## 7. Meta-modelo logit como control

Fragmento representativo:

```python
def generate_logreg_oof_probabilities(meta_df, subset, c_value):
    X_meta = meta_df[subset].to_numpy(dtype=np.float32)
    y_meta = meta_df["y_true"].to_numpy(dtype=np.int8)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_prob_1 = np.zeros(len(meta_df), dtype=np.float32)
    for fit_idx, eval_idx in cv.split(X_meta, y_meta):
        model = LogisticRegression(C=float(c_value), max_iter=2000)
        model.fit(X_meta[fit_idx], y_meta[fit_idx])
        oof_prob_1[eval_idx] = model.predict_proba(X_meta[eval_idx])[:, 1].astype(np.float32)
    return oof_prob_1
```

### Por que se incluyo aunque no ganara

El promedio ponderado gano por poco, pero habia que demostrar que una capa meta-modelo no aportaba una mejora relevante. Incluir esta comparacion fue importante para defender que la solucion final simple no fue elegida por comodidad, sino por evidencia.

## 8. Empaquetado para la fase final

Fuente:

- [package_results_pre_stack.py](/Users/williamfrankmonroymamani/Documents/mem/stadistics/mem-estadistics/challenge/colab_final_stacking/package_results_pre_stack.py)

Fragmento representativo:

```python
def should_include_run(run_dir: Path) -> bool:
    output_dir = run_dir / "output"
    return any(output_dir.rglob("oof_probabilities.csv")) and any(output_dir.rglob("test_probabilities.csv"))
```

### Por que este helper fue util

La fase final dependia de subir a Colab solo corridas con artefactos completos. Este tipo de helper redujo errores humanos, evito subir carpetas incompletas y estabilizo el flujo `package -> upload -> stack`.

## 9. Diseño por etapas y checkpoints

Aunque muchas de las notebooks se ejecutaban en Colab, no se diseñaron como notebooks de una sola corrida. Las variantes `Ultra` se estructuraron en etapas para:

- guardar progreso parcial;
- permitir reanudacion;
- limitar perdida de trabajo por reinicios de sesion;
- separar screening rapido de evaluacion cara.

Ese diseño fue importante especialmente en:

- `Challenge_04_RandomForest_Colab_Ultra.ipynb`
- `Challenge_09_SVM_Preprocessing_Colab_Ultra.ipynb`
- `Challenge_12_Final_Stacking_Colab.ipynb`

## 10. Lecciones tecnicas de implementacion

Tres decisiones de diseño explican buena parte de la robustez final del proyecto:

1. separar entrenamiento base y stacking;
2. persistir siempre artefactos tabulares intermedios;
3. tratar el pipeline como un sistema reproducible y no solo como una secuencia de notebooks.

Estas decisiones no son accesorias. Fueron parte de la razon por la que el proyecto pudo converger a una solucion final reproducible a pesar de las restricciones computacionales.
