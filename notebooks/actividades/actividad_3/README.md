# Actividad 3 - Predicción de Precios de Autos Usados

## 📋 Descripción

Desarrollo de un modelo de regresión lineal múltiple para predecir el precio de autos Toyota Corolla usados basándose en características como edad, kilometraje, potencia, etc.

## 🎯 Objetivos

1. Identificar y eliminar predictores con multicolinealidad
2. Entrenar un modelo de regresión lineal múltiple
3. Evaluar significancia estadística de los coeficientes
4. Validar el modelo usando K-fold cross validation
5. Optimizar el valor de K para la validación cruzada

## 📊 Datasets

### `ToyotaCorolla.xlsx`
- **Registros**: 1,436 autos
- **Atributos**: 38 características
- **Target**: Precio (Price) en euros
- **Predictores iniciales**: Age_08_04, KM, HP, CC, Quarterly_Tax, Weight
- **Tamaño**: ~193 KB

### `Pharmaceuticals.xlsx`
- Datos adicionales para análisis complementario
- **Tamaño**: ~11 KB

## 🚀 Uso

### Ejecutar el notebook
```bash
# Desde la raíz del proyecto
jupyter notebook notebooks/actividades/actividad_3/activity3.ipynb
```

### Convertir a PDF
```bash
# Desde la raíz del proyecto
uv run nb2pdf notebooks/actividades/actividad_3/activity3.ipynb

# El PDF se generará en:
# notebooks/actividades/actividad_3/outputs/activity3.pdf
```

## 📈 Contenido del Notebook

### 0. Carga de librerías
- pandas, numpy, sklearn, statsmodels, matplotlib, seaborn

### 1. Lectura de datos
- Carga del dataset ToyotaCorolla.xlsx

### 2. Creación de subsets
- Variable objetivo (Y): Price
- Predictores (X): Age_08_04, KM, HP, CC, Quarterly_Tax, Weight

### 3. Preprocesamiento
- **3.A**: Identificación de correlaciones altas (threshold: 0.60)
- **3.B**: Eliminación de multicolinealidad (se removió Quarterly_Tax)

### 4. Entrenamiento e interpretación
- **4.A**: Ajuste del modelo (train/test split 75/25)
- **4.B**: Coeficientes del modelo
- **4.C**: Visualización de coeficientes
- **4.D**: Pruebas t de significancia (α = 0.05)
- **4.E**: Interpretación de coeficientes significativos
- **4.F**: Gráficas de predicciones

### 5. Evaluación del modelo
- **5.A**: Búsqueda del K óptimo para cross validation
  - Resultado: **K = 13** (RMSE mínimo: 1,510.94 euros)
- **5.B**: K-fold CV con K óptimo
  - RMSE promedio: **1,431.58 euros**
  - Desviación estándar: 483.23 euros
- **5.C**: Evaluación final train/validation

## 🔍 Resultados Clave

### Modelo Final
- **Variables predictoras**: Age_08_04, KM, HP, CC, Weight (5 variables)
- **Variable eliminada**: Quarterly_Tax (correlación 0.63 con Weight)

### Métricas de Desempeño
- **RMSE Training**: 1,363 euros
- **RMSE Validation**: 1,306 euros
- **R² Training**: 85.86%
- **R² Validation**: 87.00%
- **RMSE K-fold CV (K=13)**: 1,431.58 ± 483.23 euros

### Hallazgos
✅ El modelo generaliza bien (sin sobreajuste)  
✅ Todos los coeficientes son significativos (p < 0.05) excepto CC  
✅ K=13 es el valor óptimo para cross validation  
✅ Weight es un mejor predictor que Quarterly_Tax (correlación 2.65x mayor)

## 📁 Estructura

```
actividad_3/
├── activity3.ipynb      # Notebook principal
├── data/                # Datasets
│   ├── ToyotaCorolla.xlsx
│   └── Pharmaceuticals.xlsx
├── outputs/             # Resultados (git-ignored)
│   ├── activity3.pdf
│   └── *.png           # Gráficas generadas
└── README.md           # Este archivo
```

## 📚 Referencias

- Dataset original: ToyotaCorolla.xlsx (1,436 registros, late summer 2004, Netherlands)
- Train/Test split: 75/25 con random_state=301655
- Threshold de multicolinealidad: 0.60

## 👤 Autor

**William Frank Monroy Mamani**  
ID: A00829796  
GitHub: [william-monroy](https://github.com/william-monroy)
