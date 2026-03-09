# Actividad 2 - Análisis de Datos

## 📋 Descripción

Análisis de datos usando regresión lineal para dos casos de estudio:
1. **Advertising**: Análisis de la efectividad publicitaria
2. **Colleges**: Análisis de datos de colegios

## 📊 Datasets

### `Advertising.xlsx`
- Descripción: Datos de campañas publicitarias
- Tamaño: ~14 KB

### `Colleges_Reduced.xlsx`
- Descripción: Datos de colegios (versión reducida)
- Tamaño: ~96 KB

## 🚀 Uso

### Ejecutar el notebook
```bash
# Desde la raíz del proyecto
jupyter notebook notebooks/actividades/actividad_2/activity2.ipynb
```

### Convertir a PDF
```bash
# Desde la raíz del proyecto
uv run nb2pdf notebooks/actividades/actividad_2/activity2.ipynb

# El PDF se generará en:
# notebooks/actividades/actividad_2/outputs/activity2.pdf
```

## 📈 Contenido

- Carga y exploración de datos
- Regresión lineal simple y múltiple
- Análisis de correlación
- Visualización de resultados
- Interpretación de coeficientes

## 📁 Estructura

```
actividad_2/
├── activity2.ipynb      # Notebook principal
├── data/                # Datos
│   ├── Advertising.xlsx
│   └── Colleges_Reduced.xlsx
├── outputs/             # Resultados (git-ignored)
│   ├── activity2.pdf
│   └── *.png           # Gráficas
└── README.md           # Este archivo
```
