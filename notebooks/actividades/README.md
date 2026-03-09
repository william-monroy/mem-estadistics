# Actividades de Estadística

Este directorio contiene las actividades del curso de estadística.

## 📚 Índice de Actividades

### [Actividad 2](./actividad_2/)
Análisis de datos con modelos lineales.

**Temas:**
- Regresión lineal
- Análisis de datos publicitarios
- Análisis de colegios

**Datasets:**
- `Advertising.xlsx`
- `Colleges_Reduced.xlsx`

### [Actividad 3](./actividad_3/)
Predicción de precios de autos usados usando regresión lineal múltiple.

**Temas:**
- Regresión lineal múltiple
- Validación cruzada K-fold
- Evaluación de modelos
- Multicolinealidad

**Datasets:**
- `ToyotaCorolla.xlsx`
- `Pharmaceuticals.xlsx`

## 🚀 Cómo usar

1. **Abrir un notebook:** Navega a la carpeta de la actividad y abre el archivo `.ipynb`
2. **Ejecutar las celdas:** Ejecuta las celdas en orden
3. **Convertir a PDF:** Usa el convertidor desde la raíz del proyecto:
   ```bash
   uv run nb2pdf notebooks/actividades/actividad_3/activity3.ipynb
   ```

## 📂 Estructura de cada actividad

```
actividad_X/
├── activityX.ipynb      # Notebook principal
├── data/                # Datos necesarios
│   └── *.xlsx
└── outputs/             # PDFs y gráficas generadas (git-ignored)
    └── activityX.pdf
```

## 📊 Resultados

Los PDFs generados se guardan automáticamente en la carpeta `outputs/` de cada actividad.

Los archivos en `outputs/` están ignorados por Git para mantener el repositorio limpio.
