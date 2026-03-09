# Estructura del Proyecto mem-estadistics

## 📁 Vista General

```
mem-estadistics/
├── 📦 src/                          # Código fuente
│   └── nb2pdf/                      # Paquete del convertidor
│       ├── __init__.py              # Exports públicos
│       ├── cli.py                   # CLI principal
│       ├── converter.py             # Lógica de conversión
│       ├── utils.py                 # Utilidades
│       └── config.py                # Configuración
│
├── 📓 notebooks/                    # Notebooks organizados
│   ├── actividades/                 # Actividades entregables
│   │   ├── README.md                # Índice de actividades
│   │   ├── actividad_2/
│   │   │   ├── activity2.ipynb
│   │   │   ├── data/                # Datos de esta actividad
│   │   │   │   ├── Advertising.xlsx
│   │   │   │   └── Colleges_Reduced.xlsx
│   │   │   ├── outputs/             # PDFs generados (git-ignored)
│   │   │   └── README.md
│   │   │
│   │   └── actividad_3/
│   │       ├── activity3.ipynb
│   │       ├── data/
│   │       │   ├── ToyotaCorolla.xlsx
│   │       │   └── Pharmaceuticals.xlsx
│   │       ├── outputs/
│   │       └── README.md
│   │
│   ├── ejemplos/                    # Notebooks de ejemplo
│   │   └── Linear_Models.ipynb
│   │
│   └── exploracion/                 # Notebooks exploratorios
│
├── 📚 docs/                         # Documentación
│   ├── material_teorico/            # Material educativo (PDFs)
│   │   ├── Clustering_Methods.pdf
│   │   └── Principal_Component_Analysis.pdf
│   │
│   └── guias/                       # Guías de uso
│       ├── instalacion.md
│       ├── conversion_pdf.md
│       ├── estructura_proyecto.md   # Este archivo
│       └── QUICKSTART.md
│
├── 🧪 tests/                        # Tests unitarios
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_converter.py
│   └── test_utils.py
│
├── 🔧 scripts/                      # Scripts auxiliares
│   ├── setup.sh                     # Configuración inicial
│   ├── convert_all.sh               # Conversión batch
│   └── clean_outputs.sh             # Limpieza
│
├── ⚙️ config/                       # Configuración
│   └── pdf_templates/               # Templates de PDF
│       └── tec_template.tplx
│
├── 📄 Archivos raíz
│   ├── .gitignore
│   ├── .python-version
│   ├── README.md
│   ├── pyproject.toml
│   ├── uv.lock
│   └── LICENSE
│
└── 🚫 Ignorados por Git
    ├── .venv/                       # Entorno virtual
    ├── **/__pycache__/
    ├── **/.ipynb_checkpoints/
    └── **/outputs/*.pdf             # PDFs generados
```

## 📦 Módulos del Paquete nb2pdf

### `__init__.py`
Exports públicos del paquete:
```python
from nb2pdf import convert_notebook, find_all_notebooks, Config
```

### `cli.py`
- Punto de entrada del CLI
- Parseo de argumentos
- Manejo de errores de usuario

### `converter.py`
- Lógica principal de conversión
- Función `convert_notebook()`
- Conversión múltiple de notebooks
- Manejo de rutas de salida

### `utils.py`
- Validación de rutas
- Búsqueda recursiva de notebooks
- Funciones de formateo
- Utilidades de UI

### `config.py`
- Clase `Config` con configuración global
- Directorios predeterminados
- Rutas excluidas
- Configuración de nbconvert

## 📓 Organización de Notebooks

### Actividades (`notebooks/actividades/`)
Notebooks de tareas entregables. Cada actividad tiene:
- **Notebook principal**: `activityX.ipynb`
- **Datos**: Carpeta `data/` con datasets necesarios
- **Salidas**: Carpeta `outputs/` para PDFs (git-ignored)
- **Documentación**: `README.md` con descripción

### Ejemplos (`notebooks/ejemplos/`)
Material de referencia:
- Ejemplos del curso
- Notebooks de demostración
- Código reutilizable

### Exploración (`notebooks/exploracion/`)
Trabajo exploratorio:
- Pruebas de conceptos
- Experimentos
- Notebooks temporales

## 📚 Documentación

### Material Teórico (`docs/material_teorico/`)
PDFs del curso:
- Conceptos teóricos
- Presentaciones
- Material complementario

**Nota**: Estos PDFs SÍ se trackean en Git (excepción al .gitignore)

### Guías (`docs/guias/`)
Documentación práctica:
- `instalacion.md`: Setup del proyecto
- `conversion_pdf.md`: Uso del convertidor
- `estructura_proyecto.md`: Este archivo
- `QUICKSTART.md`: Referencia rápida

## 🧪 Tests

### Estructura de Tests
```
tests/
├── __init__.py
├── test_config.py       # Tests de configuración
├── test_utils.py        # Tests de utilidades
└── test_converter.py    # Tests de conversión
```

### Ejecutar Tests
```bash
# Todos los tests
uv run pytest

# Con coverage
uv run pytest --cov=src/nb2pdf

# Tests específicos
uv run pytest tests/test_config.py
```

## 🔧 Scripts Auxiliares

### `setup.sh`
Configuración inicial del proyecto:
- Verifica uv
- Instala dependencias
- Verifica LaTeX
- Hace scripts ejecutables

### `convert_all.sh`
Conversión batch de notebooks:
- Convierte todas las actividades
- Genera reporte de éxito/fallos

### `clean_outputs.sh`
Limpieza de archivos generados:
- Elimina PDFs en `outputs/`
- Limpia archivos temporales de LaTeX
- Mantiene estructura de carpetas

## ⚙️ Archivos de Configuración

### `pyproject.toml`
- Metadatos del proyecto
- Dependencias
- Configuración de herramientas (pytest, ruff)
- Entry points del CLI

### `.gitignore`
Archivos excluidos de Git:
- Entornos virtuales (`.venv/`)
- Cache de Python (`__pycache__/`)
- PDFs generados (`**/outputs/*.pdf`)
- Archivos temporales de LaTeX
- **Excepción**: PDFs en `docs/` sí se trackean

### `uv.lock`
- Lock file de dependencias
- Garantiza reproducibilidad
- Generado automáticamente por uv

## 📊 Flujo de Trabajo Típico

### 1. Trabajar en una actividad
```bash
cd notebooks/actividades/actividad_3/
jupyter notebook activity3.ipynb
# ... trabajar en el notebook ...
```

### 2. Convertir a PDF
```bash
cd ../../..  # Volver a raíz
uv run nb2pdf notebooks/actividades/actividad_3/activity3.ipynb
# PDF generado en: notebooks/actividades/actividad_3/outputs/activity3.pdf
```

### 3. Commit de cambios
```bash
git add notebooks/actividades/actividad_3/activity3.ipynb
git commit -m "Completar actividad 3"
git push
```

**Nota**: Los PDFs en `outputs/` no se suben (están en .gitignore)

## 🎯 Convenciones

### Nombrado de Archivos
- Notebooks: `activityX.ipynb`, `lowercase_with_underscores.ipynb`
- Módulos Python: `lowercase_with_underscores.py`
- Clases: `PascalCase`
- Funciones: `snake_case`

### Organización de Datos
- Datos por actividad en `actividad_X/data/`
- Nunca datos en la raíz del proyecto
- Considerar .gitignore para datos grandes

### PDFs
- Generados: `outputs/` (git-ignored)
- Material del curso: `docs/material_teorico/` (tracked)
- Un PDF por notebook

## 💡 Mejores Prácticas

1. **Mantener notebooks ejecutados**: Facilita la conversión a PDF
2. **Usar rutas relativas**: Para máxima portabilidad
3. **Documentar código**: Comentarios claros en notebooks
4. **Tests actualizados**: Agregar tests para nuevas funcionalidades
5. **README por actividad**: Documentar objetivo y resultados

## 🔗 Enlaces Rápidos

- [README Principal](../../README.md)
- [Guía de Instalación](./instalacion.md)
- [Guía de Conversión PDF](./conversion_pdf.md)
- [Quick Start](./QUICKSTART.md)
