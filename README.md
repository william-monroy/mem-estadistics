# mem-estadistics

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Repositorio de actividades de estadística con herramientas profesionales para conversión de Jupyter Notebooks a PDF.

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Instalación Rápida](#-instalación-rápida)
- [Uso](#-uso)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Actividades](#-actividades)
- [Documentación](#-documentación)
- [Desarrollo](#-desarrollo)
- [Scripts Auxiliares](#-scripts-auxiliares)
- [Licencia](#-licencia)

## ✨ Características

- 🔄 **Conversión automatizada** de notebooks a PDF con preservación de formato
- 📁 **Organización profesional** por actividades con datos separados
- 🧪 **Tests unitarios** para garantizar calidad del código
- 📚 **Documentación completa** con guías paso a paso
- 🛠️ **Scripts auxiliares** para tareas comunes
- 🎨 **Configuración personalizable** para templates de PDF
- ⚡ **CLI intuitivo** con manejo robusto de errores

## 🚀 Instalación Rápida

### Requisitos Previos
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (gestor de paquetes)
- LaTeX (para conversión a PDF)

### Setup Completo

```bash
# 1. Clonar el repositorio
git clone https://github.com/william-monroy/mem-estadistics.git
cd mem-estadistics

# 2. Ejecutar instalación automática (Linux/macOS)
./scripts/setup.sh

# O manualmente:
uv sync                    # Instalar dependencias
chmod +x scripts/*.sh      # Hacer scripts ejecutables
```

### Instalar LaTeX

**macOS:**
```bash
brew install --cask mactex-no-gui
eval "$(/usr/libexec/path_helper)"
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install texlive-xetex texlive-fonts-recommended texlive-plain-generic
```

**Windows:**
Descargar [MiKTeX](https://miktex.org/download)

📖 [Guía detallada de instalación](docs/guias/instalacion.md)

## 💻 Uso

### Convertir Notebooks a PDF

```bash
# Convertir un notebook específico
uv run nb2pdf notebooks/actividades/actividad_3/activity3.ipynb

# Convertir con nombre personalizado
uv run nb2pdf notebooks/actividades/actividad_3/activity3.ipynb -o Mi_Reporte.pdf

# Convertir todos los notebooks en un directorio
uv run nb2pdf --all --directory notebooks/actividades/actividad_3/

# Modo verbose (para debugging)
uv run nb2pdf activity3.ipynb --verbose

# Ver ayuda
uv run nb2pdf --help
```

### Usando Scripts Auxiliares

```bash
# Convertir todas las actividades
./scripts/convert_all.sh

# Limpiar archivos generados
./scripts/clean_outputs.sh

# Ver qué se eliminaría sin hacer cambios
./scripts/clean_outputs.sh --dry-run
```

📖 [Guía completa de conversión](docs/guias/conversion_pdf.md)

## 📁 Estructura del Proyecto

```
mem-estadistics/
├── 📦 src/nb2pdf/              # Paquete del convertidor
│   ├── cli.py                  # CLI principal
│   ├── converter.py            # Lógica de conversión
│   ├── utils.py                # Utilidades
│   └── config.py               # Configuración
│
├── 📓 notebooks/               # Notebooks organizados
│   ├── actividades/            # Actividades entregables
│   │   ├── actividad_2/        # Análisis de datos
│   │   └── actividad_3/        # Predicción de precios
│   ├── ejemplos/               # Notebooks de referencia
│   └── exploracion/            # Trabajo exploratorio
│
├── 📚 docs/                    # Documentación
│   ├── material_teorico/       # PDFs del curso
│   └── guias/                  # Guías de uso
│
├── 🧪 tests/                   # Tests unitarios
├── 🔧 scripts/                 # Scripts auxiliares
└── ⚙️ config/                  # Configuración avanzada
```

📖 [Estructura completa del proyecto](docs/guias/estructura_proyecto.md)

## 📚 Actividades

### [Actividad 2](notebooks/actividades/actividad_2/) - Análisis de Datos
Análisis usando regresión lineal con datasets de publicidad y colegios.

**Datasets:** `Advertising.xlsx` (14 KB), `Colleges_Reduced.xlsx` (96 KB)

### [Actividad 3](notebooks/actividades/actividad_3/) - Predicción de Precios
Modelo de regresión lineal múltiple para predecir precios de autos usados.

**Datasets:** `ToyotaCorolla.xlsx` (193 KB, 1,436 registros)

**Resultados:**
- RMSE: 1,432 euros (K-fold CV con K=13)
- R²: 87% (alta capacidad explicativa)
- Variables significativas: Age_08_04, KM, HP, Weight

📖 Ver [README de actividades](notebooks/actividades/README.md) para más detalles

## 📖 Documentación

### Guías Principales
- 📥 [Instalación](docs/guias/instalacion.md) - Setup completo del proyecto
- 🔄 [Conversión a PDF](docs/guias/conversion_pdf.md) - Uso del convertidor
- 📁 [Estructura](docs/guias/estructura_proyecto.md) - Organización del código
- ⚡ [Quick Start](docs/guias/QUICKSTART.md) - Referencia rápida

### Material Teórico
- [Clustering Methods](docs/material_teorico/Clustering_Methods.pdf) (11 MB)
- [Principal Component Analysis](docs/material_teorico/Principal_Component_Analysis.pdf) (3 MB)

## 🛠️ Desarrollo

### Ejecutar Tests

```bash
# Todos los tests
uv run pytest

# Con cobertura
uv run pytest --cov=src/nb2pdf --cov-report=html

# Tests específicos
uv run pytest tests/test_config.py -v
```

### Agregar Dependencias

```bash
# Dependencia de producción
uv add nombre-paquete

# Dependencia de desarrollo
uv add --dev pytest ruff
```

### Linting y Formateo

```bash
# Ejecutar linter
uv run ruff check .

# Auto-formatear código
uv run ruff format .
```

### Estructura de Tests
```python
# tests/test_converter.py
from nb2pdf import convert_notebook

def test_convert_notebook(tmp_path):
    notebook = tmp_path / "test.ipynb"
    notebook.write_text("{}")
    assert convert_notebook(notebook)
```

## 🔧 Scripts Auxiliares

### `setup.sh`
Configuración inicial automática
```bash
./scripts/setup.sh
```
- ✅ Verifica uv instalado
- 📦 Instala dependencias
- 🔍 Verifica LaTeX
- 🔧 Hace scripts ejecutables

### `convert_all.sh`
Conversión batch de todas las actividades
```bash
./scripts/convert_all.sh
```
- 🔄 Convierte todos los notebooks en `actividades/`
- 📊 Reporta éxito/fallo por archivo
- ✨ Salida formateada con emojis

### `clean_outputs.sh`
Limpieza de archivos generados
```bash
./scripts/clean_outputs.sh          # Con confirmación
./scripts/clean_outputs.sh -f       # Sin confirmación
./scripts/clean_outputs.sh --dry-run # Solo mostrar
```
- 🗑️ Elimina PDFs en `outputs/`
- 🧹 Limpia archivos temporales de LaTeX
- 📋 Remueve checkpoints de Jupyter

## 📊 Estadísticas del Proyecto

- **Notebooks**: 3 (2 actividades + 1 ejemplo)
- **Tests**: 15+ casos de prueba
- **Documentación**: 5+ guías completas
- **Líneas de código**: ~1,000 (sin contar notebooks)
- **Cobertura de tests**: 85%+

## 🤝 Contribuir

Este es un proyecto educativo, pero las sugerencias son bienvenidas:

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 🐛 Reportar Issues

Si encuentras un bug o tienes una sugerencia:
1. Revisa [issues existentes](https://github.com/william-monroy/mem-estadistics/issues)
2. Abre un nuevo issue con descripción detallada
3. Incluye logs de error si aplica

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver [LICENSE](LICENSE) para más detalles.

## 👤 Autor

**William Frank Monroy Mamani**

- ID: A00829796
- GitHub: [@william-monroy](https://github.com/william-monroy)
- Email: A00829796@tec.mx

## 🙏 Agradecimientos

- Tecnológico de Monterrey - Curso de Estadística
- Comunidad de Jupyter y nbconvert
- Astral-sh por uv

## 📚 Referencias

- [nbconvert Documentation](https://nbconvert.readthedocs.io/)
- [uv Documentation](https://github.com/astral-sh/uv)
- [Jupyter Documentation](https://jupyter.org/documentation)

---

⭐ **¿Te resultó útil este proyecto? ¡Dale una estrella en GitHub!**