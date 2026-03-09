# Guía de Conversión de Notebooks a PDF

## 🎯 Introducción

Esta guía explica cómo usar la herramienta `nb2pdf` para convertir notebooks de Jupyter a documentos PDF.

## 📋 Comandos Básicos

### Convertir un notebook específico

```bash
# Desde la raíz del proyecto
uv run nb2pdf notebooks/actividades/actividad_3/activity3.ipynb
```

### Especificar nombre de salida

```bash
uv run nb2pdf notebooks/actividades/actividad_3/activity3.ipynb -o Actividad3_William_Monroy.pdf
```

### Convertir todos los notebooks de un directorio

```bash
# Convertir todos en el directorio actual
uv run nb2pdf --all

# Convertir todos en un directorio específico
uv run nb2pdf --all --directory notebooks/actividades/actividad_3/
```

### Modo verbose (detallado)

```bash
uv run nb2pdf activity3.ipynb --verbose
```

## 📂 Ubicación de PDFs Generados

### Para notebooks en `actividades/`
Los PDFs se generan automáticamente en la carpeta `outputs/`:

```
actividad_3/
├── activity3.ipynb
└── outputs/
    └── activity3.pdf  ← Aquí se genera
```

### Para otros notebooks
El PDF se genera en el mismo directorio que el notebook:

```
ejemplos/
├── Linear_Models.ipynb
└── Linear_Models.pdf  ← Aquí se genera
```

## 🔧 Opciones Avanzadas

### Ver ayuda completa

```bash
uv run nb2pdf --help
```

### Convertir y abrir automáticamente (macOS)

```bash
uv run nb2pdf activity3.ipynb && open notebooks/actividades/actividad_3/outputs/activity3.pdf
```

### Convertir sin activar entorno virtual

```bash
# Activar entorno
source .venv/bin/activate

# Ejecutar directamente
nb2pdf activity3.ipynb

# Desactivar
deactivate
```

## 📝 Antes de Convertir

### ✅ Checklist

1. **Ejecutar todas las celdas**: Asegúrate de que el notebook esté completamente ejecutado
2. **Verificar gráficas**: Las visualizaciones deben estar renderizadas
3. **Guardar el notebook**: Guarda todos los cambios
4. **Limpiar outputs no deseados**: Limpia outputs de prueba si es necesario

### Ejecutar todas las celdas desde CLI

```bash
# Ejecutar y limpiar outputs
uv run jupyter nbconvert --to notebook --execute --inplace activity3.ipynb

# Luego convertir a PDF
uv run nb2pdf activity3.ipynb
```

## 🎨 Personalización del PDF

### Configurar template personalizado

Edita `src/nb2pdf/config.py`:

```python
class Config:
    NBCONVERT_TEMPLATE = "path/to/custom_template.tplx"
```

### Usar template del TEC (ejemplo)

```python
NBCONVERT_TEMPLATE = "config/pdf_templates/tec_template.tplx"
```

## 🐛 Solución de Problemas

### Error: "pdflatex not found"

**Causa**: LaTeX no está instalado o no está en PATH

**Solución**:
```bash
# macOS
brew install --cask mactex-no-gui
eval "$(/usr/libexec/path_helper)"

# Linux (Ubuntu)
sudo apt-get install texlive-xetex texlive-fonts-recommended
```

### Error: "No such file or directory"

**Causa**: Ruta incorrecta al notebook

**Solución**:
```bash
# Usar ruta absoluta
uv run nb2pdf /full/path/to/notebook.ipynb

# O navegar al directorio
cd notebooks/actividades/actividad_3/
uv run nb2pdf activity3.ipynb
```

### Error: Gráficas no aparecen en PDF

**Causa**: Celdas no ejecutadas o gráficas inline deshabilitadas

**Solución**:
```python
# Agregar al inicio del notebook
%matplotlib inline

# Ejecutar todas las celdas
# Kernel > Restart & Run All
```

### Error: Caracteres especiales mal renderizados

**Causa**: Problemas de codificación

**Solución**:
- Verificar que el notebook use UTF-8
- Usar caracteres ASCII para nombres de archivos
- Evitar emojis en el contenido del notebook

### PDF muy grande

**Causa**: Gráficas de alta resolución

**Solución**:
```python
# Reducir DPI en matplotlib
plt.figure(figsize=(10, 6), dpi=72)  # En lugar de 300
plt.savefig('grafica.png', dpi=72)
```

## 📊 Ejemplos Completos

### Ejemplo 1: Actividad individual

```bash
# Ir al directorio
cd mem-estadistics

# Convertir
uv run nb2pdf notebooks/actividades/actividad_3/activity3.ipynb

# Resultado
# ✅ PDF generado: notebooks/actividades/actividad_3/outputs/activity3.pdf
```

### Ejemplo 2: Todas las actividades

```bash
cd notebooks/actividades/

# Convertir actividad 2
uv run nb2pdf actividad_2/activity2.ipynb

# Convertir actividad 3
uv run nb2pdf actividad_3/activity3.ipynb

# O todos a la vez (si están en el mismo directorio)
cd actividad_3/
uv run nb2pdf --all
```

### Ejemplo 3: Conversión con nombre personalizado

```bash
DATE=$(date +%Y%m%d)
uv run nb2pdf activity3.ipynb -o "Actividad3_William_Monroy_${DATE}.pdf"
```

## 🚀 Tips y Mejores Prácticas

1. **Nombrar archivos apropiadamente**: Usa nombres descriptivos sin espacios
2. **Documentar bien el código**: Los comentarios aparecerán en el PDF
3. **Usar markdown cells**: Para explicaciones formateadas
4. **Optimizar gráficas**: Balance entre calidad y tamaño de archivo
5. **Probar conversión frecuentemente**: No esperar al final del proyecto

## 📚 Recursos Adicionales

- [Documentación de nbconvert](https://nbconvert.readthedocs.io/)
- [Guía de instalación LaTeX](./instalacion.md#latex-requerido-para-conversión-a-pdf)
- [Estructura del proyecto](./estructura_proyecto.md)

## 💡 Atajos Útiles

```bash
# Alias útiles (agregar a ~/.zshrc o ~/.bashrc)
alias convert='uv run nb2pdf'
alias convert-all='uv run nb2pdf --all'

# Función para convertir y abrir
function convertopen() {
    uv run nb2pdf "$1" && open "${1%.ipynb}.pdf"
}
```
