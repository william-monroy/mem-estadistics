# Guía de Instalación - mem-estadistics

## 📋 Requisitos Previos

### Python
- Python 3.13 o superior
- Se recomienda usar `uv` como gestor de paquetes

### LaTeX (requerido para conversión a PDF)

#### macOS
```bash
# Opción 1: Instalación ligera (recomendada)
brew install --cask mactex-no-gui

# Opción 2: Instalación completa (~4GB)
brew install --cask mactex

# Después de instalar, actualizar PATH
eval "$(/usr/libexec/path_helper)"
```

#### Linux

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install texlive-xetex texlive-fonts-recommended texlive-plain-generic pandoc
```

**Fedora:**
```bash
sudo dnf install texlive-scheme-basic texlive-xetex pandoc
```

#### Windows
1. Descargar e instalar [MiKTeX](https://miktex.org/download)
2. Durante la instalación, seleccionar "Install missing packages automatically"

### Git
```bash
# macOS (si no está instalado)
brew install git

# Linux
sudo apt-get install git  # Ubuntu/Debian
sudo dnf install git       # Fedora
```

## 🚀 Instalación del Proyecto

### 1. Instalar uv (si no lo tienes)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# O usando Homebrew
brew install uv

# Verificar instalación
uv --version
```

### 2. Clonar el repositorio

```bash
git clone https://github.com/william-monroy/mem-estadistics.git
cd mem-estadistics
```

### 3. Instalar dependencias

```bash
# uv creará automáticamente el entorno virtual e instalará dependencias
uv sync

# Para desarrollo (incluye pytest, ruff, etc.)
uv sync --extra dev
```

### 4. Verificar instalación

```bash
# Probar el convertidor
uv run nb2pdf --help

# Ejecutar tests
uv run pytest

# O ejecutar el script de setup
./scripts/setup.sh
```

## ✅ Verificación de Componentes

### Verificar Python
```bash
python3 --version  # Debe ser 3.13+
```

### Verificar LaTeX
```bash
pdflatex --version
```

### Verificar uv
```bash
uv --version
```

### Verificar Jupyter
```bash
uv run jupyter --version
```

## 🔧 Solución de Problemas

### Error: "uv: command not found"
```bash
# Actualizar PATH
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Error: "pdflatex: command not found"
- Instalar LaTeX según las instrucciones de tu sistema operativo
- En macOS, después de instalar: `eval "$(/usr/libexec/path_helper)"`
- Reiniciar la terminal

### Error al instalar dependencias
```bash
# Limpiar caché y reinstalar
rm -rf .venv
uv cache clean
uv sync
```

### Permisos en scripts
```bash
chmod +x scripts/*.sh
```

## 📚 Próximos Pasos

1. Lee la [Guía de Uso](./conversion_pdf.md) para aprender a convertir notebooks
2. Explora la [Estructura del Proyecto](./estructura_proyecto.md)
3. Consulta el [README principal](../../README.md) para más información

## 💡 Consejos

- Usa `uv run` para ejecutar comandos sin activar el entorno virtual
- Los PDFs se generan automáticamente en carpetas `outputs/`
- Mantén tus notebooks ejecutados antes de convertir a PDF
- Usa `uv add <paquete>` para agregar nuevas dependencias

## 📞 Soporte

Si encuentras problemas:
1. Revisa la sección de [Solución de Problemas](#-solución-de-problemas)
2. Consulta los [Issues en GitHub](https://github.com/william-monroy/mem-estadistics/issues)
3. Verifica que todos los requisitos previos estén instalados
