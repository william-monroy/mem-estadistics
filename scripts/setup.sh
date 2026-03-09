#!/bin/bash
# Script de instalación y configuración del proyecto

set -e  # Salir si hay algún error

echo "🚀 Configuración de mem-estadistics"
echo "===================================="
echo ""

# Verificar si uv está instalado
if ! command -v uv &> /dev/null; then
    echo "❌ uv no está instalado"
    echo "   Instálalo con: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "✅ uv está instalado"

# Sincronizar dependencias
echo ""
echo "📦 Instalando dependencias..."
uv sync
echo "✅ Dependencias instaladas"

# Verificar si LaTeX está instalado
echo ""
if command -v pdflatex &> /dev/null; then
    echo "✅ LaTeX está instalado"
else
    echo "⚠️  LaTeX no está instalado"
    echo ""
    echo "   LaTeX es necesario para convertir notebooks a PDF."
    echo "   Instálalo con:"
    echo ""
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "   brew install --cask mactex-no-gui"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "   sudo apt-get install texlive-xetex texlive-fonts-recommended texlive-plain-generic"
    fi
    echo ""
fi

# Hacer el script principal ejecutable
chmod +x main.py

echo ""
echo "✅ Configuración completada"
echo ""
echo "📝 Uso:"
echo "   uv run python main.py activity3.ipynb"
echo "   uv run python main.py --all"
echo ""
