#!/bin/bash
# Script para limpiar archivos generados

set -e

echo "🧹 Limpiando archivos generados..."
echo "=================================="
echo ""

cd "$(dirname "$0")/.." # Ir a raíz del proyecto

# Función para contar archivos
count_files() {
    find . "$@" 2>/dev/null | wc -l | tr -d ' '
}

# Contar antes de limpiar
pdf_count=$(count_files -path "*/outputs/*.pdf")
aux_count=$(count_files -name "*.aux" -o -name "*.log" -o -name "*.out" -o -name "*.fls" -o -name "*.fdb_latexmk" -o -name "*.synctex.gz")
checkpoint_count=$(count_files -path "*/.ipynb_checkpoints/*")

echo "📊 Archivos encontrados:"
echo "   PDFs en outputs/: $pdf_count"
echo "   Archivos LaTeX temporales: $aux_count"
echo "   Checkpoints de Jupyter: $checkpoint_count"
echo ""

if [ "$1" == "--dry-run" ]; then
    echo "🔍 Modo dry-run - no se eliminará nada"
    exit 0
fi

# Confirmar
if [ "$1" != "-f" ] && [ "$1" != "--force" ]; then
    read -p "¿Continuar con la limpieza? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Limpieza cancelada"
        exit 0
    fi
fi

# Limpiar PDFs generados
echo "🗑️  Eliminando PDFs en outputs/..."
find . -path "*/outputs/*.pdf" -delete 2>/dev/null || true

# Limpiar archivos temporales de LaTeX
echo "🗑️  Eliminando archivos temporales de LaTeX..."
find . \( -name "*.aux" -o -name "*.log" -o -name "*.out" -o \
         -name "*.fls" -o -name "*.fdb_latexmk" -o \
         -name "*.synctex.gz" -o -name "*.bbl" -o -name "*.blg" \) \
         -delete 2>/dev/null || true

# Limpiar checkpoints de Jupyter
echo "🗑️  Eliminando checkpoints de Jupyter..."
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

# Limpiar __pycache__
echo "🗑️  Eliminando __pycache__..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Limpiar pytest cache
echo "🗑️  Eliminando caché de pytest..."
rm -rf .pytest_cache 2>/dev/null || true

# Contar después
pdf_after=$(count_files -path "*/outputs/*.pdf")
aux_after=$(count_files -name "*.aux" -o -name "*.log")

echo ""
echo "✅ Limpieza completada"
echo "   Archivos eliminados: $((pdf_count + aux_count + checkpoint_count - pdf_after - aux_after))"
echo ""
echo "💡 Tip: Usa --dry-run para ver qué se eliminará sin hacer cambios"
echo "💡 Tip: Usa -f o --force para saltar confirmación"
