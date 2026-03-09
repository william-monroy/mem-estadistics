#!/bin/bash
# Script para convertir todos los notebooks de actividades

set -e

echo "🔄 Convirtiendo todos los notebooks de actividades..."
echo "=================================================="
echo ""

cd "$(dirname "$0")/.." # Ir a raíz del proyecto

# Buscar todos los notebooks de actividades
notebooks=$(find notebooks/actividades -name "*.ipynb" -not -path "*/outputs/*" -not -path "*/.ipynb_checkpoints/*" | sort)

if [ -z "$notebooks" ];then
    echo "❌ No se encontraron notebooks en notebooks/actividades/"
    exit 1
fi

# Contar notebooks
count=$(echo "$notebooks" | wc -l | tr -d ' ')
echo "📚 Encontrados $count notebook(s)"
echo ""

# Convertir cada uno
success=0
failed=0
while IFS= read -r notebook; do
    echo "🔄 Convirtiendo: $notebook"
    if uv run nb2pdf "$notebook"; then
        ((success++))
    else
        ((failed++))
    fi
    echo ""
done <<< "$notebooks"

# Resumen
echo "=================================================="
echo "📊 Resumen:"
echo "   ✅ Exitosos: $success"
echo "   ❌ Fallidos: $failed"
echo "   📝 Total: $((success + failed))"
echo "=================================================="

if [ $failed -gt 0 ]; then
    exit 1
fi
