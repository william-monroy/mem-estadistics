# Guía Rápida - Convertidor de Notebooks

## 🎯 Comandos Más Comunes

### Convertir un notebook específico
```bash
uv run python main.py activity3.ipynb
```

### Convertir con nombre personalizado
```bash
uv run python main.py activity3.ipynb -o Actividad_3_William_Monroy.pdf
```

### Convertir todos los notebooks
```bash
uv run python main.py --all
```

## 🔧 Configuración Inicial

### Primera vez (instalar todo)
```bash
./setup.sh
```

### Solo instalar dependencias
```bash
uv sync
```

### Agregar una nueva dependencia
```bash
uv add nombre-paquete
```

## 📦 Gestión del Entorno Virtual

### Activar el entorno virtual manualmente
```bash
source .venv/bin/activate
```

### Ejecutar sin activar (recomendado)
```bash
uv run python tu_script.py
```

### Ver dependencias instaladas
```bash
uv pip list
```

## 🔍 Solución Rápida de Problemas

### LaTeX no instalado (macOS)
```bash
brew install --cask mactex-no-gui
eval "$(/usr/libexec/path_helper)"
```

### Reinstalar dependencias
```bash
rm -rf .venv
uv sync
```

### Ver errores detallados
```bash
uv run python main.py activity3.ipynb --verbose 2>&1 | tee conversion.log
```

## 📊 Estructura de Archivos

```
.
├── main.py              → Script principal de conversión
├── setup.sh             → Script de configuración automática
├── pyproject.toml       → Configuración del proyecto
├── .venv/               → Entorno virtual (auto-generado)
├── *.ipynb              → Notebooks de entrada
└── *.pdf                → PDFs generados (git-ignored)
```

## 💡 Tips

1. **Los PDFs se generan en el mismo directorio del notebook**
2. **Los archivos `.ipynb_checkpoints/` se ignoran automáticamente**
3. **Puedes ejecutar el convertidor desde cualquier directorio:**
   ```bash
   uv run python main.py ~/ruta/a/mi/notebook.ipynb
   ```

## 🚀 Atajos Útiles

### Convertir y abrir el PDF (macOS)
```bash
uv run python main.py activity3.ipynb && open activity3.pdf
```

### Convertir todos y contar éxitos
```bash
uv run python main.py --all | grep "exitosos"
```

### Ver solo errores
```bash
uv run python main.py --all 2>&1 | grep "❌"
```
