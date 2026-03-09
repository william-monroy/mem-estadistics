"""
Funciones auxiliares para el convertidor de notebooks.
"""

import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple
from .config import Config


def format_file_size(size_bytes: int) -> str:
    """
    Formatea el tamaño de archivo en formato legible.
    
    Args:
        size_bytes: Tamaño en bytes
        
    Returns:
        String formateado (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def get_relative_path(path: Path, base: Path = None) -> Path:
    """
    Obtiene la ruta relativa de un archivo respecto a una base.
    
    Args:
        path: Ruta del archivo
        base: Ruta base (por defecto: directorio actual)
        
    Returns:
        Ruta relativa
    """
    if base is None:
        base = Path.cwd()
    
    try:
        return path.relative_to(base)
    except ValueError:
        # Si no se puede calcular relativa, devolver path completo
        return path


def validate_notebook_path(path: Path) -> bool:
    """
    Valida que la ruta sea un notebook válido.
    
    Args:
        path: Ruta a validar
        
    Returns:
        True si es válida
    """
    if not path.exists():
        return False
    
    if not path.is_file():
        return False
    
    if path.suffix != Config.NOTEBOOK_EXTENSION:
        return False
    
    if Config.should_exclude_path(path):
        return False
    
    return True


def find_notebooks_recursive(directory: Path) -> List[Path]:
    """
    Busca recursivamente todos los notebooks en un directorio.
    
    Args:
        directory: Directorio donde buscar
        
    Returns:
        Lista de rutas a notebooks
    """
    if not directory.exists() or not directory.is_dir():
        return []
    
    notebooks = []
    
    for item in directory.rglob(f"*{Config.NOTEBOOK_EXTENSION}"):
        if validate_notebook_path(item):
            notebooks.append(item)
    
    return sorted(notebooks)


def print_separator(char: str = "=", length: int = 80):
    """Imprime una línea separadora."""
    print(char * length)


def print_header(title: str, emoji: str = ""):
    """Imprime un encabezado formateado."""
    print_separator()
    if emoji:
        print(f"{emoji} {title}")
    else:
        print(title)
    print_separator()


def check_command_available(command: str) -> Tuple[bool, str]:
    """
    Verifica si un comando está disponible en el sistema.
    
    Args:
        command: Nombre del comando a verificar
        
    Returns:
        Tupla (disponible, version/mensaje)
    """
    # Verificar si el comando existe
    if not shutil.which(command):
        return False, "No encontrado"
    
    # Intentar obtener la versión
    try:
        result = subprocess.run(
            [command, '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        version_output = result.stdout.strip() or result.stderr.strip()
        # Tomar solo la primera línea
        version = version_output.split('\n')[0][:50]
        return True, version
    except:
        return True, "Disponible"


def diagnose_system() -> bool:
    """
    Diagnostica el sistema para verificar que todos los requisitos estén instalados.
    
    Returns:
        True si todos los requisitos están satisfechos
    """
    print_header("Diagnóstico del Sistema", "🔍")
    
    all_ok = True
    
    # Verificar Python
    print("\n📌 Verificando Python...")
    python_available, python_version = check_command_available("python3")
    if python_available:
        print(f"   ✅ Python: {python_version}")
    else:
        print(f"   ❌ Python no encontrado")
        all_ok = False
    
    # Verificar Jupyter
    print("\n📌 Verificando Jupyter...")
    jupyter_available, jupyter_version = check_command_available("jupyter")
    if jupyter_available:
        print(f"   ✅ Jupyter: {jupyter_version}")
        
        # Verificar nbconvert específicamente
        try:
            result = subprocess.run(
                ['jupyter', 'nbconvert', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            nbconvert_version = result.stdout.strip()
            print(f"   ✅ nbconvert: {nbconvert_version}")
        except:
            print(f"   ⚠️  nbconvert no disponible")
            all_ok = False
    else:
        print(f"   ❌ Jupyter no encontrado")
        print(f"      Solución: uv sync")
        all_ok = False
    
    # Verificar Pandoc
    print("\n📌 Verificando Pandoc...")
    pandoc_available, pandoc_version = check_command_available("pandoc")
    if pandoc_available:
        print(f"   ✅ Pandoc: {pandoc_version}")
    else:
        print(f"   ❌ Pandoc no encontrado")
        print(f"      Soluciones:")
        print(f"      - macOS: brew install pandoc")
        print(f"      - Linux: sudo apt-get install pandoc")
        print(f"      - Windows: https://pandoc.org/installing.html")
        all_ok = False
    
    # Verificar LaTeX
    print("\n📌 Verificando LaTeX...")
    latex_commands = ['xelatex', 'pdflatex']
    latex_found = False
    
    for cmd in latex_commands:
        available, version = check_command_available(cmd)
        if available:
            print(f"   ✅ {cmd}: {version}")
            latex_found = True
            break
    
    if not latex_found:
        print(f"   ❌ LaTeX no encontrado (xelatex/pdflatex)")
        print(f"      Soluciones:")
        print(f"      - macOS: brew install --cask mactex-no-gui")
        print(f"      - Linux: sudo apt-get install texlive-xetex")
        print(f"      - Windows: https://miktex.org/download")
        all_ok = False
    
    # Resumen
    print()
    print_separator()
    if all_ok:
        print(f"✅ Todos los requisitos están instalados correctamente")
    else:
        print(f"❌ Algunos requisitos faltan. Instálalos antes de continuar.")
    print_separator()
    
    return all_ok
