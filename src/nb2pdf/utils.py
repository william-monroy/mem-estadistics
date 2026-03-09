"""
Funciones auxiliares para el convertidor de notebooks.
"""

from pathlib import Path
from typing import List
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
