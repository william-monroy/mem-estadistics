"""
Tests para utilidades.
"""

import pytest
from pathlib import Path
from nb2pdf.utils import (
    format_file_size,
    validate_notebook_path,
    get_relative_path,
)


def test_format_file_size():
    """Verifica el formateo de tamaños de archivo."""
    assert format_file_size(0) == "0.0 B"
    assert format_file_size(1024) == "1.0 KB"
    assert format_file_size(1024 * 1024) == "1.0 MB"
    assert format_file_size(1536) == "1.5 KB"
    assert format_file_size(int(1.5 * 1024 * 1024)) == "1.5 MB"


def test_validate_notebook_path_valid(tmp_path):
    """Verifica validación de notebooks válidos."""
    notebook = tmp_path / "test.ipynb"
    notebook.write_text("{}")
    
    assert validate_notebook_path(notebook)


def test_validate_notebook_path_invalid():
    """Verifica rechazo de rutas inválidas."""
    # Archivo no existe
    assert not validate_notebook_path(Path("nonexistent.ipynb"))
    
    # Extensión incorrecta
    temp_file = Path(__file__).parent / "test_utils.py"
    assert not validate_notebook_path(temp_file)


def test_validate_notebook_path_excluded():
    """Verifica que rutas excluidas sean rechazadas."""
    excluded = Path(".ipynb_checkpoints/test.ipynb")
    assert not validate_notebook_path(excluded)


def test_get_relative_path():
    """Verifica el cálculo de rutas relativas."""
    base = Path("/home/user/project")
    path = Path("/home/user/project/notebooks/activity.ipynb")
    
    relative = get_relative_path(path, base)
    assert relative == Path("notebooks/activity.ipynb")


def test_get_relative_path_no_base():
    """Verifica rutas relativas sin base especificada."""
    path = Path.cwd() / "test.ipynb"
    relative = get_relative_path(path)
    
    assert relative == Path("test.ipynb")
