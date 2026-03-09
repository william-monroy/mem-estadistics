"""
Tests para el módulo de configuración.
"""

import pytest
from pathlib import Path
from nb2pdf.config import Config


def test_config_constants():
    """Verifica que las constantes de configuración estén definidas."""
    assert Config.PROJECT_ROOT.exists()
    assert Config.NOTEBOOKS_DIR.name == "notebooks"
    assert Config.NOTEBOOK_EXTENSION == ".ipynb"
    assert Config.NBCONVERT_FORMAT == "pdf"


def test_should_exclude_path():
    """Verifica la lógica de exclusión de rutas."""
    # Rutas que deberían ser excluidas
    assert Config.should_exclude_path(Path(".venv/some/file.py"))
    assert Config.should_exclude_path(Path("notebooks/.ipynb_checkpoints/file.ipynb"))
    assert Config.should_exclude_path(Path(".git/config"))
    assert Config.should_exclude_path(Path("__pycache__/module.pyc"))
    
    # Rutas que NO deberían ser excluidas
    assert not Config.should_exclude_path(Path("notebooks/activity.ipynb"))
    assert not Config.should_exclude_path(Path("src/module.py"))


def test_get_output_dir_actividades():
    """Verifica la generación de directorio de salida para actividades."""
    notebook_path = Path("notebooks/actividades/actividad_3/activity3.ipynb")
    output_dir = Config.get_output_dir(notebook_path)
    
    assert output_dir.name == "outputs"
    assert "actividad_3" in str(output_dir)


def test_get_output_dir_otros():
    """Verifica que notebooks fuera de actividades usen su propio directorio."""
    notebook_path = Path("notebooks/ejemplos/Linear_Models.ipynb")
    output_dir = Config.get_output_dir(notebook_path)
    
    assert output_dir == notebook_path.parent
