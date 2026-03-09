"""
Configuración del convertidor de notebooks.
"""

from pathlib import Path
from typing import Optional


class Config:
    """Configuración global del convertidor."""
    
    # Directorios predeterminados
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
    OUTPUTS_DIR_NAME = "outputs"
    
    # Configuración de nbconvert
    NBCONVERT_FORMAT = "pdf"
    NBCONVERT_TEMPLATE = None  # None = usar template por defecto
    
    # Configuración de búsqueda
    EXCLUDED_DIRS = {".ipynb_checkpoints", ".venv", ".git", "__pycache__"}
    NOTEBOOK_EXTENSION = ".ipynb"
    
    # Mensajes de UI
    EMOJI_SUCCESS = "✅"
    EMOJI_ERROR = "❌"
    EMOJI_PROCESSING = "🔄"
    EMOJI_SUMMARY = "📊"
    
    @classmethod
    def get_output_dir(cls, notebook_path: Path) -> Path:
        """
        Obtiene el directorio de salida para un notebook.
        
        Args:
            notebook_path: Ruta al notebook
            
        Returns:
            Ruta al directorio de salida
        """
        # Si el notebook está en la estructura de actividades, usar carpeta outputs
        if "actividades" in notebook_path.parts:
            output_dir = notebook_path.parent / cls.OUTPUTS_DIR_NAME
            output_dir.mkdir(exist_ok=True)
            return output_dir
        # De lo contrario, mismo directorio que el notebook
        return notebook_path.parent
    
    @classmethod
    def should_exclude_path(cls, path: Path) -> bool:
        """
        Determina si una ruta debe ser excluida de la búsqueda.
        
        Args:
            path: Ruta a verificar
            
        Returns:
            True si debe ser excluida
        """
        return any(excluded in path.parts for excluded in cls.EXCLUDED_DIRS)
