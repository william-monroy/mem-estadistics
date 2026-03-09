"""
nb2pdf - Convertidor de Jupyter Notebooks a PDF

Un paquete Python para convertir notebooks de Jupyter a documentos PDF
usando nbconvert y LaTeX.
"""

__version__ = "0.2.0"
__author__ = "William Frank Monroy Mamani"
__email__ = "A00829796@tec.mx"

from .converter import convert_notebook, find_all_notebooks
from .config import Config
from .utils import diagnose_system

__all__ = [
    "convert_notebook",
    "find_all_notebooks",
    "diagnose_system",
    "Config",
]
