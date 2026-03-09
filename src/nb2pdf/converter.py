"""
Lógica principal de conversión de notebooks a PDF.
"""

import subprocess
from pathlib import Path
from typing import List, Optional

from .config import Config
from .utils import (
    validate_notebook_path,
    find_notebooks_recursive,
    get_relative_path,
    print_header,
)


class ConversionError(Exception):
    """Excepción personalizada para errores de conversión."""
    pass


def convert_notebook(
    notebook_path: Path,
    output_path: Optional[Path] = None,
    verbose: bool = False
) -> bool:
    """
    Convierte un notebook de Jupyter a PDF usando nbconvert.
    
    Args:
        notebook_path: Ruta al archivo .ipynb
        output_path: Ruta de salida para el PDF (opcional)
        verbose: Mostrar salida detallada
    
    Returns:
        True si la conversión fue exitosa, False en caso contrario
        
    Raises:
        ConversionError: Si hay un error durante la conversión
    """
    # Validar entrada
    if not validate_notebook_path(notebook_path):
        raise ConversionError(
            f"Ruta inválida o archivo no es un notebook válido: {notebook_path}"
        )
    
    # Determinar ruta de salida
    if output_path is None:
        output_dir = Config.get_output_dir(notebook_path)
        output_path = output_dir / notebook_path.with_suffix('.pdf').name
    else:
        # Asegurar que tenga extensión .pdf
        if output_path.suffix != '.pdf':
            output_path = output_path.with_suffix('.pdf')
        # Crear directorio si no existe
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Construir comando de nbconvert
        cmd = [
            'jupyter', 'nbconvert',
            '--to', Config.NBCONVERT_FORMAT,
            '--output', str(output_path.absolute()),
            str(notebook_path.absolute())
        ]
        
        # Agregar template si está configurado
        if Config.NBCONVERT_TEMPLATE:
            cmd.extend(['--template', Config.NBCONVERT_TEMPLATE])
        
        relative_path = get_relative_path(notebook_path)
        print(f"{Config.EMOJI_PROCESSING} Convirtiendo '{relative_path}'...")
        
        # Ejecutar conversión
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        if verbose and result.stdout:
            print(result.stdout)
        
        relative_output = get_relative_path(output_path)
        print(f"{Config.EMOJI_SUCCESS} PDF generado: {relative_output}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        error_msg = f"Error al convertir '{notebook_path.name}'"
        stderr = e.stderr.strip() if e.stderr else ""
        
        # Detectar tipos específicos de errores
        if "pandoc" in stderr.lower() and ("wasn't found" in stderr.lower() or "not found" in stderr.lower()):
            print(f"{Config.EMOJI_ERROR} {error_msg}")
            print(f"\n💡 Causa probable: Pandoc no está instalado o no está en el PATH")
            print(f"   Solución: Instala Pandoc:")
            print(f"   - macOS: brew install pandoc")
            print(f"   - Linux: sudo apt-get install pandoc")
            print(f"   - Windows: https://pandoc.org/installing.html")
        elif "xelatex" in stderr.lower() or "pdflatex" in stderr.lower():
            print(f"{Config.EMOJI_ERROR} {error_msg}")
            print(f"\n💡 Causa probable: LaTeX no está instalado o no está en el PATH")
            print(f"   Solución: Instala LaTeX o verifica tu instalación:")
            print(f"   - macOS: brew install --cask mactex-no-gui")
            print(f"   - Linux: sudo apt-get install texlive-xetex texlive-fonts-recommended")
            print(f"   - Windows: https://miktex.org/download")
        elif "jupyter: command not found" in stderr.lower() or "No such file or directory" in stderr.lower():
            print(f"{Config.EMOJI_ERROR} {error_msg}")
            print(f"\n💡 Causa probable: Jupyter no está instalado correctamente")
            print(f"   Solución: uv sync para reinstalar dependencias")
        elif "nbformat" in stderr.lower() or "invalid notebook" in stderr.lower():
            print(f"{Config.EMOJI_ERROR} {error_msg}")
            print(f"\n💡 Causa probable: El archivo notebook está corrupto o tiene formato inválido")
            print(f"   Solución: Abre el notebook en Jupyter y verifica que se cargue correctamente")
        elif "permission denied" in stderr.lower():
            print(f"{Config.EMOJI_ERROR} {error_msg}")
            print(f"\n💡 Causa probable: Sin permisos de escritura en '{output_path.parent}'")
            print(f"   Solución: Verifica los permisos del directorio de salida")
        else:
            # Error genérico con detalles
            print(f"{Config.EMOJI_ERROR} {error_msg}")
            if stderr:
                # Mostrar las últimas líneas relevantes del error
                error_lines = stderr.split('\n')
                relevant_lines = [line for line in error_lines[-10:] if line.strip()]
                if relevant_lines:
                    print(f"\n💡 Detalles del error:")
                    for line in relevant_lines[-3:]:  # Últimas 3 líneas relevantes
                        print(f"   {line}")
        
        if verbose and stderr:
            print(f"\n🔍 Error completo:\n{stderr}")
        
        if verbose:
            raise ConversionError(error_msg) from e
        
        return False
        
    except FileNotFoundError as e:
        error_msg = f"Archivo no encontrado: {e.filename if hasattr(e, 'filename') else notebook_path.name}"
        print(f"{Config.EMOJI_ERROR} {error_msg}")
        print(f"\n💡 Verifica que la ruta del archivo sea correcta")
        
        if verbose:
            raise ConversionError(error_msg) from e
        
        return False
        
    except PermissionError as e:
        error_msg = f"Sin permisos para acceder a '{notebook_path.name}'"
        print(f"{Config.EMOJI_ERROR} {error_msg}")
        print(f"\n💡 Verifica los permisos del archivo o directorio")
        
        if verbose:
            raise ConversionError(error_msg) from e
        
        return False
        
    except Exception as e:
        error_type = type(e).__name__
        error_msg = f"Error inesperado al convertir '{notebook_path.name}': {error_type}"
        print(f"{Config.EMOJI_ERROR} {error_msg}")
        print(f"   Mensaje: {str(e)}")
        
        if verbose:
            print(f"\n🔍 Traceback completo:")
            raise ConversionError(error_msg) from e
        
        return False


def find_all_notebooks(directory: Optional[Path] = None) -> List[Path]:
    """
    Encuentra todos los archivos .ipynb en el directorio especificado.
    
    Args:
        directory: Directorio donde buscar (por defecto: directorio actual)
    
    Returns:
        Lista de rutas a archivos .ipynb ordenadas alfabéticamente
    """
    if directory is None:
        directory = Path.cwd()
    
    if not directory.exists() or not directory.is_dir():
        return []
    
    # Buscar solo en el directorio actual (no recursivo para evitar outputs/)
    notebooks = []
    for item in directory.glob(f"*{Config.NOTEBOOK_EXTENSION}"):
        if validate_notebook_path(item):
            notebooks.append(item)
    
    return sorted(notebooks)


def convert_multiple_notebooks(
    notebooks: List[Path],
    verbose: bool = False
) -> tuple[int, int]:
    """
    Convierte múltiples notebooks a PDF.
    
    Args:
        notebooks: Lista de rutas a notebooks
        verbose: Mostrar salida detallada
        
    Returns:
        Tupla de (exitosos, fallidos)
    """
    if not notebooks:
        print(f"{Config.EMOJI_ERROR} No se encontraron notebooks para convertir")
        return 0, 0
    
    print_header(f"Convertir {len(notebooks)} notebook(s)", "📚")
    
    for i, nb in enumerate(notebooks, 1):
        relative = get_relative_path(nb)
        print(f"\n[{i}/{len(notebooks)}] {relative}")
    
    print()
    
    successful = 0
    failed = 0
    
    for notebook in notebooks:
        try:
            if convert_notebook(notebook, verbose=verbose):
                successful += 1
            else:
                failed += 1
        except ConversionError:
            failed += 1
        print()  # Línea en blanco entre conversiones
    
    # Resumen final
    print_header("Resumen de Conversión", Config.EMOJI_SUMMARY)
    print(f"{Config.EMOJI_SUCCESS} Exitosos: {successful}")
    print(f"{Config.EMOJI_ERROR} Fallidos: {failed}")
    print(f"Total procesados: {successful + failed}/{len(notebooks)}")
    
    return successful, failed
