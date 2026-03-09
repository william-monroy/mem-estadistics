#!/usr/bin/env python3
"""
CLI para convertir archivos Jupyter Notebook (.ipynb) a PDF.
"""

import argparse
import sys
from pathlib import Path

from .converter import convert_notebook, find_all_notebooks, convert_multiple_notebooks
from .config import Config


def main():
    """Punto de entrada principal del CLI."""
    parser = argparse.ArgumentParser(
        prog='nb2pdf',
        description='Convierte archivos Jupyter Notebook (.ipynb) a PDF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s activity3.ipynb
  %(prog)s notebooks/actividades/actividad_3/activity3.ipynb
  %(prog)s activity3.ipynb --output mi_reporte.pdf
  %(prog)s --all
  %(prog)s --all --directory notebooks/actividades/actividad_3/
        """
    )
    
    parser.add_argument(
        'notebook',
        nargs='?',
        type=Path,
        help='Archivo .ipynb a convertir'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Nombre del archivo PDF de salida'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Convertir todos los archivos .ipynb en el directorio especificado'
    )
    
    parser.add_argument(
        '-d', '--directory',
        type=Path,
        default=Path.cwd(),
        help='Directorio donde buscar notebooks (por defecto: directorio actual)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Mostrar salida detallada'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {Config.PROJECT_ROOT / "src/nb2pdf/__init__.py"}'
    )
    
    args = parser.parse_args()
    
    # Validar argumentos
    if not args.notebook and not args.all:
        parser.error('Debes especificar un archivo .ipynb o usar --all')
    
    if args.all and args.notebook:
        parser.error('No puedes usar --all junto con un archivo específico')
    
    if args.output and args.all:
        parser.error('No puedes especificar --output con --all')
    
    # Procesar notebooks
    try:
        if args.all:
            # Convertir todos los notebooks en el directorio
            notebooks = find_all_notebooks(args.directory)
            
            if not notebooks:
                print(f"{Config.EMOJI_ERROR} No se encontraron archivos .ipynb en '{args.directory}'")
                sys.exit(1)
            
            successful, failed = convert_multiple_notebooks(notebooks, verbose=args.verbose)
            sys.exit(0 if failed == 0 else 1)
        
        else:
            # Convertir un solo archivo
            notebook_path = args.notebook.resolve()
            output_path = args.output.resolve() if args.output else None
            
            success = convert_notebook(notebook_path, output_path, verbose=args.verbose)
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        print(f"\n{Config.EMOJI_ERROR} Operación cancelada por el usuario")
        sys.exit(130)
    except Exception as e:
        print(f"{Config.EMOJI_ERROR} Error inesperado: {e}")
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
