#!/bin/bash

SOURCE_DIR="../notebooks"
TARGET_DIR="./notebooks"
FORCE=false

# Verificamos si se pasó el flag --force
if [ "$1" == "--force" ]; then
  FORCE=true
fi

echo "Origen:    $SOURCE_DIR"
echo "Destino:   $TARGET_DIR"
echo "Modo:      $( [ "$FORCE" == true ] && echo "FORZADO (recrear todos los symlinks)" || echo "Normal (solo enlaces nuevos)")"

# Validar existencia del directorio fuente
if [ ! -d "$SOURCE_DIR" ]; then
  echo "ERROR: '$SOURCE_DIR' no existe. Verifica tu estructura de carpetas."
  exit 1
fi

# Crear carpeta destino si no existe
mkdir -p "$TARGET_DIR"

echo "🔗 Creando enlaces simbólicos (symlinks)..."

# Recorremos los notebooks del directorio fuente
find "$SOURCE_DIR" -type f -name "*.ipynb" | while read notebook; do
  relative_path="${notebook#$SOURCE_DIR/}"
  target_path="$TARGET_DIR/$relative_path"
  target_dir=$(dirname "$target_path")

  mkdir -p "$target_dir"

  # Modo --force: borra siempre
  if [ "$FORCE" == true ]; then
    [ -L "$target_path" ] || [ -e "$target_path" ] && rm -f "$target_path"
    ln -s "$(realpath "$notebook")" "$target_path"
  else
    # Solo crea el symlink si no existe
    if [ ! -e "$target_path" ]; then
      ln -s "$(realpath "$notebook")" "$target_path"
    fi
  fi
done

echo "🧹 Limpiando build anterior..."
jupyter-book clean . --all

echo "🏗️  Compilando el Jupyter Book..."
jupyter-book build .

echo ""
echo "¡Listo! Puedes abrir tu libro aquí:"
echo "   file://$(pwd)/_build/html/index.html"

# bash build_book.sh
# bash build_book.sh --force