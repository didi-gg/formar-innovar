# Crea el entorno virtual solo si no existe ya
if conda info --envs | grep -q "formar-innovar"; then
    echo "El entorno 'formar-innovar' ya existe."
else
    echo "Creando el entorno 'formar-innovar'..."
    conda create --name formar-innovar python=3.10 -y
fi

# Activa el entorno
echo "Activando el entorno 'formar-innovar'..."
conda activate formar-innovar

# Actualiza pip y setuptools
echo "Actualizando pip y setuptools..."
pip install --upgrade pip setuptools

# Instala las dependencias
if [ -f "requirements.txt" ]; then
    echo "Instalando las dependencias desde requirements.txt..."
    pip install -r requirements.txt
else
    echo "El archivo 'requirements.txt' no se encuentra."
    exit 1
fi

# Verifica la instalación de las dependencias
echo "Verificando la instalación de las dependencias..."
pip check

# Crea un archivo .env si es necesario
if [ ! -f ".env" ]; then
    echo "Creando un archivo .env de ejemplo..."
    echo "MOODLE_URL=moodle-url" > .env
    echo "MOODLE_API_KEY=your-secret-key" >> .env
    echo "Variables de entorno de ejemplo escritas en .env"
fi