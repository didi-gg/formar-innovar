# 🐳 Docker Setup Simple

Contenedor Docker para desarrollo con `docker run` directo.

## 🚀 Inicio Rápido

### 1. Construir imagen
```bash
docker build -t formar-innovar .
```

### 2. Ejecutar contenedor
```bash
# Ejecutar contenedor interactivo
docker run -it -p 5000:5000 -p 8888:8888 -v $(pwd):/app formar-innovar bash
```

### 3. Dentro del contenedor
```bash
# Ir a la carpeta de modelos
cd /app/scripts/modelling

# Iniciar MLflow UI (en background)
mlflow ui --host 0.0.0.0 --port 5000 &

# O iniciar MLflow UI (foreground)
mlflow ui --host 0.0.0.0 --port 5000
```

### 4. Acceder desde tu navegador
- **MLflow UI**: http://localhost:5000

## 📋 Comandos Útiles

### 🐳 Gestión del contenedor
```bash
# Ejecutar contenedor interactivo
docker run -it -p 5000:5000 -p 8888:8888 -v $(pwd):/app formar-innovar bash

# Ejecutar contenedor en background
docker run -d -p 5000:5000 -p 8888:8888 -v $(pwd):/app --name formar-innovar-dev formar-innovar

# Entrar a contenedor en background
docker exec -it formar-innovar-dev bash

# Ver contenedores activos
docker ps

# Parar contenedor
docker stop formar-innovar-dev

# Eliminar contenedor
docker rm formar-innovar-dev

# Reconstruir imagen
docker build --no-cache -t formar-innovar .
```

### 🔬 Dentro del contenedor
```bash
# Ejecutar experimentos
cd /app/scripts/modelling
python simple_demo.py

# Ver resultados
mlflow ui --host 0.0.0.0 --port 5000

# Ejecutar Jupyter (opcional)
cd /app
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=''

# Explorar archivos MLflow (ahora en raíz)
ls -la /app/mlruns/
ls -la /app/model_results/
```

## 🌐 Puertos Expuestos

| Puerto | Servicio | Comando |
|--------|----------|---------|
| 5000 | MLflow UI | `mlflow ui --host 0.0.0.0 --port 5000` |
| 8888 | Jupyter Lab | `jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root` |

## 💡 Flujo de Trabajo Típico

```bash
# 1. Construir imagen
docker build -t formar-innovar .

# 2. Ejecutar contenedor
docker run -it -p 5000:5000 -v $(pwd):/app formar-innovar bash

# 3. Dentro del contenedor - ejecutar experimentos
Ejecutar entrenamiento y evaluación de modelos

# 4. Ver resultados
mlflow ui --host 0.0.0.0 --port 5000

# 5. Acceder desde navegador
# http://localhost:5000
```
