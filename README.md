# Proyecto de Análisis de Datos - Formar Innovar

Este proyecto está dedicado al análisis de datos de los colegios **Formar Innovar** en Fusagasugá y Girardot. Su objetivo es gestionar y analizar diferentes tipos de datos, incluidos los datos de calificaciones, otros datos relevantes, y la integración con Moodle.

## Descripción General del Proyecto

El proyecto abarca varias áreas clave:

- **Procesamiento de Calificaciones**: Automatización del procesamiento de archivos PDF de calificaciones, que son convertidos a formato CSV para facilitar el análisis.
- **Análisis de Otros Datos**: Herramientas y scripts para manejar y analizar otros datos relevantes del entorno educativo.
- **Integración con Moodle**: Scripts para interactuar con Moodle, extrayendo y manipulando datos para mejorar la gestión y el análisis.

### Componentes Principales

- `procesar_calificaciones_pdf.py`: Este script procesa los archivos PDF de calificaciones almacenados en una carpeta específica y genera archivos CSV organizados.
- Otros scripts (por ejemplo, para análisis de datos adicionales y manejo de Moodle).
- `requirements.txt`: Lista de todas las dependencias necesarias para ejecutar los scripts del proyecto.
- `virtualenv.sh`: Un script de bash que configura el entorno virtual de Python, instala las dependencias necesarias y prepara el entorno de desarrollo.

## Requisitos Previos

Antes de comenzar, se debe asegurar de tener instalado **Conda** para gestionar los entornos virtuales de Python.

## Configuración del Entorno

Para configurar el entorno de desarrollo e instalar las dependencias necesarias, se deben seguir estos pasos:

1. Ejecutar el script `virtualenv.sh`:

   ```bash
   bash virtualenv.sh
   ```

Este script realizará las siguientes acciones:

- Creará un entorno virtual llamado `formar-innovar`.
- Activará el entorno virtual.
- Instalará las dependencias listadas en `requirements.txt`.

2. Una vez configurado el entorno, se podrán ejecutar los scripts del proyecto.

## Procesamiento de Calificaciones

Para procesar los archivos PDF de calificaciones:

1. Colocar los archivos PDF en la carpeta `data/calificaciones/`.
2. Ejecutar el script `procesar_calificaciones_pdf.py` para leer los archivos PDF y convertirlos a CSV:

   ```bash
   python scripts/procesar_calificaciones_pdf.py
   ```

Esto generará un archivo CSV en la misma carpeta con los datos procesados, listo para su análisis.