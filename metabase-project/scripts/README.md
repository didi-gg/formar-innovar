# Documentación Scripts

## Tabla de Contenido

- [Mapeo de Secciones y Módulos en Moodle](#mapeo-de-secciones-y-módulos-en-moodle)

## Mapeo de Secciones y Módulos en Moodle
[create_parquet_rel_course_activity](create_parquet_rel_course_activity.py)


### Descripción
Este script procesa los datos de secciones de cursos en Moodle y extrae los **IDs de módulos** (module_id) almacenados en la columna sequence. Luego, transforma esta información para generar un mapeo detallado de la relación **sección-módulo**, estructurándolo en un nuevo archivo Parquet.

### Funcionamiento
1. **Carga de Datos**:
   * Se lee la tabla `mdlvf_course_sections` desde un archivo Parquet, la cual contiene información de las secciones de cursos y sus módulos asociados en la columna sequence.
2. **Transformación de Datos**:
   * Se eliminan filas donde sequence es NULL (es decir, aquellas secciones sin módulos asignados).
   * Se convierte la columna sequence en una **lista de módulos**, separando los valores por comas.
   * Se "explota" la lista, generando **una fila por cada módulo asociado a una sección**.
   * Se asegura que module_id sea numérico y se eliminan valores NULL.
3. **Generación de la Tabla de Relación**:
   * Se seleccionan las columnas relevantes:
     * `course_id` (ID del curso)
     * `section_id` (ID de la sección)
     * `section_name` (Nombre de la sección)
     * `module_id` (ID del módulo asociado)
   * Se guarda el resultado en un archivo Parquet (`activities_section_mapping.parquet`).

### Estructura de las Tablas

1. **Tabla Original (mdlvf_course_sections.parquet)**
Esta tabla almacena información sobre las secciones de los cursos y los módulos asignados.

| **Columna** | **Descripción** | **Ejemplo** |
|:-:|:-:|:-:|
| id (section_id) | ID único de la sección en Moodle. | 10 |
| course (course_id) | ID del curso al que pertenece la sección. | 101 |
| name (section_name) | Nombre de la sección del curso. | "Introducción" |
| sequence | Lista de módulos en la sección (IDs separados por comas). | "201,202,203" |
2. **Tabla Transformada (activities_section_mapping.parquet)**
Después de la transformación, la información queda estructurada con una fila por cada **relación sección-módulo**.

| **Columna** | **Antes (Original)** | **Después (Transformado)** |
|:-:|:-:|:-:|
| id / section_id | ID de la sección | Se mantiene igual |
| course / course_id | ID del curso | Se mantiene igual |
| name / section_name | Nombre de la sección | Se mantiene igual |
| sequence | Lista de IDs de módulos en una cadena de texto | Se convierte en filas separadas con module_id |

### Representación Gráfica de la Jerarquía

📚 Curso (course_id) → 101
   ├── 📂 Sección (section_id) → 10
   │    ├── 🧩 Módulo (module_id) → 201
   │    ├── 🧩 Módulo (module_id) → 202
   │    ├── 🧩 Módulo (module_id) → 203
   ├── 📂 Sección (section_id) → 11
   │    ├── 🧩 Módulo (module_id) → 204
   │    ├── 🧩 Módulo (module_id) → 205