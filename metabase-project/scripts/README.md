# Documentación Scripts

## Tabla de Contenido

- [Mapeo de Secciones y Módulos en Moodle](#mapeo-de-secciones-y-módulos-en-moodle)
- [Relación entre Estudiantes y Cursos en Moodle](#relación-entre-estudiantes-y-cursos-en-moodle)
- [Relación entre Estudiantes y Actividades en Moodle](#relación-entre-estudiantes-y-actividades-en-moodle)

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

1. **Tabla Original (`mdlvf_course_sections.parquet`)**
Esta tabla almacena información sobre las secciones de los cursos y los módulos asignados.

| **Columna** | **Descripción** | **Ejemplo** |
|:-:|:-:|:-:|
| id (section_id) | ID único de la sección en Moodle. | 10 |
| course (course_id) | ID del curso al que pertenece la sección. | 101 |
| name (section_name) | Nombre de la sección del curso. | "Introducción" |
| sequence | Lista de módulos en la sección (IDs separados por comas). | "201,202,203" |

### Estructura de la Tabla de Salida (`activities_section_mapping.parquet`)
Después de la transformación, la información queda estructurada con una fila por cada **relación sección-módulo**.

| **Columna** | **Antes (Original)** | **Después (Transformado)** |
|:-:|:-:|:-:|
| id / section_id | ID de la sección | Se mantiene igual |
| course / course_id | ID del curso | Se mantiene igual |
| name / section_name | Nombre de la sección | Se mantiene igual |
| sequence | Lista de IDs de módulos en una cadena de texto | Se convierte en filas separadas con module_id |

### Representación Gráfica de la Jerarquía

```
📚 Curso (course_id) → 101
   ├── 📂 Sección (section_id) → 10
   │    ├── 🧩 Módulo (module_id) → 201
   │    ├── 🧩 Módulo (module_id) → 202
   │    ├── 🧩 Módulo (module_id) → 203
   ├── 📂 Sección (section_id) → 11
   │    ├── 🧩 Módulo (module_id) → 204
   │    ├── 🧩 Módulo (module_id) → 205
```

## Relación entre Estudiantes y Cursos en Moodle
[create_parquet_student_courses](create_parquet_student_courses.py)

### Descripción
Este script genera un archivo **Parquet** que contiene la relación entre **estudiantes y los cursos en los que están inscritos** en Moodle. Utiliza los datos de inscripciones y métodos de matrícula para extraer la información y normalizarla en una tabla estructurada.

### Funcionamiento
1. **Carga de Datos**:
* Se leen los archivos Parquet con información de inscripciones y estudiantes:
  * `mdlvf_user_enrolments.parquet` → Contiene los usuarios inscritos y sus IDs de inscripción (enrolid).
  * `mdlvf_enrol.parquet` → Conecta los IDs de inscripción (enrolid) con los cursos (courseid).
  * `students.parquet` → Lista de usuarios registrados como estudiantes.

2. **Transformación de Datos**:
   * Se realiza un **JOIN** entre `mdlvf_user_enrolments` y `mdlvf_enrol` para obtener los cursos a los que está inscrito cada usuario.
   * Se filtran solo los estudiantes válidos cruzando con `students.parquet` (para evitar incluir usuarios que no sean estudiantes activos).
   * Se eliminan duplicados, dejando solo **una relación única por estudiante y curso**.

3. **Generación de la Tabla de Relación**:
   * Se guarda el resultado en un **archivo Parquet** (`student_courses.parquet`) para análisis y visualización.

### Estructura de las Tablas

1. **Tabla de Inscripciones de Usuarios (`mdlvf_user_enrolments.parquet`)**
Contiene la relación de usuarios con sus inscripciones.

| **Columna** | **Descripción** | **Ejemplo** |
|---|---|---|
| userid | ID único del usuario (estudiante). | 5001 |
| enrolid | ID de la inscripción que conecta con mdlvf_enrol. | 201 |
2. **Tabla de Métodos de Inscripción (`mdlvf_enrol.parquet`)**
Asocia las inscripciones con los cursos.

| **Columna** | **Descripción** | **Ejemplo** |
|:-:|:-:|:-:|
| id | ID del método de inscripción. | 201 |
| courseid | ID del curso asociado a la inscripción. | 101 |
3. **Tabla de Estudiantes (`students.parquet`)**
Lista de usuarios que están registrados como estudiantes en Moodle.

| **Columna** | **Descripción** | **Ejemplo** |
|:-:|:-:|:-:|
| UserID | ID único del estudiante. | 5001 |
| FullName | Nombre del estudiante. | "Juan Pérez" |

### Estructura de la Tabla de Salida (`student_courses.parquet`)
Después de la transformación, la información se organiza en la siguiente estructura:
| **Columna** | **Descripción** | **Ejemplo** |
|:-:|:-:|:-:|
| userid | ID único del estudiante. | 5001 |
| course_id | ID del curso en el que está inscrito el estudiante. | 101 |

### Representación Gráfica de la Jerarquía

```
🎓 Estudiante (userid) → 5001
   ├── 📜 Inscripción (enrolid) → 201
   │    ├── 📚 Curso (course_id) → 101
   ├── 📜 Inscripción (enrolid) → 202
   │    ├── 📚 Curso (course_id) → 102
   ├── 📜 Inscripción (enrolid) → 203
   │    ├── 📚 Curso (course_id) → 103 
```

## Relación entre Estudiantes y Actividades en Moodle
[create_parquet_rel_course_activity](create_parquet_rel_course_activity.py)

### Descripción
Este script genera un archivo **Parquet** con información sobre las **actividades de los cursos en los que están inscritos los estudiantes** en Moodle. Se integra información de múltiples fuentes para crear un mapeo detallado de la relación **estudiante ↔ curso ↔ sección ↔ actividad**.

### Funcionamiento
1. **Carga de Datos**:
   * Se leen cuatro archivos Parquet:
     * `activities_section_mapping.parquet` → Relación de secciones y módulos en cada curso.
     * `student_courses.parquet` → Relación de estudiantes y los cursos en los que están inscritos.
     * `mdlvf_course_modules.parquet` → Información detallada sobre los módulos (actividades) en Moodle.
     * `mdlvf_modules.parquet` → Relación de módulos con sus nombres (tipos de actividad).
2. **Transformación de Datos**:
   * Se unen los **estudiantes con sus cursos inscritos** (`student_courses.parquet`).
   * Se asocian las **secciones del curso y sus actividades** (`activities_section_mapping.parquet`).
   * Se extraen detalles adicionales sobre los **módulos** (`mdlvf_course_modules.parquet`), como su instance y su module_id.
   * Se obtiene el **nombre del tipo de actividad** (`mdlvf_modules.parquet`).

3. **Generación de la Tabla de Relación**:
   * Se guarda el resultado en un **archivo Parquet**(`student_course_activities.parquet`) para su posterior análisis.


### Estructura de las Tablas

1. **Tabla de Secciones y Actividades (`activities_section_mapping.parquet`)**
Esta tabla almacena información sobre las secciones de los cursos y los módulos asignados.

| **Columna** | **Descripción** | **Ejemplo** |
|:-:|:-:|:-:|
| course_id | ID del curso al que pertenece la sección. | 101 |
| section_id | ID único de la sección dentro del curso. | 10 |
| module_id | ID del módulo (actividad) dentro de la sección. | 201 |

2. **Tabla de Inscripciones de Estudiantes (`student_courses.parquet`)**

| **Columna** | **Descripción** | **Ejemplo** |
|:-:|:-:|:-:|
| userid | ID único del estudiante. | 5001 |
| course_id | ID del curso en el que está inscrito el estudiante. | 101 |
3. **Tabla de Módulos del Curso (`mdlvf_course_modules.parquet`)**

| **Columna** | **Descripción** | **Ejemplo** |
|:-:|:-:|:-:|
| id | ID del módulo (actividad). | 201 |
| module | ID del tipo de módulo. | 3 |
| instance | Identificador de la instancia del módulo. | 500 |
4. **Tabla de Tipos de Módulos (`mdlvf_modules.parquet`)**

| id | name |
|---|---|
| 1 | assign |
| 2 | assignment |
| 3 | book |
| 4 | chat |
| 5 | choice |
| 6 | data |
| 7 | feedback |
| 8 | folder |
| 9 | forum |
| 10 | glossary |
| 11 | imscp |
| 12 | label |
| 13 | lesson |
| 14 | lti |
| 15 | page |
| 16 | quiz |
| 17 | resource |
| 18 | scorm |
| 19 | survey |
| 20 | url |
| 21 | wiki |
| 22 | workshop |
| 23 | bootstrapelements |
| 24 | hvp |
| 26 | h5pactivity |

### Estructura de la Tabla de Salida (`student_course_activities.parquet`)
Después de la transformación, la información se organiza en la siguiente estructura:

| **Columna** | **Descripción** | **Ejemplo** |
|:-:|:-:|:-:|
| userid | ID único del estudiante. | 5001 |
| course_id | ID del curso en el que está inscrito el estudiante. | 101 |
| section_id | ID de la sección donde se encuentra la actividad. | 10 |
| module_id | ID del módulo (actividad) dentro de la sección. | 201 |
| activity_type | Tipo de actividad (Ej. "quiz", "forum"). | "quiz" |
| instance | Identificador único de la instancia del módulo. | 500 |


### Representación Gráfica de la Jerarquía

```
🎓 Estudiante (userid) → 5001
   ├── 📚 Curso (course_id) → 101
   │    ├── 📂 Sección (section_id) → 10
   │    │    ├── 🧩 Módulo (module_id) → 201
   │    │    │    ├── 📌 Tipo de Actividad (activity_type) → "quiz"
   │    │    │    ├── 🔢 Instancia de la actividad (instance) → 500
   │    │    ├── 🧩 Módulo (module_id) → 202
   │    │    │    ├── 📌 Tipo de Actividad (activity_type) → "forum"
   │    │    │    ├── 🔢 Instancia de la actividad (instance) → 501
   │    ├── 📂 Sección (section_id) → 11
   │    │    ├── 🧩 Módulo (module_id) → 203
   │    │    │    ├── 📌 Tipo de Actividad (activity_type) → "assignment"
   │    │    │    ├── 🔢 Instancia de la actividad (instance) → 502
```

