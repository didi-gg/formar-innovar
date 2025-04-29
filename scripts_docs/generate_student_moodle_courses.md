# Procesamiento Relación entre Estudiantes y Cursos en Moodle
[generate_student_moodle_courses](generate_student_moodle_coursesy.py)

## Descripción
Este script genera un archivo CSV que relaciona estudiantes con los cursos en los que están inscritos en Moodle, durante los años 2024 y 2025. Además, asigna a cada curso una asignatura (id_asignatura) oficial.

Se excluyen cursos no académicos o no relevantes para el análisis, como cursos administrativos, pruebas, y programas de inteligencia emocional.
---

## ¿Qué hace el script?
- Lee archivos `.parquet` de inscripciones y cursos de Moodle (`user_enrolments`, `enrol`, `course`).
- Une las inscripciones de usuarios con la base de estudiantes (`enrollments.csv`).
- Filtra:
  - Cursos de prueba (IDs: 549, 550, 332).
  - Cursos de inteligencia emocional sin calificación (IDs: 154, 386, 411, 515, 155, 390, 394, 156, 157, 398, 158, 402, 159, 416, 160, 418, 213, 502, 409, 565).
- Mapea los cursos restantes a un `id_asignatura` oficial según un diccionario definido.
- Consolida los registros de 2024 y 2025.
- Guarda el resultado en un archivo CSV limpio (`student_moodle_courses.csv`).

---

## Funcionamiento
1. **Carga** de inscripciones de usuarios y cursos de Moodle (años 2024 y 2025).
2. **Cruce** de datos con los estudiantes matriculados (`enrollments.csv`).
3. **Filtrado** de cursos no relevantes (administrativos o no evaluativos).
4. **Mapeo** de cada curso a una asignatura curricular (`id_asignatura`).
5. **Unión** de los registros de los dos años (2024 y 2025).
6. **Exportación** a un archivo CSV final para su posterior análisis.

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

3. **Tabla de Estudiantes (`enrollments.csv`)**
Esta tabla contiene **información de la matrícula oficial** de los estudiantes, incluyendo su identificación y sede.

| **Columna** | **Descripción** | **Ejemplo** |
|:------------|:----------------|:------------|
| documento_identificación | Documento de identidad del estudiante (cifrado/anónimo) | b8bd5170... |
| moodle_user_id | ID del estudiante en Moodle | 1561 |
| year | Año académico al que corresponde la matrícula | 2024 |
| edukrea_user_id | ID en Edukrea (solo disponible para estudiantes 2025) | NaN |
| id_grado | Grado académico que cursa el estudiante | 1 |
| sede | Sede educativa del estudiante (Fusagasugá o Girardot) | Fusagasugá |

Notas importantes:
- **edukrea_user_id** solo aplica para estudiantes del año **2025**.
- Se excluyen **cursos de prueba** e **institucionales** en el procesamiento (IDs: 549, 550, 332, etc.).
- Se realiza un mapeo especial para asignar **id_asignatura** a partir de **course_id**.

### Estructura de la Tabla de Salida (`student_moodle_courses.csv`)
Después de la transformación, la información se organiza en la siguiente estructura:
| **Columna** | **Descripción** |
|-------------|------------------|
| moodle_user_id | ID del estudiante en Moodle |
| year | Año académico |
| id_grado | Grado escolar |
| course_id | ID del curso en Moodle |
| course_name | Nombre del curso en Moodle |
| id_asignatura | ID de la asignatura oficial asociada |

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