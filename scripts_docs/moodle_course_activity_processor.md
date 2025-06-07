# Procesamiento Eventos de Moodle por Asignatura

## Descripción
Este script procesa eventos de interacción académica de los estudiantes dentro de Moodle durante los años 2024 y 2025, enfocándose en actividades relacionadas con cursos, recursos, tareas, evaluaciones y foros.

El objetivo principal es consolidar la actividad por estudiante, curso, periodo y asignatura, generando un resumen estructurado para análisis posteriores.

---

## ¿Qué hace el script?

- Lee logs de eventos de Moodle (`logstore_standard_log`) para los años 2024 y 2025.
- Filtra únicamente eventos relevantes para la actividad académica.
- Clasifica los eventos en categorías: acceso a contenido, entrega de evaluaciones, participación en foros.
- Asocia cada evento con un curso y una asignatura oficial del estudiante (basado en la matrícula previa).
- Organiza los eventos por periodo académico.
- Genera un resumen tipo pivot table con conteos por tipo de evento.
- Exporta el resultado a un CSV limpio para análisis.

---

## Funcionamiento detallado

1. **Carga de datos de logs Moodle**:
   - Lee archivos `.parquet` de logs de actividad.
   - Solo se procesan eventos predefinidos de interés ([Eventos de interés y su clasificación](#eventos-de-interés-y-su-clasificación)).

2. **Conversión de Timestamps**:
   - Convierte `timecreated` a formato datetime, ajustado a la zona horaria Bogotá.

3. **Asignación de Periodo Académico**:
   - Se utiliza `AcademicPeriodUtils` para asignar a qué periodo (bimestre) pertenece cada evento basado en su fecha.

4. **Filtrado de cursos válidos**:
   - Se cruza la actividad con la matrícula de estudiantes (`student_moodle_courses.csv`) para considerar solo cursos reales.

5. **Clasificación de eventos**:
   - Cada evento se mapea a un tipo de actividad más amigable.

6. **Consolidación**:
   - Se crea un resumen que cuenta cuántos eventos de cada tipo realizó cada estudiante en cada curso por periodo.

7. **Exportación**:
   - Se guarda el resultado en `data/interim/moodle/moodle_course_activity_summary.csv`.

---

## Estructura de las Tablas Fuente

### Tabla: `data/raw/moodle/2024/Log/mdlvf_logstore_standard_log.parquet`
| **Columna** | **Descripción** |
|-------------|------------------|
| id | ID del evento |
| eventname | Nombre del evento generado |
| component | Componente Moodle que generó el evento |
| action | Acción específica (create, view, update) |
| target | Objeto del evento (course, user, forum, etc.) |
| userid | ID del usuario que generó el evento |
| courseid | ID del curso relacionado |
| timecreated | Timestamp de creación del evento |

(Se usan principalmente `eventname`, `userid`, `courseid`, `timecreated`)

---

### Tabla: `data/interim/moodle/student_moodle_courses.csv`
| **Columna** | **Descripción** |
|-------------|------------------|
| moodle_user_id | ID de usuario en Moodle |
| course_id | ID del curso en Moodle |
| id_asignatura | Asignatura oficial mapeada |
| year | Año académico |
| course_name | Nombre del curso |

---

## Estructura de la Tabla de Salida

### Tabla: `data/interim/moodle/moodle_course_activity_summary.csv`
| **Columna** | **Descripción** |
|-------------|------------------|
| userid | ID del estudiante |
| courseid | ID del curso |
| period | Periodo académico (ej: 2024-1, 2024-2) |
| id_asignatura | ID de la asignatura oficial |
| year | Año académico |
| course_name | Nombre del curso |
| (eventos...) | Contadores de cada tipo de evento |

---

## Eventos de interés y su clasificación

Los eventos se agrupan en tres grandes categorías:

### 1. Acceso a Contenido

| **Evento (interno)** | **Significado** |
|----------------------|-----------------|
| course_viewed | Acceso al curso principal |
| page_module_viewed | Lectura de una página de contenido |
| resource_module_viewed | Visualización de un recurso descargable |
| url_module_viewed | Acceso a un enlace web externo |
| hvp_module_viewed | Visualización de contenido interactivo H5P |
| feedback_module_viewed | Visualización de un formulario de retroalimentación |
| choice_module_viewed | Visualización de una encuesta simple |
| lti_module_viewed | Acceso a herramientas externas (LTI) |
| chat_module_viewed | Acceso a chats en Moodle |

---

### 2. Evaluaciones y Entregas

| **Evento (interno)** | **Significado** |
|----------------------|-----------------|
| assign_module_viewed | Acceso a una tarea asignada |
| submission_status_viewed | Revisión del estado de entrega |
| assign_submission_form_viewed | Visualización del formulario de entrega |
| assign_assessable_submitted | Entrega formal de una tarea |
| assign_feedback_viewed | Visualización de retroalimentación de una tarea |
| file_assessable_uploaded | Subida de archivo para evaluación |
| file_submission_created | Creación de una entrega de archivo |
| onlinetext_assessable_uploaded | Entrega de texto en línea |
| onlinetext_submission_created | Creación de texto en línea para evaluación |
| quiz_module_viewed | Visualización de un examen |
| quiz_attempt_started | Inicio de un intento de examen |
| quiz_attempt_submitted | Entrega de un intento de examen |
| quiz_attempt_viewed | Visualización de resultados de un intento |
| quiz_attempt_summary_viewed | Vista del resumen del intento |
| quiz_attempt_reviewed | Revisión detallada de un examen |
| quiz_attempt_preview_started | Inicio de un intento de prueba |
| choice_answer_created | Registro de una respuesta en una encuesta |
| feedback_response_submitted | Envío de una respuesta a un formulario de retroalimentación |

---

### 3. Participación Social (Foros)

| **Evento (interno)** | **Significado** |
|----------------------|-----------------|
| forum_module_viewed | Acceso a un foro de discusión |
| forum_post_created | Creación de un nuevo post en un foro |
| forum_post_updated | Actualización de un post existente |
| forum_post_deleted | Eliminación de un post |
| forum_discussion_viewed | Visualización de una discusión en foro |
| forum_discussion_created | Creación de una discusión nueva |
| forum_discussion_deleted | Eliminación de una discusión |
| forum_subscription_created | Suscripción a un foro |
| forum_discussion_subscription_created | Suscripción a una discusión específica |
| forum_discussion_subscription_deleted | Eliminación de la suscripción a una discusión |
| forum_course_searched | Búsqueda realizada dentro de foros |
| forum_assessable_uploaded | Entrega evaluable en foros (p.ej., tarea debatida) |

---

# Notas Importantes:

- **Timezone**: Todos los eventos se transforman a hora local Bogotá (UTC-5).
- **Compleción de columnas**: Se aseguran todas las columnas esperadas, incluso si un evento no aparece en los datos crudos.
- **Solo cursos válidos**: Se consideran únicamente cursos en los que los estudiantes están inscritos oficialmente.