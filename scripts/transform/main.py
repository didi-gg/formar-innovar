import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from scripts.transform.process_unique_courses import UniqueCoursesProcessor
from scripts.transform.process_enrollments import EnrollmentProcessor
from scripts.transform.process_teacher_logs import TeacherLoginProcessor
from scripts.transform.process_teacher_user_moodle import TeacherMoodleUserProcessor
from scripts.transform.process_student_modules import StudentModulesProcessor
from scripts.transform.process_student_logs import StudentLoginProcessor
from scripts.transform.process_student_logins import StudentLoginsProcessor
from scripts.transform.process_student_courses import StudentMoodleCoursesProcessor
from scripts.transform.process_student_course_interactions import StudentCourseInteractions
from scripts.transform.process_modules_hvp import MoodleHVPProcessor
from scripts.transform.process_courses import CoursesProcessor
from scripts.transform.process_course_modules import MoodleModulesProcessor
from scripts.transform.process_course_modules_featured import ModuleFeaturesProcessor


# 1. Crear cursos únicos de Moodle y Edukrea
# Este script crea un dataset con los cursos únicos de Moodle y Edukrea
# Tiene las siguientes columnas: year, id_grado, course_id,course_name,sede,id_asignatura
processor = UniqueCoursesProcessor()
processor.process_course_data()
processor.close()

# 2. Procesar inscripciones de estudiantes
# Este script procesa las inscripciones de estudiantes para los años 2024 y 2025
# y crea un dataset con las siguientes columnas: documento_identificación, moodle_user_id, year, edukrea_user_id, id_grado, sede
processor = EnrollmentProcessor()
final_enrollments = processor.process_all_years()
processor.logger.info("Enrollment data processed successfully.")
processor.close()

# 3. Procesar logs de profesores
# Este script exporta los logs de los profesores relacionados a los cursos de moodle
# Crea un dataset con las siguientes columnas: year, id, eventname, component, action, target, objectid, contextinstanceid, userid, courseid, timecreated, origin, ip
processor = TeacherLoginProcessor()
processor.process_teacher_logs()
processor.logger.info("Teacher logs processed successfully.")
processor.close()

# 4. Procesar actividad de profesores
# Este script crea un dataset con la información de usuario de moodle de los profesores
# Tiene las siguientes columnas: id_docente, nombre, moodle_user_id, edukrea_user_id, sede
processor = TeacherMoodleUserProcessor()
processor.process_teacher_logs()
processor.logger.info("Teacher logs processed successfully.")
processor.close()

# 5. Procesar cursos de Moodle para estudiantes
# Este script crea un dataset con la relación entre los estudiantes y los cursos en los que están inscritos,
# Tiene las siguientes columnas: moodle_user_id, year, id_grado, course_id, course_name, documento_identificación, sede, id_asignatura
processor = StudentMoodleCoursesProcessor()
processor.process_student_courses()
processor.logger.info("Student courses processed successfully.")
processor.close()

# 6. Procesar logs de estudiantes
# Este script crea un dataset con los logs de los estudiantes relacionados a los cursos de moodle
# Crea un dataset con las siguientes columnas: year, id, eventname, component, action, target, objectid, contextinstanceid, userid, courseid, timecreated, origin, ip
processor = StudentLoginProcessor()
processor.process_student_logs()
processor.logger.info("Student logs processed successfully.")
processor.close()

# 7. Procesar inicios de sesión de estudiantes
# Este script crea un dataset con los inicios de sesión de los estudiantes en Moodle
# Crea un dataset con las siguientes columnas: documento_identificación, year, periodo, count_login, max_inactividad, count_login_mon, count_login_tue, count_login_wed, count_login_thu, count_login_fri, count_login_sat, count_login_sun, count_jornada_madrugada, count_jornada_mañana, count_jornada_tarde, count_jornada_noche
processor = StudentLoginsProcessor()
processor.process_moodle_logins()
processor.close()
processor.logger.info("Student logins processed successfully.")

# 8. Procesar módulos de Moodle
# Este script crea un dataset de los módulos de Moodle y Edukrea
# Tiene las siguientes columnas: year, course_id, course_module_id, sede, id_grado, id_asignatura, asignatura_name, course_name, section_id, section_name, module_type_id, instance, module_creation_date, module_type, module_name, week, period, is_interactive, is_in_english, planned_start_date, planned_end_date
processor = MoodleModulesProcessor()
processor.process_course_data()
processor.logger.info("Modules processed successfully.")
processor.close()

# 9. Procesar módulos HVP de Moodle
# Este script procesa los módulos HVP de Moodle y Edukrea
# Crea un dataset con las siguientes columnas: year, course_id, course_module_id, asignatura_name, sede, week, period, id_grado, machine_name, timecreated, timemodified, completionpass, course_hvp, instance, section_name, section_id, hvp_id, hvp_name, hvp_type, json_keys, h5p_libraries, json_kb, libraries_count, is_in_english, is_interactive
processor = MoodleHVPProcessor()
processor.process_all_hvp()
processor.close()
processor.logger.info("Moodle HVP processing completed.")

# 10. Procesar características de módulos de cursos
# Este script procesa las características de los módulos de cursos de Moodle y Edukrea
# Crea un dataset con las siguientes columnas:year, course_id, course_module_id, sede, id_grado, id_asignatura, asignatura_name, course_name, section_id, section_name, module_type_id, instance, module_creation_date, module_type, module_name, week, period, is_interactive, is_in_english, planned_start_date, planned_end_date, last_update_date, teacher_total_updates, days_since_creation, days_since_last_update, teacher_total_views, teacher_first_view_date, teacher_last_view_date, total_students, student_total_views, students_who_viewed, student_total_interactions, students_who_interacted, min_views_per_student, max_views_per_student, median_views_per_student, percent_students_interacted, percent_students_viewed, interaction_to_view_ratio, teacher_accessed_before_start, teacher_updated_before_start, teacher_updated_during_week_planned, teacher_active
processor = ModuleFeaturesProcessor()
processor.process_course_data()
processor.close()
processor.logger.info("Course modules processed successfully.")

# 11. Procesar módulos de estudiantes
# Este script crea un dataset con los módulos de los estudiantes en Moodle, es decir para cada estudiante todos los módulos de los cursos y sus interacciones
# Tiene las siguientes columnas:moodle_user_id, documento_identificación, year, course_id, sede, course_module_id, id_grado, id_asignatura, asignatura_name, course_name, section_id, section_name, module_type_id, instance, module_creation_date, module_type, module_name, week, period, is_interactive, is_in_english, planned_start_date, planned_end_date, num_views, num_interactions, first_view, last_view, has_viewed, has_participated, days_from_planned_start, days_after_end, was_on_time
processor = StudentModulesProcessor()
processor.process_course_data()
processor.logger.info("Student Modules processed successfully.")
processor.close()

# 12. Procesar interacciones de estudiantes en cursos
# Este script hace un agrupamiento de los resultados en StudentModulesProcessor por curso y estudiante
# Crea un dataset con las siguiente columnas: moodle_user_id, documento_identificación, course_id, period, year, sede, total_views, total_interactions, num_modules_viewed, num_modules_interacted, total_modules, percent_modules_viewed, percent_modules_interacted
processor = StudentCourseInteractions()
processor.process_all_course_interactions()
processor.close()
processor.logger.info("Student course interactions processed successfully.")

# 13. Procesar cursos de Moodle
# Este script hace un agrupamiento de los resultados en ModuleFeaturesProcessor por curso
# Crea un dataset con las siguientes columnas: sede, id_grado, id_asignatura, asignatura_name, course_id, course_name, period, year, total_students, count_evaluation, count_collaboration, count_content, count_in_english, count_interactive, num_modules, num_modules_updated, num_teacher_views_before_planned_start_date, teacher_total_updates, teacher_total_views, student_total_views, student_total_interactions, min_days_since_creation, max_days_since_creation, avg_days_since_creation, median_days_since_creation, avg_days_since_last_update, median_days_since_last_update, percent_evaluation, percent_collaboration, percent_content, percent_in_english, percent_interactive, percent_updated, num_students, num_students_viewed, num_students_interacted, num_modules_viewed, avg_views_per_student, median_views_per_student, avg_interactions_per_student, median_interactions_per_student, id_least_viewed_module, students_viewed_least_module, id_most_late_opened_module, days_from_planned_start, percent_modules_out_of_date, percent_students_viewed, percent_students_interacted, percent_modules_viewed
processor = CoursesProcessor()
processor.process_course_data()
processor.logger.info("Modules processed successfully.")
processor.close()
