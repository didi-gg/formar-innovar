# Procesamiento Logins en Moodle

## Descripción
Este script procesa los eventos de login de los estudiantes en la plataforma Moodle durante los años 2024 y 2025.  
Permite construir un resumen de los patrones de acceso de los usuarios, incluyendo cantidad de inicios de sesión, inactividad máxima entre accesos, frecuencia de logins por día de la semana y por jornada del día.

Se hace un ajuste especial para excluir los logins ocurridos en vacaciones al momento de calcular inactividad.

---

## ¿Qué hace el script?

- Carga los registros de login de los archivos de logs de Moodle.
- Cruza los eventos con los estudiantes oficialmente matriculados (`enrollments.csv`).
- Procesa la fecha y hora de cada login.
- Determina a qué periodo académico pertenece cada acceso.
- Calcula:
  - Número total de logins por estudiante y periodo.
  - Tiempo máximo de inactividad entre logins (en horas).
  - Número de logins por día de la semana (lunes a domingo).
  - Número de logins por jornada del día (madrugada, mañana, tarde, noche).
- Excluye los eventos ocurridos en vacaciones para el análisis de inactividad.
- Exporta un archivo CSV consolidado para su análisis posterior.

---

## Funcionamiento detallado

1. **Carga de datos de logs Moodle**:
   - Lee archivos `.parquet` de logs del evento `\core\event\user_loggedin`.
   - Solo se consideran eventos de login reales.

2. **Conversión de timestamps**:
   - `timecreated` se convierte a fecha-hora y se ajusta a la zona horaria Bogotá (UTC-5).

3. **Asignación de periodo académico**:
   - Se asigna automáticamente el bimestre en el que ocurrió cada login usando `AcademicPeriodUtils`.

4. **Filtrado de vacaciones**:
   - Se marca cada login si ocurrió en vacaciones escolares.

5. **Cálculo de inactividad**:
   - Se calcula el tiempo máximo (en horas) entre logins consecutivos en el mismo periodo, **excluyendo** días de vacaciones.

6. **Conteo de logins**:
   - Total de logins por usuario, periodo y año.
   - Número de logins realizados cada día de la semana (`monday`, `tuesday`, etc.).
   - Número de logins por jornada del día (`madrugada`, `mañana`, `tarde`, `noche`).

7. **Consolidación final**:
   - Se combinan todos los conteos en un único DataFrame.

8. **Exportación**:
   - Se guarda el resumen final en `data/interim/moodle/moodle_logins.csv`.

---

## Estructura de las Tablas Fuente

### Tabla: `data/raw/moodle/2024/Log/mdlvf_logstore_standard_log.parquet`
| **Columna** | **Descripción** |
|-------------|------------------|
| id | ID del evento |
| eventname | Nombre del evento generado (`\core\event\user_loggedin`) |
| userid | ID del usuario que realizó el login |
| timecreated | Timestamp en formato UNIX del evento |

(Se utilizan solo los eventos `user_loggedin`.)

---

### Tabla: `data/interim/estudiantes/enrollments.csv`
| **Columna** | **Descripción** |
|-------------|------------------|
| documento_identificación | Hash del documento del estudiante |
| moodle_user_id | ID del estudiante en Moodle |
| year | Año académico (2024 o 2025) |
| edukrea_user_id | ID en Edukrea (solo 2025) |
| id_grado | ID del grado escolar |
| sede | Sede educativa (Fusagasugá o Girardot) |

---

## Estructura de la Tabla de Salida

### Tabla: `data/interim/moodle/moodle_logins.csv`
| **Columna** | **Descripción** |
|-------------|------------------|
| userid | ID del usuario en Moodle |
| year | Año académico (2024 o 2025) |
| periodo | Periodo académico (ej: `2024-1`) |
| count_login | Número total de logins en ese periodo |
| max_inactividad | Máximo tiempo de inactividad entre logins (en horas) |
| documento_identificación | Hash del documento del estudiante |
| count_login_mon | Número de logins el lunes |
| count_login_tue | Número de logins el martes |
| count_login_wed | Número de logins el miércoles |
| count_login_thu | Número de logins el jueves |
| count_login_fri | Número de logins el viernes |
| count_login_sat | Número de logins el sábado |
| count_login_sun | Número de logins el domingo |
| count_jornada_madrugada | Número de logins entre 0:00 - 5:59 |
| count_jornada_mañana | Número de logins entre 6:00 - 11:59 |
| count_jornada_tarde | Número de logins entre 12:00 - 17:59 |
| count_jornada_noche | Número de logins entre 18:00 - 23:59 |

---

## Consideraciones importantes

- **Vacaciones escolares**: No se cuentan para el cálculo de inactividad.
- **Duración máxima por defecto**: Si un estudiante no tiene dos logins consecutivos en un periodo, su `max_inactividad` será la duración total del periodo en horas.
- **Zonificación horaria**: Se usa América/Bogotá para todos los registros.
- **Filtrado de datos**: Solo logins de estudiantes oficialmente matriculados.
- **Tiempos de carga**: Se usa DuckDB para procesamiento rápido sobre los archivos Parquet.