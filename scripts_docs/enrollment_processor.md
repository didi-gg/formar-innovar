# Procesamiento Matrículas de Estudiantes

## Descripción
Este script procesa, cruza y estandariza la información de inscripciones (matrículas) de estudiantes en la plataforma Moodle para los años 2024 y 2025, generando un archivo unificado `enrollments.csv`.

El procesamiento incluye la asociación de estudiantes con sus ID de Moodle y Edukrea, junto con la asignación de grado y sede. Aplica encriptado seguro (`hash`) a documentos de identidad para proteger la información sensible.

---

## ¿Qué hace el script?

- Carga inscripciones de usuarios en Moodle (archivos `.parquet` de usuarios y atributos).
- Aplica hashing estable a los documentos de identidad.
- Cruza los datos de Moodle con bases internas de estudiantes (`estudiantes_2024_hashed.csv` y `estudiantes_imputed_encoded.csv`).
- Recupera o genera el `edukrea_user_id` en 2025, si está disponible.
- Asocia cada estudiante a su respectivo grado y sede escolar.
- Elimina estudiantes de grados de preescolar (Prejardín, Jardín, Transición).
- Exporta un archivo CSV de inscripciones limpio y consolidado.

---

## Funcionamiento detallado

1. **Carga de datos de usuarios Moodle**:
   - A partir de los archivos `mdlvf_user.parquet` y `mdlvf_user_info_data.parquet` para 2024 y 2025.
   - Solo se seleccionan registros de usuarios con rol de *Estudiante* y que no estén eliminados.

2. **Hash de documentos de identidad**:
   - Se utiliza la clase `HashUtility` para proteger los documentos mediante un hash seguro y reproducible.

3. **Cruce con base de estudiantes**:
   - Se une cada estudiante Moodle con su registro en `estudiantes_2024_hashed.csv` o `estudiantes_imputed_encoded.csv`.
   - Se asegura la presencia del atributo `sede` para cada estudiante.

4. **Asignación de ID de Edukrea (solo 2025)**:
   - Se cruza opcionalmente con usuarios de Edukrea para obtener el `edukrea_user_id`, solo para datos del 2025.

5. **Asociación de grado escolar**:
   - Se cruza con la tabla `grados.csv` para asociar cada estudiante a su `id_grado` oficial.

6. **Depuración**:
   - Se eliminan los registros correspondientes a grados de preescolar.

7. **Consolidación**:
   - Se combinan las bases de 2024 y 2025.
   - Se guarda el archivo final `data/interim/estudiantes/enrollments.csv`.

---

## Estructura de las Tablas Fuente

### Tabla: `data/raw/moodle/2024/Users/mdlvf_user.parquet`
| **Columna** | **Descripción** |
|-------------|------------------|
| id | ID del usuario en Moodle |
| idnumber | Documento de identificación del usuario |
| firstname | Primer nombre del usuario |
| lastname | Apellido del usuario |
| city | Ciudad/sede asociada al usuario |
| firstaccess | Timestamp del primer acceso |
| lastaccess | Timestamp del último acceso |
| lastlogin | Timestamp del último inicio de sesión |
| timecreated | Timestamp de creación de la cuenta |
| deleted | Estado de eliminación de la cuenta |

---

### Tabla: `data/raw/moodle/2024/Users/mdlvf_user_info_data.parquet`
| **Columna** | **Descripción** |
|-------------|------------------|
| userid | ID del usuario en Moodle |
| data | Rol adicional o atributo personalizado del usuario (Ej: Estudiante) |

---

### Tabla: `data/raw/moodle/Edukrea/Users/mdl_user.parquet`
| **Columna** | **Descripción** |
|-------------|------------------|
| id | ID del usuario en Edukrea |
| idnumber | Documento de identidad en Edukrea |
| deleted | Estado de eliminación |

(Usada solo para el año 2025)

---

### Tabla: `data/interim/estudiantes/estudiantes_2024_hashed.csv`
| **Columna** | **Descripción** |
|-------------|------------------|
| documento_identificación | Documento de identidad (hashed) |
| grado | Grado escolar |
| sede | Sede del estudiante |

---

### Tabla: `data/interim/estudiantes/estudiantes_imputed_encoded.csv`
| **Columna** | **Descripción** |
|-------------|------------------|
| documento_identificación | Documento de identidad (hashed) |
| grado | Grado escolar |
| sede | Sede del estudiante |
| (otras variables codificadas o imputadas) |

---

### Tabla: `data/raw/tablas_maestras/grados.csv`
| **Columna** | **Descripción** |
|-------------|------------------|
| grado | Nombre del grado |
| ID | ID oficial del grado usado para unificación |

---

## Estructura de la Tabla de Salida

### Tabla: `data/interim/estudiantes/enrollments.csv`
| **Columna** | **Descripción** |
|-------------|------------------|
| documento_identificación | Documento de identidad del estudiante (hashed) |
| moodle_user_id | ID del estudiante en Moodle |
| year | Año académico |
| edukrea_user_id | ID del estudiante en Edukrea (solo en 2025, si aplica) |
| id_grado | ID oficial del grado escolar |
| sede | Sede educativa (Girardot o Fusagasugá) |

---

# Nota importante:
- **Preescolar filtrado**: Se eliminan registros correspondientes a Prejardín, Jardín y Transición.
- **Protección de datos**: No se almacenan documentos de identidad en texto plano; todos los documentos están hasheados.
- **Integridad de datos**: Solo estudiantes válidamente enrolados y activos son considerados.