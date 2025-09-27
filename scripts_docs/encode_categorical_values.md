# Codificación de Variables Categóricas - FICC

## Descripción General

El script `encode_categorical_values.py` aplica transformaciones sistemáticas a las variables categóricas del dataset del proyecto FICC. Este documento detalla todas las transformaciones implementadas con ejemplos prácticos de cómo se transforma la data.

## Tipos de Transformaciones

El script implementa cuatro tipos principales de codificación:

1. **Variables Binarias** - Transformación a 0/1 con renombrado semántico
2. **Variables Ordinales** - Codificación numérica preservando el orden natural
3. **Variables Dummy** - Para listas separadas por comas
4. **One-Hot Encoding** - Para variables categóricas nominales

---

## 1. Variables Binarias

Las variables binarias se transforman a valores 0/1 y se renombran para mayor claridad semántica.

### 1.1 País de Origen → Es Colombiano

**Variable original:** `país_origen`  
**Variable nueva:** `es_colombiano`

**Transformación:**

- `'Colombia'` → `1`
- Cualquier otro país → `0`

**Ejemplo:**

| ANTES: país_origen | DESPUÉS: es_colombiano |
|-------------------|------------------------|
| Colombia          | 1                      |
| Venezuela         | 0                      |
| Colombia          | 1                      |
| Ecuador           | 0                      |

### 1.2 Medio de Transporte → Vehículo Privado

**Variable original:** `medio_transporte`  
**Variable nueva:** `medio_transporte_vehiculo_privado`

**Transformación:**

- `'Vehículo privado'` → `1`
- Cualquier otro medio → `0`

**Ejemplo:**

| ANTES: medio_transporte | DESPUÉS: medio_transporte_vehiculo_privado |
|------------------------|-------------------------------------------|
| Vehículo privado       | 1                                         |
| Transporte público     | 0                                         |
| Vehículo privado       | 1                                         |
| Bicicleta              | 0                                         |

### 1.3 Tipo de Vivienda → Es Alquiler

**Variable original:** `tipo_vivienda`  
**Variable nueva:** `es_alquiler`

**Transformación:**
- `'Alquilada'` → `1`
- Cualquier otro tipo → `0`

**Ejemplo:**

| ANTES: tipo_vivienda | DESPUÉS: es_alquiler |
|---------------------|---------------------|
| Propia              | 0                   |
| Alquilada           | 1                   |
| Familiar            | 0                   |
| Alquilada           | 1                   |

### 1.4 Zona de Vivienda → Zona Urbana

**Variable original:** `zona_vivienda`  
**Variable nueva:** `zona_vivienda_urbana`

**Transformación:**
- `'Urbana'` → `1`
- `'Rural'` → `0`

**Ejemplo:**

| ANTES: zona_vivienda | DESPUÉS: zona_vivienda_urbana |
|---------------------|------------------------------|
| Urbana              | 1                            |
| Rural               | 0                            |
| Urbana              | 1                            |
| Rural               | 0                            |

### 1.5 Rol Adicional → Tiene Rol Adicional

**Variable original:** `rol_adicional`  
**Variable nueva:** `tiene_rol_adicional`

**Transformación:**
- `'Ninguno'` → `0`
- Cualquier otro rol → `1`

**Ejemplo:**

| ANTES: rol_adicional | DESPUÉS: tiene_rol_adicional |
|---------------------|------------------------------|
| Ninguno             | 0                            |
| Monitor             | 1                            |
| Representante       | 1                            |
| Ninguno             | 0                            |

### 1.6 Género → Es Masculino

**Variable original:** `género`  
**Variable nueva:** `es_masculino`

**Transformación:**
- `'Masculino'` → `1`
- `'Femenino'` → `0`

**Ejemplo:**

| ANTES: género | DESPUÉS: es_masculino |
|---------------|----------------------|
| Masculino     | 1                    |
| Femenino      | 0                    |
| Masculino     | 1                    |
| Femenino      | 0                    |

### 1.7 NEE → Tiene NEE

**Variable original:** `nee`  
**Variable nueva:** `tiene_nee`

**Transformación:**
- `'Sí'` → `1`
- `'No'` → `0`

**Ejemplo:**

| ANTES: nee | DESPUÉS: tiene_nee |
|------------|-------------------|
| Sí         | 1                 |
| No         | 0                 |
| No         | 0                 |
| Sí         | 1                 |

### 1.8 Enfermedades → Tiene Enfermedades

**Variable original:** `enfermedades`  
**Variable nueva:** `tiene_enfermedades`

**Transformación:**
- `'Sí'` → `1`
- `'No'` → `0`

**Ejemplo:**

| ANTES: enfermedades | DESPUÉS: tiene_enfermedades |
|--------------------|----------------------------|
| No                 | 0                          |
| Sí                 | 1                          |
| No                 | 0                          |
| No                 | 0                          |

### 1.9 Antigüedad → Es Antiguo

**Variable original:** `antigüedad`  
**Variable nueva:** `es_antiguo`

**Transformación:**
- `'Antiguo'` → `1`
- `'Nuevo'` → `0`

**Ejemplo:**

| ANTES: antigüedad | DESPUÉS: es_antiguo |
|------------------|-------------------|
| Nuevo            | 0                 |
| Antiguo          | 1                 |
| Nuevo            | 0                 |
| Antiguo          | 1                 |

---

## 2. Variables Ordinales

Las variables ordinales mantienen su orden natural pero se codifican numéricamente.

### 2.1 Participación en Clase

**Variable original:** `participación_clase`  
**Variable nueva:** `participacion_clase`

**Mapeo ordinal:**
- `'Baja'` → `1`
- `'Media'` → `2`
- `'Alta'` → `3`

**Ejemplo:**

| ANTES: participación_clase | DESPUÉS: participacion_clase |
|---------------------------|------------------------------|
| Alta                      | 3                            |
| Baja                      | 1                            |
| Media                     | 2                            |
| Alta                      | 3                            |

### 2.2 Apoyo Familiar

**Variable original:** `apoyo_familiar`  
**Variable nueva:** `apoyo_familiar` (mismo nombre)

**Mapeo ordinal:**
- `'Bajo'` → `1`
- `'Medio'` → `2`
- `'Alto'` → `3`

**Ejemplo:**

| ANTES: apoyo_familiar | DESPUÉS: apoyo_familiar |
|----------------------|------------------------|
| Alto                 | 3                      |
| Bajo                 | 1                      |
| Medio                | 2                      |
| Alto                 | 3                      |

### 2.3 Nivel de Motivación

**Variable original:** `nivel_motivación`  
**Variable nueva:** `nivel_motivación` (mismo nombre)

**Mapeo ordinal:**
- `'Bajo'` → `1`
- `'Medio'` → `2`
- `'Alto'` → `3`

**Ejemplo:**

| ANTES: nivel_motivación | DESPUÉS: nivel_motivación |
|------------------------|--------------------------|
| Medio                  | 2                        |
| Alto                   | 3                        |
| Bajo                   | 1                        |
| Alto                   | 3                        |

### 2.4 Demuestra Confianza → Nivel de Confianza

**Variable original:** `demuestra_confianza`  
**Variable nueva:** `nivel_confianza`

**Mapeo ordinal:**
- `'Nunca lo demuestra'` → `0`
- `'Rara vez lo demuestra'` → `1`
- `'A veces lo demuestra'` → `2`
- `'Frecuentemente lo demuestra'` → `3`
- `'Siempre lo demuestra'` → `4`

**Ejemplo:**

| ANTES: demuestra_confianza | DESPUÉS: nivel_confianza |
|---------------------------|-------------------------|
| Siempre lo demuestra      | 4                       |
| Nunca lo demuestra        | 0                       |
| A veces lo demuestra      | 2                       |
| Frecuentemente lo demuestra | 3                     |

### 2.5 Interés en Estudios Superiores

**Variable original:** `interés_estudios_superiores`  
**Variable nueva:** `interes_estudios_superiores`

**Mapeo ordinal:**
- `'Bajo'` → `1`
- `'Medio'` → `2`
- `'Alto'` → `3`

**Ejemplo:**

| ANTES: interés_estudios_superiores | DESPUÉS: interes_estudios_superiores |
|-----------------------------------|-------------------------------------|
| Alto                              | 3                                   |
| Bajo                              | 1                                   |
| Medio                             | 2                                   |
| Alto                              | 3                                   |

### 2.6 Nivel Educativo → Educación Profesional

**Variable original:** `nivel_educativo`  
**Variable nueva:** `educación_profesional`

**Mapeo binario:**
- `'Técnico'` → `0`
- `'Pregrado'` → `1`
- `'Maestría'` → `1`
- Otros valores → `0`

**Ejemplo:**

| ANTES: nivel_educativo | DESPUÉS: educación_profesional |
|-----------------------|-------------------------------|
| Pregrado              | 1                             |
| Técnico               | 0                             |
| Maestría              | 1                             |
| Bachillerato          | 0                             |

---

## 3. Variables Dummy

Para variables que contienen listas de elementos separados por comas, se crean variables dummy que indican la presencia de cada elemento.

### 3.1 Actividades Extracurriculares

**Variable original:** `actividades_extracurriculares`  
**Variables nuevas:** `actividad_artes`, `actividad_deportes`, `actividad_idiomas`

**Lógica de transformación:**
- `actividad_artes`: 1 si contiene "Arte" o "arte"
- `actividad_deportes`: 1 si contiene "Deporte" o "deporte"  
- `actividad_idiomas`: 1 si contiene "Idioma" o "idioma"

**Ejemplo:**

| ANTES: actividades_extracurriculares | actividad_artes | actividad_deportes | actividad_idiomas |
|-------------------------------------|----------------|--------------------|------------------|
| Arte, Deporte                       | 1              | 1                  | 0                |
| Idioma                              | 0              | 0                  | 1                |
| Deporte, Idioma                     | 0              | 1                  | 1                |
| Arte                                | 1              | 0                  | 0                |

### 3.2 Composición Familiar

**Variable original:** `familia`  
**Variables nuevas:** `familia_madre`, `familia_padre`, `familia_otros`

**Lógica de transformación:**
- `familia_madre`: 1 si contiene "Mamá", "mamá", "Madre" o "madre"
- `familia_padre`: 1 si contiene "Papá", "papá", "Padre" o "padre"
- `familia_otros`: 1 si contiene "otros", "hermanos", "abuelos" o "tíos"

**Ejemplo:**

| ANTES: familia | familia_madre | familia_padre | familia_otros |
|---------------|---------------|---------------|---------------|
| Mamá, Papá    | 1             | 1             | 0             |
| Madre, hermanos | 1           | 0             | 1             |
| Papá, abuelos | 0             | 1             | 1             |
| Mamá, otros   | 1             | 0             | 1             |

---

## 4. One-Hot Encoding

Para variables categóricas nominales (sin orden natural), se aplica one-hot encoding creando una columna binaria para cada categoría.

### 4.1 Proyección Vocacional

**Variable original:** `proyección_vocacional`  
**Variables nuevas:** `proyeccion_[categoría]` para cada valor único

**Ejemplo:**

| ANTES: proyección_vocacional | proyeccion_Educación | proyeccion_Ingeniería | proyeccion_Medicina |
|------------------------------|---------------------|-----------------------|--------------------|
| Ingeniería                   | 0                   | 1                     | 0                  |
| Medicina                     | 0                   | 0                     | 1                  |
| Ingeniería                   | 0                   | 1                     | 0                  |
| Educación                    | 1                   | 0                     | 0                  |

### 4.2 Jornada Preferida

**Variable original:** `jornada_preferida`  
**Variables nuevas:** `jornada_[categoría]` para cada valor único

**Ejemplo:**

| ANTES: jornada_preferida | jornada_Mañana | jornada_Noche | jornada_Tarde |
|-------------------------|----------------|---------------|---------------|
| Mañana                  | 1              | 0             | 0             |
| Tarde                   | 0              | 0             | 1             |
| Mañana                  | 1              | 0             | 0             |
| Noche                   | 0              | 1             | 0             |

### 4.3 Día Preferido

**Variable original:** `dia_preferido`  
**Variables nuevas:** `dia_[categoría]` para cada valor único

**Ejemplo:**

| ANTES: dia_preferido | dia_Lunes | dia_Miércoles | dia_Viernes |
|---------------------|-----------|---------------|-------------|
| Lunes               | 1         | 0             | 0           |
| Viernes             | 0         | 0             | 1           |
| Miércoles           | 0         | 1             | 0           |
| Lunes               | 1         | 0             | 0           |

---

## 5. Variables Numéricas

Estas variables se mantienen como están, solo se asegura que sean de tipo numérico.

### 5.1 Estrato Socioeconómico

**Variable:** `estrato`  
**Transformación:** Conversión a numérico (valores 1-6)

### 5.2 Período Académico

**Variable:** `period`  
**Transformación:** Conversión a numérico (valores 1,2,3,4)

---

## Resumen del Proceso

### Flujo de Transformación

1. **Carga del dataset original**
2. **Aplicación secuencial de transformaciones:**
   - Variables binarias (9 transformaciones)
   - Variables ordinales (6 transformaciones)
   - Variables dummy (2 grupos de variables)
   - One-hot encoding (3 variables)
   - Verificación de variables numéricas (2 variables)

## Uso del Script

### Desde línea de comandos

```bash
python scripts/transform/encode_categorical_values.py --input data/input.csv --output data/output.csv
```

### Como módulo

```python
from scripts.transform.encode_categorical_values import encode_categorical_variables
df_encoded, processed_features = encode_categorical_variables(df)
```

## Consideraciones Técnicas

1. **Preservación de datos:** El script trabaja con copias del DataFrame original
2. **Manejo de valores faltantes:** Se utiliza `na=False` en operaciones de string para evitar errores
3. **Logging:** Todas las transformaciones se registran para trazabilidad
4. **Flexibilidad:** El script verifica la existencia de columnas antes de procesarlas
5. **Consistencia:** Los nombres de las nuevas variables siguen convenciones semánticas claras

## Beneficios de estas Transformaciones

1. **Preparación para ML:** Los datos quedan listos para algoritmos de machine learning
2. **Interpretabilidad:** Los nombres de variables son más descriptivos
3. **Eficiencia:** Reducción de dimensionalidad en algunos casos
4. **Consistencia:** Todas las variables categóricas siguen el mismo patrón de codificación
5. **Trazabilidad:** Registro completo de todas las transformaciones aplicadas
