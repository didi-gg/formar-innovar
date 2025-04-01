import pandas as pd


class StudentsDataCleaner:
    def __init__(self, dataframe):
        self.df = dataframe

        # Diccionarios de reemplazo solo para valores mal escritos
        self.replacements = {
            "país_origen": {},
            "tipo_vivienda": {},
            "zona_vivienda": {"URBANO": "Urbana", "Urbano": "Urbana"},
            "interés_estudios_superiores": {},
            "apoyo_familiar": {"Slto": "Alto", "Alta": "Alto"},
            "participación_clase": {"Medio": "Media", "Alto": "Alta", "Bajo": "Baja"},
            "nivel_motivación": {"MEDIIO": "Medio", "ALRTA": "Alto", "MALTA": "Alto", "Alta": "Alto", "Baja": "Bajo", "Media": "Medio"},
            "nee": {"SÍ": "Si"},
            "enfermedades": {"SÍ": "Si"},
            "medio_transporte": {
                "Vehiculo privado": "Vehículo privado",
                "Motocicleta": "Vehículo privado",
                "Moto": "Vehículo privado",
                "Transporte publico": "Transporte público",
                "Tranporte privado": "Vehículo privado",
            },
        }

        # Reglas de validación para asegurar valores permitidos (siempre en Proper Case)
        self.allowed_values = {
            "país_origen": ["Colombia", "Argentina", "España", "República dominicana"],
            "antigüedad": ["Nuevo", "Antiguo"],
            "zona_vivienda": ["Urbana", "Rural"],
            "tipo_vivienda": ["Propia", "Alquilada", "Familiar"],
            "interés_estudios_superiores": ["Alto", "Medio", "Bajo"],
            "apoyo_familiar": ["Alto", "Medio", "Bajo"],
            "participación_clase": ["Alta", "Media", "Baja"],
            "nivel_motivación": ["Alto", "Medio", "Bajo"],
            "nee": ["Si", "No"],
            "enfermedades": ["Si", "No"],
            "medio_transporte": ["Vehículo privado", "Transporte escolar", "A pie", "Transporte público", "Bicicleta"],
        }

        self.mapeo_vocacional = {
            "Deportes": [
                "DEPORTISTA",
                "GIMNASTA",
                "FUTBOLISTA",
                "PATINADORA",
                "PATINADOR",
                "ATLETISMO",
                "JUGADORA DE PING PONG",
                "CICLISTA",
                "TENISTA",
                "KARATE PROFESIONAL",
                "ENTRENADORA DE VOLEIBOL",
                "FUTBOLISTA PROFESIONAL",
                "PILOTO DE CARRERAS",
                "EMPRESARIO Y FUTBOLISTA",
                "DOCTOR Y FUTBOLISTA",
                "VETERINARIA POSIBLEMENTE",
                "FUTBOLISTA O VETERINARIO",
                "FUTBOLISTA Y POLICIA",
            ],
            "Ciencias de la Salud": [
                "DOCTOR",
                "DOCTORA",
                "MEDICINA",
                "PEDIATRA",
                "PEDIATRIA",
                "MÉDICO",
                "MEDICO",
                "MEDICO CARDIOLOGO",
                "MEDICINA (TRAUMATOLOGÍA)",
                "ENFERMERA",
                "INVESTIGADORA",
                "MEDICINA Y ADM DE EMPRESAS",
                "MEDICA O INGENIERIA CIVIL",
            ],
            "STEM": [
                # Ingenierías
                "ING EN SISTEMAS",
                "INGENIERO",
                "INGENIERO DE SISTEMA",
                "INGENIERO FORESTAL",
                "INGENIERO DE SISTEMAS",
                "INGENIERO CIVIL",
                "INGENIERO CIVIL, MILITAR",
                "INGENIERO O ARQUITECTO",
                "INGENIERÍA DE SISTEMAS",
                "INGENIERIA",
                "INGENIERIA INDUSTRIAL",
                "INGENIERO MECATRÓNICO",
                "INGENIERIA MECANICA",
                "INGENIERIA MULTIMEDIA",
                # Programación y Tecnología
                "PROGRAMADOR",
                "PROGRAMACION DE SISTEMAS",
                # Ciencias
                "CIENTÍFICA",
                "CIENTIFICO",
                "CIENTÍFICO",
                "ECONOMISTA",
                "ASTRONAUTA",
                "VETERIANARIA",
                # Ciencias Biológicas y Afines
                "VETERINARIA",
                "VETERINARIO",
                "BIOLOGA MARINA",
                "BIOLOGIA MARINA",
                "GÉNETICA",
                "ZOOTECNISTA",
                "PALEONTÓLOGO",
                "PALEONTOLOGO",
                "BIÓLOGO",
                "ARQUEOLOGA MARINA",
                "ECOLOGISTA",
                # Otros STEM
                "ROBÓTICA Y BAILE",
            ],
            "Arte y Entretenimiento": [
                "FUTBOLISTA Y YOUTUBER",
                "ARTÍSTA MÚSICAL",
                "CANTANTE, POLICIA",
                "ACTRIZ O BAILARINA PROFESIONAL",
                "MUSICA",
                "YOUTUBER",
                "ACTOR",
                "MAGO",
                "ARTISTA",
                "BAILARINA",
                "CANTANTE",
                "VIOLINISTA",
                "PINTORA",
                "YOTUBER",
                "YOUTOBER",
                "MÚSICO",
                "MUSICO",
                "ACTRIZ",
                "MODELO",
                "ARTISTA MÚSICAL",
                "ARTÍSTA",
                "ARTÍSTA PLÁSTICA",
                "ARTES ESCENICAS",
                "ARTES VISULAES",
                "TATUADORA",
            ],
            "Sociales y Humanidades": [
                "ABOGADA/INGIERA/ MODELO",
                "DERECHO O PSICOLOGÍA",
                "PUBLICIDAD DIGITAL",
                "ABOGADO",
                "ABOGADA",
                "COMUNICADORA SOCIAL",
                "PUBLICISTA",
                "CONTADOR PÚBLICO",
                "DOCENTE",
                "PROFESORA",
                "PROFESOR",
                "DOCENTE DE DEPORTES",
                "ADMINISTRADO DE EMPRESAS",
                "NEGOCIOS INTERNACIONALES O DISEÑO GRAFICO",
                "NEGOCIOS IINTERNACIONALES",
                "PSICÓLOGA",
                "DERECHO",
                "CRIMINALISTICA",
                "CRIMINALISTA",
                "FISCAL DE MENORES",
                "IDIOMAS",
                "REPORTERA",
                "PSICILOGA",
                "",
            ],
            "Diseño y Arquitectura": [
                "DISEÑADORA DE MODAS O PSICOLÓGA",
                "DISEÑO INDUSTRIAL",
                "DISEÑO INDSUTRIAL",
                "DISEÑADOR INDUSTRIAL",
                "DISEÑADOR GRAFICO",
                "DISEÑADORA DE MODAS",
                "ARQUITECTO",
                "ARQUITECTA",
                "ARQUITECTCA",
                "DISEÑADOR GRÁFICO",
                "DISEÑADORA GRÁFICA",
                "DISEÑADORA DE INTERIORES",
                "DISEÑADORA DE INTERIORES, ARQUITECTA",
            ],
            "Negocios y Emprendimiento": [
                "EMPRESARIO",
                "EMPRESARIA",
                "EMPREARIA",
                "COMERCIANTE",
                "NEGOCIOS INTERNACIONALES",
                "GASTRONOMIA",
                "GASTRÓNOMO",
                "CHEF",
                "COCINERO",
                "INDEPENDIENTE",
                "EMPRENDEDOR",
            ],
            "Seguridad y Fuerzas Armadas": [
                "MILITAR",
                "BOMBERO",
                "POLICIA",
                "AZAFATA",
                "AUXILIAR DE VUELO",
                "PILOTO",
                "MILITAR O ASTROFÍSICO",
                "TAXISTA",
            ],
            "Indefinido o No Sabe": [
                "AUN NO SABE",
                "AÚN NO SABE",
                "NO SABE",
                "NO",
                "NO DEFINIDO",
                "NO SABE AÚN",
                "NO SABE AUN",
                "NO LO TIENE DEFINIDO",
                "NINGUNA",
                "NO SABES",
            ],
        }

    def categorize_family_convivence(self, valor):
        """
        Categoriza la convivencia familiar del estudiante en grupos simples.

        Args:
            valor: Puede ser un número (int), NaN, o un string con la descripción de la convivencia

        Returns:
            str: Categoría de convivencia
        """
        # Manejar valores NaN
        if pd.isna(valor):
            return None

        # Si es un número, usar el mapping numérico
        if isinstance(valor, (int, float)) and not pd.isna(valor):
            mapping = {"2": "Mamá y papá", "3": "Mamá, papá y otros", "4": "Mamá, papá y otros", "5": "Mamá, papá y otros", "6": "Mamá, papá y otros"}
            return mapping.get(int(valor), None)

        # Convertir a string y limpiar
        valor = str(valor).strip().lower()

        numeric_mapping = {
            "2": "Mamá y papá",
            "3": "Mamá, papá y otros",
            "4": "Mamá, papá y otros",
            "5": "Mamá, papá y otros",
            "6": "Mamá, papá y otros",
        }

        # Lista de palabras clave para cada categoría
        palabras_clave = {
            "mama": ["mamá", "mama", "madre"],
            "papa": ["papá", "papa", "padre"],
            "otros": [
                "hermano",
                "hermana",
                "hermanos",
                "abuelo",
                "abuela",
                "tío",
                "tia",
                "tia",
                "primo",
                "prima",
                "cuñado",
                "cuñada",
                "padrastro",
                "madrastra",
                "tios",
                "tías",
                "tias",
            ],
        }

        # Verificar presencia de palabras clave
        tiene_mama = any(palabra in valor for palabra in palabras_clave["mama"])
        tiene_papa = any(palabra in valor for palabra in palabras_clave["papa"])
        tiene_otros = any(palabra in valor for palabra in palabras_clave["otros"])

        # Determinar categoría
        if tiene_mama and tiene_papa and not tiene_otros:
            return "Mamá y papá"
        elif tiene_mama and not tiene_papa and not tiene_otros:
            return "Solo mamá"
        elif tiene_papa and not tiene_mama and not tiene_otros:
            return "Solo papá"
        elif tiene_mama and tiene_papa and tiene_otros:
            return "Mamá, papá y otros"
        elif tiene_otros:
            return "Otros familiares"
        elif valor in ["", "nan", "none", "null"]:
            return None
        elif str(valor) in numeric_mapping:
            return numeric_mapping.get(str(valor), None)
        else:
            print(f"Valor no categorizado: {valor}")
            print(f"Tipo de valor: {type(valor)}")
            return None

    def categorize_vocational_projection(self, valor):
        if pd.isnull(valor):
            return None
        valor_limpio = valor.strip()
        for categoria, profesiones in self.mapeo_vocacional.items():
            if valor_limpio in profesiones:
                return categoria
        if valor_limpio is not None:
            print(f"Valor no categorizado: {valor_limpio}")
        return None

    def categorize_extracurricular_activities(self, actividad):
        if pd.isnull(actividad):
            return None

        actividad = str(actividad).strip().lower()

        if any(p in actividad for p in ["niguna", "ninguna", "ningun", "no", "nninguna"]):
            return "Ninguna"

        deportes_ind = [
            "equitación",
            "campamento",
            "montar cicla",
            "actividades fisicas",
            "gimnasio",
            "patinae",
            "natación",
            "natacion",
            "tenis",
            "ciclismo",
            "atletismo",
            "karate",
            "taekwondo",
            "pilates",
            "golf",
            "motocross",
            "patinaje",
            "gimnasia",
            "bicicleta",
            "tennis",
        ]
        deportes_eq = ["fútbol", "futbol", "voley", "voleibol", "volleyball", "baloncesto", "tenis de mesa", "básquetbol"]
        artes = ["música", "musica", "danza", "baile", "violin", "piano", "dibujo", "artes"]
        idiomas = ["inglés", "ingles", "francés", "idiomas", "duolingo"]
        tecnologia = ["diseño digital"]

        categorias = set()

        if any(d in actividad for d in deportes_ind):
            categorias.add("Deporte")
        if any(d in actividad for d in deportes_eq):
            categorias.add("Deporte")
        if any(a in actividad for a in artes):
            categorias.add("Artes")
        if any(i in actividad for i in idiomas):
            categorias.add("Idiomas")
        if any(t in actividad for t in tecnologia):
            categorias.add("Tecnología / Diseño")

        if not categorias:
            print(f"Actividad no categorizada: '{actividad}'")  # Esto te ayudará a identificar problemas.
            return None
        else:
            return ", ".join(sorted(categorias))

    def handle_enfermedades(self, valor):
        if pd.isnull(valor):
            return None
        valor = str(valor).strip().lower()
        valor = self.proper_case(valor)
        # Reemplazar SÍ por Si
        if valor == "Sí":
            return "Si"
        # Si el valor es diferente a Si o No, entonces es Si
        if valor not in ["No"]:
            return "Si"
        # Aplicar Proper Case en todos los valores finales
        return valor

    @staticmethod
    def proper_case(value):
        """Convierte texto a Proper Case (primera letra mayúscula, resto minúsculas)."""
        if isinstance(value, str):
            return value.capitalize()
        return value

    def clean_column(self, col_name):
        """
        Limpia una columna: elimina espacios, estandariza valores y realiza reemplazos específicos.
        """

        # Quitar espacios y pasar a mayúscula
        self.df[col_name] = self.df[col_name].str.strip().str.upper()

        if col_name == "familia":
            self.df[col_name] = self.df[col_name].apply(self.categorize_family_convivence)
            return

        if col_name == "proyección_vocacional":
            self.df[col_name] = self.df[col_name].apply(self.categorize_vocational_projection)
            return

        if col_name == "actividades_extracurriculares":
            self.df[col_name] = self.df[col_name].apply(self.categorize_extracurricular_activities)
            return

        if col_name == "enfermedades":
            self.df[col_name] = self.df[col_name].apply(self.handle_enfermedades)
            return

        # Aplicar Proper Case en todos los valores finales
        self.df[col_name] = self.df[col_name].apply(self.proper_case)

        # Primero validar contra allowed_values, luego verificar en replacements
        allowed = self.allowed_values.get(col_name, [])
        replacements_dict = self.replacements.get(col_name, {})
        new_values = []

        for val in self.df[col_name]:
            if pd.isnull(val):
                new_values.append(None)
            elif val in allowed:
                new_values.append(val)
            elif val in replacements_dict:
                new_val = replacements_dict[val]
                if new_val in allowed:
                    new_values.append(new_val)
                else:
                    print(f"Valor reemplazado '{new_val}' tampoco está permitido en '{col_name}'")
                    new_values.append(None)
            else:
                print(f"Valor no permitido ni reemplazado encontrado en '{col_name}': '{val}'")
                new_values.append(None)

        self.df[col_name] = new_values

    def clean_all(self):
        """Limpia todas las columnas especificadas."""
        for col in set(self.replacements.keys()).union(self.allowed_values.keys()):
            self.clean_column(col)
        return self.df

    def create_df_unique_values(self, cols_to_search):
        """Crea un dataframe con los valores únicos de cada columna."""
        df_unique_values = {col: self.df[col].dropna().unique() for col in cols_to_search}
        df_unique = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in df_unique_values.items()]))
        return df_unique_values, df_unique

    def display_unique_values(self, cols_to_search):
        """Muestra los valores únicos y su cantidad en cada columna."""
        print("Valores únicos y su cantidad en cada columna:")
        column_unique_counts = {}
        for col in cols_to_search:
            unique_values = self.df[col].dropna().unique()
            column_unique_counts[col] = len(unique_values)
            if len(unique_values) < self.df.shape[0]:
                print(f"{col}: {len(unique_values)} valores únicos")

    def unique_values_by_column(self, col):
        """Muestra los valores únicos de una columna."""
        df_unique_values, df_unique = self.create_df_unique_values([col])
        return df_unique_values[col]


# Ejemplo de uso:
# cleaner = StudentsDataCleaner(df)
# df_limpio = cleaner.clean_all()
