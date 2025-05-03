import pandas as pd
from sklearn.preprocessing import LabelEncoder

class NotesDataCleaner:
    def __init__(self, dataframe):
        self.df = dataframe

        # Diccionarios de reemplazo solo para valores mal escritos
        self.replacements = {
            "Periodo": {"I": "1", "II": "2", "III": "3", "IV": "4"},
        }

    def load_and_clean_data(self):
        '''
        Realiza limpieza básica:
        - Limpia y estandariza nombres de columnas
        - Convierte columnas de texto a mayúsculas
        - Convierte columnas numéricas a tipo numérico
        '''
        df = self.df  # uso del dataframe guardado en la instancia

        # Limpiar y estandarizar nombres de columnas
        df.columns = df.columns.astype(str).str.strip().str.replace(' ', '_')

        # Columnas de texto que deben convertirse a string y pasar a mayúsculas
        columnas_texto = ['Sede','Estudiante', 'id','Asignatura', 'Docente', 'Nivel']
        for col in columnas_texto:
            if col in df.columns:
                df[col] = df[col].astype(str).str.upper().str.strip()

        # Columnas numericas que deben convertirse a object
        columnas_object = ['Grado','Año','Grupo', 'Periodo']
        for col in columnas_object:
            if col in df.columns:
                df[col] = df[col].astype(object)

        # Columnas numéricas que deben mantenerse como números
        columnas_numericas = [
             'Intensidad_Horaria',
            'Cog', 'Proc', 'Act', 'Axi', 'Resultado'
        ]
        for col in columnas_numericas:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        self.df = df  # actualiza el atributo con el dataframe limpio
        return df
    
    def codificar_periodos(self, df):
        """
        Codifica la columna 'periodo' usando LabelEncoder, y le suma 1 para empezar desde 1.
        """
        if 'Periodo' in df.columns:
            le = LabelEncoder()
            df['Periodo'] = df['Periodo'].astype(str).str.upper().str.strip()
            df['Periodo'] = le.fit_transform(df['Periodo']) + 1
            df['Periodo'] = df['Periodo'].astype(object)

        return df