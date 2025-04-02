import pandas as pd
from sklearn.impute import KNNImputer

class ImputerHandler:
    def __init__(self):
        pass

    def impute_missing_values(self, dataframe, columna, metodo="moda", k=5):
        """
        Imputa los valores faltantes de una columna según el método seleccionado.

        Parametros:
            dataframe (pd.DataFrame): DataFrame a procesar.
            columna (str): nombre de la columna a imputar.
            metodo (str): "moda", "media", "mediana", "knn" o "constante".
            k (int): número de vecinos para KNN (solo si metodo="knn").

        Retorna:
            DataFrame con la columna imputada.
        """
        df = dataframe.copy()

        if columna not in df.columns:
            raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

        if metodo == "moda":
            valor = df[columna].mode().iloc[0]
            df[columna] = df[columna].fillna(df[columna].mode()[0])

        elif metodo == "media":
            valor = df[columna].mean()
            df[columna] = df[columna].fillna(df[columna].mean())

        elif metodo == "mediana":
            valor = df[columna].median()
            df[columna] = df[columna].fillna(df[columna].median())

        elif metodo == "constante":
            valor = input(f"Ingrese el valor constante para imputar en '{columna}': ")
            df[columna] = df[columna].fillna(valor)

        elif metodo == "knn":
            if df[columna].dtype == 'O':
                raise ValueError("KNN solo se puede aplicar a columnas numéricas.")
            imputer = KNNImputer(n_neighbors=k)
            df[[columna]] = imputer.fit_transform(df[[columna]])

        else:
            raise ValueError(f"Método '{metodo}' no reconocido. Usa: moda, media, mediana, constante o knn.")
        return df
    
    def handle_outliers(self, dataframe, columna, metodo="winsorizar"):
        """
        Maneja valores atípicos en una columna numérica según el método especificado.
        Opciones: "winsorizar", "mediana", "media", "eliminar"
        """
        df = dataframe.copy()

        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        IQR = Q3 - Q1

        limite_inf = Q1 - 1.5 * IQR
        limite_sup = Q3 + 1.5 * IQR

        if metodo == "winsorizar":
            df[columna] = df[columna].apply(lambda x: limite_inf if x < limite_inf else limite_sup if x > limite_sup else x)

        elif metodo == "mediana":
            mediana = df[columna].median()
            df[columna] = df[columna].apply(lambda x: mediana if x < limite_inf or x > limite_sup else x)

        elif metodo == "media":
            media = df[columna].mean()
            df[columna] = df[columna].apply(lambda x: media if x < limite_inf or x > limite_sup else x)

        elif metodo == "eliminar":
            df = df[(df[columna] >= limite_inf) & (df[columna] <= limite_sup)]

        else:
            raise ValueError("Método no válido. Usa: 'winsorizar', 'mediana', 'media', 'eliminar'")

        return df

# Ejemplo de uso:
# handler = ImputerHandler()
# df_imputado = handler.imputar(df, "edad", metodo="media")
