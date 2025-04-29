import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class EncodingHandler:
    def __init__(self):
        pass

    def label_encode(self, df, column):
        """
        Aplica codificación ordinal (Label Encoding) a una columna categórica.
        """
        df = df.copy()
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        return df

    def one_hot_encode(self, df, column, drop_first=False):
        """
        Aplica codificación One-Hot a una columna categórica.
        """
        df = df.copy()
        dummies = pd.get_dummies(df[column], prefix=column, drop_first=drop_first)
        df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
        return df

    def binary_encode_multilabel(self, df, column, delimiter=","):
        """
        Codifica columnas con múltiples valores separados por comas como variables binarias por categoría.
        """
        df = df.copy()
        df[column] = df[column].fillna("")
        multilabels = df[column].str.get_dummies(sep=delimiter)
        multilabels.columns = [f"{column}_{c.strip()}" for c in multilabels.columns]
        df = pd.concat([df.drop(columns=[column]), multilabels], axis=1)
        return df

    def frequency_encode(self, df, column):
        """
        Codifica una columna categórica con la frecuencia de cada categoría.
        """
        df = df.copy()
        freq = df[column].value_counts() / len(df)
        df[column + '_freq'] = df[column].map(freq)
        return df.drop(columns=[column])

    def weight_encode(self, df, column, mapping):
        """
        Codifica una columna usando un diccionario con pesos específicos por categoría.
        """
        df = df.copy()
        df[column + '_peso'] = df[column].map(mapping)
        return df

    def binary_encode_by_value(self, df, column, valor, nombre_columna=None):
        """
        Crea una columna binaria 1/0: 1 si la columna coincide con el valor dado, 0 si no.
        """
        df = df.copy()
        if nombre_columna is None:
            nombre_columna = f"{column}_{valor}".lower().replace(" ", "_")

        df[nombre_columna] = (df[column] == valor).astype(int)
        return df