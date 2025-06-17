import hashlib
import os
import hmac


class HashUtility:
    """
    Clase para realizar transformaciones auxiliares de datos.
    """

    @staticmethod
    def hash_with_salt(value):
        """
        Aplica SHA-256 con un salt aleatorio al identificador del estudiante.

        :param value: String del ID del estudiante
        :return: Hash con el salt almacenado en formato hexadecimal
        """
        if value is None or value == "":
            return None  # Evita errores en valores vacíos

        salt = os.urandom(16)  # Genera un salt aleatorio de 16 bytes
        hash_obj = hashlib.sha256(salt + value.encode())

        return salt.hex() + hash_obj.hexdigest()  # Concatenar salt y hash en hexadecimal

    @staticmethod
    def hash_stable(value):
        """
        Aplica HMAC-SHA-256 con una clave secreta para producir un hash determinístico.

        :param value: El ID del estudiante (cadena)
        :return: Hash en formato hexadecimal
        """
        if not value:
            return None

        # Clave secreta almacenada en un archivo de entorno
        SECRET_KEY = os.getenv("SECRET_KEY_DOC_ID").encode()

        value_str = str(value).encode()
        return hmac.new(SECRET_KEY, value_str, hashlib.sha256).hexdigest()
