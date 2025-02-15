import hashlib
import os


class DataTransformation:
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
