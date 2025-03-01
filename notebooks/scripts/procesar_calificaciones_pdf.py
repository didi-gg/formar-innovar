import os
import pdfplumber
import pandas as pd
import logging
import re

# Configuración del logging
logging.basicConfig(filename='errores_procesamiento.log', level=logging.ERROR, 
                    format='%(asctime)s %(levelname)s: %(message)s')

# Array de asignaturas
ASIGNATURAS = [
    "Ciencias Naturales", "Ciencias Sociales", "Matemáticas", "Lengua Castellana", "Inglés", "Innovación y Emprendimiento",
    "Aprendizaje Basado en Investigación", "Educación Física", "Lectura Crítica", "Artes", "Robótica y TICS",
    "Inteligencia emocional e Integralidad", "Creatividad e Innovación", "Aprendizaje Basado en Proyectos",
    "Ciencias Políticas y Económicas", "Filosofía", "Trigonometría", "Francés", "Educación Ambiental", "Proyecto de inversión",
    "Pre Naturales", "Pre Sociales", "Lectoescritura", "Aprendizaje Basado en Retos", "Lectoescritura 6", "Pre Sociales 6",
    "Ciencias Naturales Integradas", "Ciencias Políticas y Económicas", "Trigonometría", "Lengua Castellana","Lectura Crítica","English",
    "Français", "Filosofía", "Educación Física y Deportes", "Plan de inversión", "Artes", "Tecnologías Informáticas", "Integralidad", "Inteligencia emocional",
    "Aprendizaje Basado en Proyectos", "Creatividad e innovación", "Lectura Crítica", "Formación Integral"
]

# Definir las columnas del DataFrame
COLUMNS = ["Sede", "Estudiante", "Documento de identidad", "Grado", "Grupo", "Periodo", "Año", "Intensidad Horaria", "Asignatura", "Cognitiva", "Procedimental", "Actitudinal", "Axiologica"]

GRADOS = ["Transición", "Jardín", "Pre Jardín", "Párvulos", '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']

class ProcesarCalificacionesPDF:

    def __init__(self):
        self.asignaturas = ASIGNATURAS
        self.grados = GRADOS

    def extraer_asignatura(self, linea, pdf_path):
        for asignatura in self.asignaturas:
            if asignatura in linea:
                logging.debug(f"Asignatura '{asignatura}' encontrada en la línea.")
                return asignatura

        # Verificar si la línea contiene algún número
        if any(char.isdigit() for char in linea):
            pdf_link = f"file://{os.path.abspath(pdf_path).replace(' ', '%20')}"
            # Si contiene números pero no se encuentra la asignatura, registrar en el log
            logging.error(f"Asignatura no encontrada en la línea: {linea} - Archivo: {pdf_link}")
        
        # Devolver "Asignatura no encontrada" si no se encuentra la asignatura
        return "Asignatura no encontrada"

    def es_linea_de_parada(self, linea):
        patron = r'\b(Bajo|Básico|Alto|Superior)\b.*\d{1,2}\s*-\s*\d{1,2}'
        if re.search(patron, linea):
            logging.debug(f"Línea de parada encontrada: {linea}")
            return True
        return False

    def procesar_datos_preescolar(self, pdf_path, estudiante, doc_id, lines, grado, grupo, periodo, año, student_data_line, sede):
        data = []
        asignaturas_start_index = lines.index(student_data_line) + 4
        
        for i in range(asignaturas_start_index, len(lines)):
            line = lines[i]
            if self.es_linea_de_parada(line):
                break

            axiologica = ''
            intensidad_horaria = ''

            asignatura_data = line.split()
            asignatura = self.extraer_asignatura(line, pdf_path)
            
            if (asignatura in ['Asignatura no encontrada', 'Inteligencia emocional', 'Formación Integral']):
                continue

            cognitiva = asignatura_data[-5] if len(asignatura_data) > 4 and asignatura_data[-5].isdigit() else None
            procedimental = asignatura_data[-4] if len(asignatura_data) > 3 and asignatura_data[-4].isdigit() else None
            actitudinal = asignatura_data[-2] if len(asignatura_data) > 1 and asignatura_data[-2].isdigit() else None

            # Verificar si todos los valores son válidos (numéricos)
            if None not in [cognitiva, procedimental, actitudinal]:
                data.append([
                    sede,
                    estudiante,
                    doc_id,
                    grado,
                    grupo,
                    periodo,
                    año,
                    intensidad_horaria,
                    asignatura,
                    cognitiva,
                    procedimental,
                    actitudinal,
                    axiologica
                ])
            else:
                pdf_link = f"file://{os.path.abspath(pdf_path).replace(' ', '%20')}"
                logging.error(f"Asignatura:{asignatura} Valores no válidos encontrados en la línea: {line} en el archivo {pdf_link}")
        return data

    def procesar_datos_primaria_secundaria(self, pdf_path, estudiante, doc_id, lines, grado, grupo, periodo, año, student_data_line, sede):
        data = []
        asignaturas_start_index = lines.index(student_data_line) + 5
        pdf_link = f"file://{os.path.abspath(pdf_path).replace(' ', '%20')}"

        for i in range(asignaturas_start_index, len(lines)):
            line = lines[i]
            if self.es_linea_de_parada(line):
                break

            asignatura_data = line.split()
            if len(asignatura_data) < 6:
                logging.warning(f"Línea inesperada: {line}")
                continue

            asignatura = self.extraer_asignatura(line, pdf_path)

            if (asignatura in ['Asignatura no encontrada', 'Inteligencia emocional', 'Formación Integral']):
                continue

            try:
                cognitiva = asignatura_data[-4] if asignatura_data[-4].isdigit() or asignatura_data[-4] == "" else None
                procedimental = asignatura_data[-3] if asignatura_data[-3].isdigit() or asignatura_data[-3] == "" else None
                actitudinal = asignatura_data[-2] if asignatura_data[-2].isdigit() or asignatura_data[-2] == "" else None
                axiologica = asignatura_data[-1] if asignatura_data[-1].isdigit() or asignatura_data[-1] == "" else None
                intensidad_horaria = asignatura_data[-6] if asignatura_data[-6].isdigit() or asignatura_data[-6] == "" else None

                # Si alguno de los valores es None, no se procesa la línea y se registra en el log
                if None in [cognitiva, procedimental, actitudinal, axiologica, intensidad_horaria]:
                    logging.error(f"Asignatura:{asignatura} Valores no válidos encontrados en la línea: {line} en el archivo {pdf_link}")
                else:
                    data.append([
                        sede,
                        estudiante,
                        doc_id,
                        grado,
                        grupo,
                        periodo,
                        año,
                        intensidad_horaria,
                        asignatura,
                        cognitiva,
                        procedimental,
                        actitudinal,
                        axiologica
                    ])
            except IndexError:
                logging.error(f"Error de índice al procesar la línea: {line} en el archivo {pdf_link}")
        return data

    def procesar_pdf(self, pdf_path, sede):
        data = []
        pdf_link = f"file://{os.path.abspath(pdf_path).replace(' ', '%20')}"
        try:
            with pdfplumber.open(pdf_path) as pdf:
                first_page = pdf.pages[0]
                text = first_page.extract_text()
                if not text:
                    logging.error(f"No se pudo extraer texto del archivo {pdf_link}")
                    return data

                lines = text.splitlines()
                student_info_line = next((line for line in lines if "Documento Identidad" in line), None)
                student_data_line = next((line for line in lines if "Asignatura Docente" in line), None) or \
                                    next((line for line in lines if "Asignatura Nivel" in line), None)

                if not student_info_line or not student_data_line:
                    logging.error(f"No se encontraron líneas necesarias en el archivo {pdf_link}")
                    return data

                student_data = lines[lines.index(student_info_line) + 1].split()
                periodo = student_data[-2]
                año = student_data[-1]
                
                grupo = student_data[-3]
                
                if not student_data[-5].isdigit():
                    doc_id = student_data[-4]
                    estudiante = ' '.join(student_data[:-4])
                else:
                    doc_id = student_data[-5]
                    estudiante = ' '.join(student_data[:-5])

                if student_data[-4] == "Pre":
                    grado = ' '.join(student_data[-4:-2])
                if student_data[-3] in self.grados:
                    grupo = "A"
                    grado = student_data[-3]
                else:
                    grado = student_data[-4]
                    
                if not doc_id.isdigit():
                    logging.error(f"Documento de identidad no válido en el archivo {pdf_link}")
                    return data

                logging.info(f"Procesando estudiante {estudiante} - {doc_id} - Grado: {grado} - Grupo: {grupo} - Periodo: {periodo} - Año: {año}")

                if "Asignatura Nivel" in student_data_line:
                    data = self.procesar_datos_preescolar(pdf_path, estudiante, doc_id, lines, grado, grupo, periodo, año, student_data_line, sede)
                else:
                    data = self.procesar_datos_primaria_secundaria(pdf_path, estudiante, doc_id, lines, grado, grupo, periodo, año, student_data_line, sede)

        except Exception as e:
            logging.error(f"Error procesando el archivo {pdf_link}: {e}")
        return data

    def procesar_carpeta(self, carpeta_path):
        all_data = []
        for archivo in os.listdir(carpeta_path):
            if archivo.endswith(".pdf"):
                sede = "Fusagasugá" if archivo.startswith("Fusa") else "Girardot"
                pdf_path = os.path.join(carpeta_path, archivo)
                data = self.procesar_pdf(pdf_path, sede)
                if data:
                    all_data.extend(data)

        if all_data:
            df = pd.DataFrame(all_data, columns=COLUMNS)
            csv_path = os.path.join(carpeta_path, 'resultado.csv')
            df.to_csv(csv_path, index=False)
            logging.info(f"CSV guardado en {csv_path}")
        else:
            logging.info("No se encontraron datos para procesar.")


if __name__ == '__main__':
    carpeta_pdf = os.path.abspath('data/calificaciones')
    procesador = ProcesarCalificacionesPDF()
    procesador.procesar_carpeta(carpeta_pdf)

    #pdf_file_path = ''
    #procesador.procesar_pdf(pdf_file_path, 'Fusagasugá')