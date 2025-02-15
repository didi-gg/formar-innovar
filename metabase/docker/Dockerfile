FROM openjdk:19-buster

# Directorio para plugins
ENV MB_PLUGINS_DIR=/home/plugins/

# Agregar el archivo JAR de Metabase y el plugin de DuckDB
ADD https://downloads.metabase.com/v0.46.2/metabase.jar /home
ADD ./duckdb.metabase-driver.jar /home/plugins/

# Permisos para el plugin
RUN chmod 744 /home/plugins/duckdb.metabase-driver.jar

# Comando para ejecutar Metabase
CMD ["java", "-jar", "/home/metabase.jar"]