# Usa la imagen oficial de Ubuntu 22.04 como base
FROM ubuntu:22.04

# Actualiza el sistema y asegúrate de que está actualizado
RUN apt-get update && apt-get upgrade -y

# Instala Python 3.11 y pip
RUN apt-get install -y python3.11 python3-pip git 

# Copia tu archivo de requirements.txt en el contenedor
COPY requirements.txt requirements.txt 

# Instala las dependencias de Python desde el archivo de requirements
RUN pip3 install --no-cache-dir -r requirements.txt

