import logging
from llama_cpp import Llama
import json
import os

def guardar_en_json(nombre_archivo, datos):
    """
    Guarda datos en un archivo JSON con una lista de diccionarios con las claves 'Titulo' y 'Resultado'.
    
    :param nombre_archivo: Nombre del archivo JSON de salida.
    :param datos: Una lista de tuplas donde cada tupla contiene el título y el resultado.
    """
    # Convertir los datos a una lista de diccionarios
    datos_json = [{'Titulo': titulo, 'Resultado': resultado} for titulo, resultado in datos]
    
    # Abrir el archivo para escritura en modo texto
    with open(nombre_archivo, 'w') as archivo_json:
        # Guardar los datos en formato JSON
        json.dump(datos_json, archivo_json, indent=4)


def cargar_datos_json(ruta_archivo):
    """Carga y devuelve los datos de un archivo JSON."""
    with open(ruta_archivo) as archivo:
        datos = json.load(archivo)
    return datos

def obtener_datos_filtrados_y_ground_truth(ruta_archivo, claves_excluidas=('ground_truth', 'Título')):
    """Extrae los datos del JSON y devuelve junto con el ground_truth, excluyendo claves no deseadas."""
    datos = cargar_datos_json(ruta_archivo)
    datos_filtrados = {clave: datos[clave] for clave in datos if clave not in claves_excluidas}
    ground_truth = datos.get('ground_truth')
    return datos_filtrados, ground_truth

def listar_directorios(ruta):
    """Devuelve una lista de nombres de directorios dentro de la ruta dada si existe."""
    if not os.path.isdir(ruta):
        print("La ruta especificada no existe o no es un directorio válido.")
        return []
    return [nombre for nombre in os.listdir(ruta) if os.path.isdir(os.path.join(ruta, nombre))]

def listar_archivos(ruta):
    """Devuelve una lista de rutas de archivos dentro del directorio dado."""
    if not os.path.isdir(ruta):
        print("La ruta especificada no existe o no es un directorio válido.")
        return []
    return [os.path.join(ruta, nombre) for nombre in os.listdir(ruta) if os.path.isfile(os.path.join(ruta, nombre))]

def crear_prompt_clasificacion(nombres_clases, ejemplos, texto_a_clasificar):
    """Crea un prompt de texto para la clasificación de texto."""
    lista_clases = ', '.join(nombres_clases) if isinstance(nombres_clases, (list, tuple)) else nombres_clases
    ejemplos_formateados = "\n\n".join([f" Texto: '{texto}'\n Clasificación: {etiqueta}" for texto, etiqueta in ejemplos])
    
    return (
        f"Clasifica el texto en esta clase: {lista_clases}. "
        f"Responde solo con una palabra: {lista_clases}.\n\n"
        f"Ejemplos:\n{ejemplos_formateados}\n\n"
        f"Texto: '{texto_a_clasificar}'\n"
        f"Clasificación: "
    )

def obtener_todos_los_datos_archivo(ruta_base, maximo=-1):
    """Reúne todos los datos de los archivos y los prompts de los directorios bajo la ruta base."""
    directorios = listar_directorios(ruta_base)
    archivos_por_clase = {}
    prompts_por_clase = {}
    todos_prompts = []
    todos_archivos = []

    for nombre_clase in directorios or ['test']:
        archivos = listar_archivos(os.path.join(ruta_base, nombre_clase))[:maximo]
        archivos_por_clase[nombre_clase] = archivos
        prompts = [obtener_datos_filtrados_y_ground_truth(archivo) for archivo in archivos]
        prompts_por_clase[nombre_clase] = prompts
        todos_prompts.extend(prompts)
        todos_archivos.extend(archivos)

    return archivos_por_clase, prompts_por_clase, todos_prompts, todos_archivos


 
def promting(llm,prompt,logging):
    # Genera la respuesta
    output = llm(prompt)

    # Registra el prompt y la respuesta en el archivo de registro
    logging.info("Prompt: %s", prompt)
    logging.info("Respuesta: %s", output['choices'])
    logging.info("ALL_INFO: %s", output)
    print(output)
    return output['choices'][0]['text']

# Configura el archivo de registro
log_filename = "llama_promt_log.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Inicializa Llama y guarda la información del modelo en el registro
model_path = "./llama_models/llama-2-7b.Q4_K_M.gguf"
logging.info("Modelo: %s", model_path)
llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=30)

class_name = listar_directorios('./data/splits/train')
train,promt_train,b,d = obtener_todos_los_datos_archivo('./data/splits/train_json',1)
test,promt_test,a,c = obtener_todos_los_datos_archivo('./data/splits/test',100)

prompts=[crear_prompt_clasificacion(class_name,b[:3],text_class )for text_class in a]

datos_a_guardar = []

for idx,prompt in enumerate(prompts):
    result=promting(llm,prompt,logging)
    datos_a_guardar.append((c[idx],result))

guardar_en_json('resultados.json', datos_a_guardar)   

# Cierra el archivo de registro
logging.shutdown()
