import logging
from llama_cpp import Llama
import json
import os
import csv


def guardar_en_csv(nombre_archivo, datos):
    """
    Guarda datos en un archivo CSV con dos columnas: 'Titulo' y 'Resultado'.
    
    :param nombre_archivo: Nombre del archivo CSV de salida.
    :param datos: Una lista de tuplas donde cada tupla contiene el título y el resultado.
    """
    with open(nombre_archivo, 'w', newline='') as archivo_csv:
        escritor = csv.writer(archivo_csv)
        
        # Escribe la cabecera con los nombres de las columnas
        escritor.writerow(['Titulo', 'Resultado'])
        
        # Escribe los datos en el archivo CSV
        for dato in datos:
            escritor.writerow(dato)

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



# ### System: You possess expertise in web page classification
# ### User:
def get_file(path_file):
    with open(path_file) as f:
        data = json.load(f)
    # print(data.keys())
    return ('\n').join([f"{data[key].split('Content-length')}" for key in data.keys() if key != 'ground_truth' and key != 'Título']),data['ground_truth']

def get_file_2(path_file):  
  #print(path_file)
  with open(path_file) as f:
      data = json.load(f)
  #print(data.keys())
  return [f"{key} : {data[key]}" for key in data.keys() if key != 'ground_truth' ],data['ground_truth']
  

def get_class(train_path):
#   print(train_path)
  if os.path.exists(train_path) and os.path.isdir(train_path):
    carpetas = [nombre for nombre in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, nombre))]
    return carpetas
  else:
    print("El directorio especificado no existe o no es un directorio válido.")


def get_archives(train_path):
  if os.path.exists(train_path) and os.path.isdir(train_path):
    carpetas = [os.path.join(train_path,nombre) for nombre in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, nombre))]
    return carpetas
  else:
    print("El directorio especificado no existe o no es un directorio válido.")

def creating_promt(class_name,text_traindata,text_classification):
  if isinstance(class_name, (list, tuple)):
      class_name = (', ').join(class_name)
  text_traindata = ('\n\n').join([f" Text:'{text}' \n Classication: {label}" for text,label in text_traindata])
  return f" Classify the text in this class : {class_name}. Reply with only one word:  {class_name}. \n\
  Examples: \n\
  {text_traindata} \n\n\
  Text: '{text_classification}' \n\
  Classication: "

def get_all_files(path_data,max_cont=-1):
    #'./data/splits/train_json'
    classes=get_class(path_data)
    archiver_per_clases={}
    prompt_per_clases={}
    promt_all_reson=[]
    archiver_all_reson=[]
    if len(classes)>0:
      for class_name in classes:
        #print(get_archives(os.path.join(path_data,i)))
        archiver_per_clases[class_name]=get_archives(os.path.join(path_data,class_name))[0:max_cont]
        prompt_per_clases[class_name]=[get_file_2(archive) for archive in  archiver_per_clases[class_name]]
        promt_all_reson.extend(prompt_per_clases[class_name])
        archiver_all_reson.extend(archiver_per_clases[class_name])

        # print(len(archiver_per_clases[class_name]),promt_all_reson)
    else:
       archiver_per_clases['test']=get_archives(os.path.join(path_data,class_name))
       prompt_per_clases['test']=[get_file_2(archive) for archive  in  archiver_per_clases['test']]
    return archiver_per_clases,prompt_per_clases,promt_all_reson,archiver_all_reson

 
def promting(llm,prompt,logging):
    # Genera la respuesta
    output = llm(prompt)
    #, max_tokens=32, stop=["Q:", "\n"], echo=True)

    # Registra el prompt y la respuesta en el archivo de registro
    logging.info("Prompt: %s", prompt)
    logging.info("Respuesta: %s", output['choices'])
    logging.info("ALL_INFO: %s", output)
    # print(output['choices'])
    # print(output['choices'][0]['text'])
    return output['choices'][0]['text']

# Configura el archivo de registro
log_filename = "llama_promt_log.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Inicializa Llama y guarda la información del modelo en el registro
model_path = "./llama_models/llama-2-7b.Q4_K_M.gguf"
logging.info("Modelo: %s", model_path)
llm = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=30)

class_name=get_class('./data/splits/train')
train,promt_train,b,d=get_all_files('./data/splits/train_json',1)
test,promt_test,a,c=get_all_files('./data/splits/test',10)

prompts=[creating_promt(class_name,b[:3],text_class )for text_class in a]

datos_a_guardar = []

for idx,prompt in enumerate(prompts):
    result=promting(llm,prompt,logging)
    datos_a_guardar.append((c[idx],result))

print("\n\n\n\n\n\n\n\n\n")
print(datos_a_guardar)
print("\n\n\n\n\n\n\n\n\n")
guardar_en_json('resultados.json', datos_a_guardar)   

# Cierra el archivo de registro
logging.shutdown()
