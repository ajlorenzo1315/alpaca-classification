import logging
from llama_cpp import Llama
import json
from tqdm import tqdm
import os
import json
from datetime import datetime

def guardar_en_json(nombre_archivo, datos):
    cabecera = ['Titulo', 'Resultado', 'Intentos']

    for dato in datos:
        datos = dict(zip(cabecera, dato))

    with open(nombre_archivo, 'a', encoding='utf-8') as archivo_json:
        archivo_json.write(json.dumps(datos) + ",\n")


def get_file(path_file):  
  with open(path_file, 'r', encoding='utf-8') as f:
      data = json.load(f)
  return [f"{key} : {data[key]}" for key in data.keys() if key != 'ground_truth' ],data['ground_truth']
  

def get_class(train_path):
  print(train_path)
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
  class_name=(', ').join(class_name)
  text_traindata = ('\n\n').join([f" Text:'{text}' \n Classication: {label}" for text,label in text_traindata])

  return f"Classify the text given in one of this classes: {class_name}. Reply with only the name of the class. \n\
  Examples: \n\
  {text_traindata} \n\n\
  Text: '{text_classification}' \n\
  Classication: "

# def creating_promt_3(class_name,text_traindata,text_classification):
#   class_name=(', ').join(class_name)
#   text_traindata = ('\n\n').join([f" Text:'{text}' \n Classication: {label}" for text,label in text_traindata])
#   return f"<s>[INST] <<SYS>> You are a helpful assistant. <</SYS>> Classify the text in this class : [{class_name}]. Reply with only one of these words: [{class_name}. \n\
#   Text: '{text}' \n\
#   Classication: \n\
#   [/INST] "

def get_all_files(path_data,max_cont=-1):
    classes=get_class(path_data)
    archiver_per_clases={}
    prompt_per_clases={}
    promt_all_reson=[]
    archiver_all_reson=[]
    if len(classes)>0:
      for class_name in classes:
        archiver_per_clases[class_name]=get_archives(os.path.join(path_data,class_name))[0:max_cont]
        prompt_per_clases[class_name]=[get_file(archive) for archive in  archiver_per_clases[class_name]]
        promt_all_reson.extend(prompt_per_clases[class_name])
        archiver_all_reson.extend(archiver_per_clases[class_name])
    else:
       archiver_per_clases['test']=get_archives(os.path.join(path_data,class_name))
       prompt_per_clases['test']=[get_file(archive) for archive  in  archiver_per_clases['test']]

    return archiver_per_clases,prompt_per_clases,promt_all_reson,archiver_all_reson

 
def promting(llm,prompt,logging):
    # Genera la respuesta
    output = llm(prompt)
    #, max_tokens=32, stop=["Q:", "\n"], echo=True)

    # Registra el prompt y la respuesta en el archivo de registro
    logging.info("Prompt: %s", prompt)
    logging.info("Respuesta: %s", output['choices'])
    logging.info("ALL_INFO: %s", output)
    return output['choices'][0]['text']


# Configura el archivo de registro
log_filename = "llama_promt_log.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Inicializa Llama
# Inicializa Llama y guarda la información del modelo en el registro
model_path = "./llama_models/llama-2-7b.Q4_K_M.gguf"
logging.info("Modelo: %s", model_path)


llm = Llama(model_path=model_path, n_ctx=4096)

llm_gpu = ()


class_name=get_class('./data/splits/train')

train,promt_train,b,d=get_all_files('./data/splits/train_new',1)
test,promt_test,a,c=get_all_files('./data/splits/test_new',-1)

print(class_name,train.keys())
prompts=[(name,creating_promt(class_name,b,text_class[0] )) for name,text_class in zip(c,a)]

datos_a_guardar = []


fecha_actual = datetime.now()
sufijo_fecha = fecha_actual.strftime("%Y-%m-%d_%H-%M-%S")
nombre_archivo_con_fecha = f"resultados_{sufijo_fecha}.json"

# for prompt in tqdm(prompts,desc="promting"):
#     name,prompt=prompt
#     logging.info("Prompt: %s", prompt)
#     result=promting(llm,prompt,logging)
#     datos_a_guardar.append((name,result))
#     guardar_en_json(nombre_archivo_con_fecha, datos_a_guardar)   

palabras_clave = ['course', 'department', 'faculty', 'other', 'project', 'staff', 'student']

for prompt in tqdm(prompts, desc="promting"):
    name, prompt_text = prompt
    logging.info("Prompt: %s", prompt_text)
    resultado_valido = False
    result = None
    intentos = 0  # Inicializar contador de intentos
    
    while not resultado_valido:
        intentos += 1  # Incrementar el contador de intentos
        # Ejecutar la función de prompting
        result = promting(llm, prompt_text, logging)
        
        # Verificar si alguna palabra clave está en el resultado
        if any(palabra in result for palabra in palabras_clave):
            # Escoger la primera palabra clave encontrada
            result = next((palabra for palabra in palabras_clave if palabra in result), result)
            resultado_valido = True
        else:
            logging.info("Resultado no válido, repitiendo el prompting para: %s", name)
    
    # Una vez obtenido un resultado válido, guardarlo
    datos_a_guardar.append((name, result, intentos))
    guardar_en_json(nombre_archivo_con_fecha, datos_a_guardar)

# Cierra el archivo de registro
logging.shutdown()
