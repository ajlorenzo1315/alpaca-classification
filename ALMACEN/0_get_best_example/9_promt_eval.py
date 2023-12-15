from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from llama_cpp import Llama
import llama_cpp
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
import time
from langchain.vectorstores import Pinecone
import json
import csv
import os
from langchain.chains import RetrievalQA
import logging
import csv
import tqdm
from transformers import AutoTokenizer
import pandas as pd

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
    cabecera = ['Titulo', 'Resultado', 'Intentos']

    for dato in datos:
        datos = dict(zip(cabecera, dato))
    
    with open(nombre_archivo, 'a', encoding='utf-8') as archivo_json:
        archivo_json.write(json.dumps(datos) + ",\n")

def get_common_key_per_class(path_file,max_key=10):
    
    with open(path_file) as f:
      data = json.load(f)
    #print(data.keys())
    return data['ground_truth'],data['words'][:max_key]

def get_file_2(path_file):  
  #print(path_file)
  with open(path_file) as f:
      data = json.load(f)
  #print(data.keys())
  dict_text={key: data[key] for key in data.keys() if key in {'title','summary','links'} }
  try:
    dict_text['keyword']=data['keyword_frequency_kebert'][:3]
  except:
    print(data.keys())
  return dict_text,data['ground_truth']

def get_file_3(path_file):  
  #print(path_file)
  with open(path_file) as f:
      data = json.load(f)
  #print(data.keys())
  dict_text={key: data[key] for key in data.keys() if key in {'title','summary','links'} }
  #try:
  #  dict_text['keyword']=data['keyword_frequency_kebert']
  #except:
  #  print(data.keys())
  return dict_text,data['ground_truth']

def get_class(train_path):
  #print(train_path)
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

def get_all_files(path_data,max_cont=-1):

    #'./data/splits/train_json'
    classes=get_class(path_data)
    archiver_per_clases={}
    prompt_per_clases={}
    cantidad_archivos={}
    promt_all_reson=[]
    archiver_all_reson=[]
    if len(classes)>0:
      for class_name in classes:
        #print(get_archives(os.path.join(path_data,i)))
        archiver_per_clases[class_name]=get_archives(os.path.join(path_data,class_name))[0:max_cont]
        prompt_per_clases[class_name]=[get_file_2(archive) for archive in  archiver_per_clases[class_name]]
        promt_all_reson.extend(prompt_per_clases[class_name])
        archiver_all_reson.extend(archiver_per_clases[class_name])

        #print(len(archiver_per_clases[class_name]),promt_all_reson)
        cantidad_archivos[class_name]=len(archiver_per_clases[class_name])
    else:
       archiver_per_clases['test']=get_archives(os.path.join(path_data,class_name))
       prompt_per_clases['test']=[get_file_2(archive) for archive  in  archiver_per_clases['test']]

    # Clave especial para ir de último
    # clave_especial = 'other'
    clave_especial = '-1'
    # Ordenar el diccionario por el valor, manteniendo 'other' al final si existe
    ordenado = sorted(cantidad_archivos.items(), key=lambda item: (item[0] == clave_especial, -item[1]))

    ordenado=[i[0] for i in ordenado ]
    return archiver_per_clases,prompt_per_clases,promt_all_reson,archiver_all_reson,ordenado




def get_all_keys(path_data,max_cont=-1,other=True,diferent=True):
    #'./data/splits/train_json'
    classes=get_class(path_data)
    promt_all_reson={}

    if len(classes)>0:
      for class_name in classes:
        if class_name != 'other' or other:
          archive=get_archives(os.path.join(path_data,class_name))[0]
          if diferent:
            promt_all_reson[class_name]=get_common_key_per_class(archive,max_cont)[1]
          else:
            promt_all_reson[class_name]=get_common_key_per_class(archive,max_cont)[1]
    print(promt_all_reson)
    return promt_all_reson



class LlamaLLM(LLM):
    model_path: str
    llm: Llama

    @property
    def _llm_type(self) -> str:
        return "llama-cpp-python"

    def __init__(self, model_path: str, **kwargs: Any):
        model_path = model_path
        llm = Llama(model_path=model_path,**kwargs)
        super().__init__(model_path=model_path, llm=llm, **kwargs)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.llm(prompt, stop=stop or [])
        return response["choices"][0]["text"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_path": self.model_path}
        
def creating_promt_2(class_name,text):
  class_name=(', ').join(class_name)
  return f"Classify the text in this class : [{class_name}]. Reply with only one of these words: [{class_name}]. \n\
  Text: '{text}' \n\
  Classification: "

def creating_promt(class_name,text_traindata,text_classification):
  class_name=(', ').join(class_name)
  text_traindata = ('\n\n').join([f" Text:'{text}' \n Classification: {label}" for text,label in text_traindata if label!='other'])

  return f" Classify the text in this class : {class_name}. Reply with only one word:  {class_name}. \n\
  Examples: \n\
  {text_traindata} \n\n\
  Text: '{text_classification}' \n\
  Classification: "

def tranform_text(text):
  return ' '.join([str(text[key]) for key in text.keys()])

def creating_promt_3(class_name_list,text_traindata,text_classification,keywords):
  print(class_name_list)
  class_name=(', ').join(class_name_list)
  text_traindata = ('\n\n').join([f" Text:'{text}' \n Classification: {label}" for text,label in text_traindata])
  palabras_clave = ('\n').join([f"{label}:{','. join(keywords[label])} " for label in class_name_list])#('\n').join([f"{label} keywords are {keys} " for label,keys in keywords])
  return f" Classify the text in this class : {class_name}.\n\
  Reply with only one word:  {class_name}. \n\
  {palabras_clave} \n\n \
  Examples: \n\
  {text_traindata} \n\n\
  Text: '{text_classification}' \n\
  Classification: "

def creating_promt_4(class_name_list,text_traindata,text_classification,keywords):
  #print(text_classification)
  # quitamos other de los ejemplos y dejamos que sea como todo lo que no tenga clase como otro
  class_name=(', ').join(class_name_list)
  text_traindata = ('\n\n').join([f" Text:'{text}' \n Classification: {label}" for text,label in text_traindata if label!='other'])
  palabras_clave = ('\n').join([f"{label}:{','. join(keywords[label])} " for label in class_name_list])#('\n').join([f"{label} keywords are {keys} " for label,keys in keywords])
  return f" Classify the text in this class : {class_name}.\n\
  Reply with only one word:  {class_name}. \n\
  {palabras_clave} \n\n \
  Examples: \n\
  {text_traindata} \n\n\
  Text: '{text_classification}' \n\
  Classification: "


def creating_promt_6(class_name_list,text_traindata,text_classification,keywords):
  #print(text_classification)
  # quitamos other de los ejemplos y dejamos que sea como todo lo que no tenga clase como otro
  class_name=(', ').join(class_name_list)
  text_traindata = ('\n\n').join([f" Text:'{text}' \n Classification: {label}" for text,label in text_traindata ])
  palabras_clave = ('\n').join([f"{label}:{','. join(keywords[label])} " for label in class_name_list])#('\n').join([f"{label} keywords are {keys} " for label,keys in keywords])
  return f" Classify the text in this class : {class_name}.\n\
  Reply with only one word:  {class_name}. \n\
  {palabras_clave} \n\n \
  Examples: \n\
  {text_traindata} \n\n\
  Text: '{text_classification}' \n\
  Classification: "


def creating_promt_5(class_name_list,text_traindata,text_classification,keywords):
  # quitamos other de los ejemplos y dejamos que sea como todo lo que no tenga clase como otro
  class_name=(', ').join(class_name_list)
  text_traindata = ('\n\n').join([f" Text:'{tranform_text(text)}' \n class: {label}" for text,label in text_traindata if label!='other'])
  palabras_clave = ('\n').join([f"{label}:{','. join(keywords[label])} " for label in class_name_list])#('\n').join([f"{label} keywords are {keys} " for label,keys in keywords])
  return f" Classify  text into one of 7 class : {class_name}.\n\
  Reply with only one word:  {class_name}. \n\
  {palabras_clave} \n\n \
  Examples: \n\
  {text_traindata} \n\n\
  Text: '{tranform_text(text_classification)}' \n\
  class: "


def creating_promt_7(class_name_list,text_traindata,text_classification,keywords):
  """"Classify the following texts into one of these categories: student, faculty, course, project, department, staff, or other. Provide your response using only one word from these options.

    Associated keywords for each category are:

    Staff: gmtserver, cs, apache, programming, computing, web, compiler, computer, compilers, distributed
    Project: computing, programming, compiler, robotics, distributed, software, computational, parallel, ai, multimedia
    Faculty: computing, programming, cs, algorithms, computational, edu, parallel, computer, compilers, distributed
    Department: cs, faculty, computing, edu, university, campus, ncsa, courses, webmaster, web
    Student: cs, programming, computing, resume, web, gmtserver, edu, page, university, computer
    Course: programming, syllabus, assignments, cs, lecture, algorithms, exam, cse, instructor, grading"""

  # quitamos other de los ejemplos y dejamos que sea como todo lo que no tenga clase como otro
  class_name=(', ').join(class_name_list)
  text_traindata = ('\n\n').join([f" Text:'{text}' \n Classification: {label}" for text,label in text_traindata ])
  palabras_clave = ('\n').join([f"{label}:{','. join(keywords[label])} " for label in class_name_list]) #('\n').join([f"{label}:{','. join(keys)}\ " for label,keys in keywords])
  return f"Classify the last text into one of these categories: {class_name}. Provide your response using only one word from these options.\n\
Associated keywords for each category are:\n{palabras_clave}\n\n\
Examples:\n{text_traindata}\n\n\
Classify:\n\
Text: {text_classification}\n\
Classification: "

def creating_promt_8(class_name,text_traindata,text_classification,keywords):
  """"Classify the following texts into one of these categories: student, faculty, course, project, department, staff, or other. Provide your response using only one word from these options.

    Associated keywords for each category are:

    Staff: gmtserver, cs, apache, programming, computing, web, compiler, computer, compilers, distributed
    Project: computing, programming, compiler, robotics, distributed, software, computational, parallel, ai, multimedia
    Faculty: computing, programming, cs, algorithms, computational, edu, parallel, computer, compilers, distributed
    Department: cs, faculty, computing, edu, university, campus, ncsa, courses, webmaster, web
    Student: cs, programming, computing, resume, web, gmtserver, edu, page, university, computer
    Course: programming, syllabus, assignments, cs, lecture, algorithms, exam, cse, instructor, grading"""

  # quitamos other de los ejemplos y dejamos que sea como todo lo que no tenga clase como otro
  class_name=(', ').join(class_name_list)
  text_traindata = ('\n\n').join([f" Text:'{text}' \n Classification: {label}" for text,label in text_traindata ])
  palabras_clave = ('\n').join([f"{label}:{','. join(keywords[label])} " for label in class_name_list])#('\n').join([f"{label}:{','. join(keys)} " for label,keys in keywords])
  return f"Classify the last Text into one of these categories: {class_name}. Provide your response using only one word from these options.\n\
\nAssociated keywords for each category are:\n\n{palabras_clave}\n\n\
Examples:\n{text_traindata}\n\n\
Text: {text_classification}\n\
Classification: "

def creating_promt_9(class_name_list,text_traindata,text_classification,keywords):
  """"Classify the following texts into one of these categories: student, faculty, course, project, department, staff, or other. Provide your response using only one word from these options.

    Associated keywords for each category are:

    Staff: gmtserver, cs, apache, programming, computing, web, compiler, computer, compilers, distributed
    Project: computing, programming, compiler, robotics, distributed, software, computational, parallel, ai, multimedia
    Faculty: computing, programming, cs, algorithms, computational, edu, parallel, computer, compilers, distributed
    Department: cs, faculty, computing, edu, university, campus, ncsa, courses, webmaster, web
    Student: cs, programming, computing, resume, web, gmtserver, edu, page, university, computer
    Course: programming, syllabus, assignments, cs, lecture, algorithms, exam, cse, instructor, grading"""

  # quitamos other de los ejemplos y dejamos que sea como todo lo que no tenga clase como otro
  class_name=(', ').join(class_name_list)
  text_traindata = ('\n\n').join([f" Text:'{text}' \n Classification: {label}" for text,label in text_traindata ])
  palabras_clave = ('\n').join([f"{label}:{','. join(keywords[label])} " for label in class_name_list])
  return f"Classify the last Text into one of these categories: {class_name}. Provide your response using only one word from these options.\n\
\nAssociated keywords for each category are:\n\n{palabras_clave}\n\n\
Examples:\n{text_traindata}\n\n\
Text: {text_classification}\n\
Classification: "

def creating_promt_9(class_name_list,text_traindata,text_classification,keywords):
  """"Classify the following texts into one of these categories: student, faculty, course, project, department, staff, or other. Provide your response using only one word from these options.

    Associated keywords for each category are:

    Staff: gmtserver, cs, apache, programming, computing, web, compiler, computer, compilers, distributed
    Project: computing, programming, compiler, robotics, distributed, software, computational, parallel, ai, multimedia
    Faculty: computing, programming, cs, algorithms, computational, edu, parallel, computer, compilers, distributed
    Department: cs, faculty, computing, edu, university, campus, ncsa, courses, webmaster, web
    Student: cs, programming, computing, resume, web, gmtserver, edu, page, university, computer
    Course: programming, syllabus, assignments, cs, lecture, algorithms, exam, cse, instructor, grading"""

  # quitamos other de los ejemplos y dejamos que sea como todo lo que no tenga clase como otro
  class_name=(', ').join(class_name_list)
  text_traindata = ('\n\n').join([f" Text:'{text}' \n Classification: {label}" for text,label in text_traindata ])
  palabras_clave = ('\n').join([f"{label}:{','. join(keywords[label])} " for label in class_name_list])
  return f"Classify the last Text into one of these categories: {class_name}. Provide your response using only one word from these options.\n\
\nAssociated keywords for each category are:\n\n{palabras_clave}\n\n\
Examples:\n{text_traindata}\n\n\
Text: {text_classification}\n\
Classification: "


def creating_promt_10(class_name_list,text_traindata,text_classification,keywords):
  """"Classify the following texts into one of these categories: student, faculty, course, project, department, staff, or other. Provide your response using only one word from these options.

    Associated keywords for each category are:

    Staff: gmtserver, cs, apache, programming, computing, web, compiler, computer, compilers, distributed
    Project: computing, programming, compiler, robotics, distributed, software, computational, parallel, ai, multimedia
    Faculty: computing, programming, cs, algorithms, computational, edu, parallel, computer, compilers, distributed
    Department: cs, faculty, computing, edu, university, campus, ncsa, courses, webmaster, web
    Student: cs, programming, computing, resume, web, gmtserver, edu, page, university, computer
    Course: programming, syllabus, assignments, cs, lecture, algorithms, exam, cse, instructor, grading"""

  # quitamos other de los ejemplos y dejamos que sea como todo lo que no tenga clase como otro
  class_name=(', ').join(class_name_list)
  text_traindata = ('\n\n').join([f" Text:'{text}' \n Category: {label}" for text,label in text_traindata ])
  palabras_clave = ('\n').join([f"{label}:{','. join(keywords[label])} " for label in class_name_list])
  return f"Classify the last Text into one of these categories: {class_name}. Provide your response using only one word from these options.\n\
\nAssociated keywords for each category are:\n\n{palabras_clave}\n\n\
Examples:\n{text_traindata}\n\n\
Text: {text_classification}\n\
Category: "



def get_name_archive(archive):
  return os.path.splitext(os.path.basename(archive))[0]
#../data/splits/test_new_data/test/aaclkul.json
#alpaca-classification/data/splits/train_new_data/course/aaexyuw.json
promt=get_file_2('../data/splits/train_new_data/course/aaexyuw.json')
key_promt='key_promt'
keys_wod=get_all_keys('../data_extrac/key_promt/',7)
print(keys_wod)
path_data='../data/splits/train_new_data'
classes=get_class(path_data)
promt=creating_promt_3(classes,[(promt,'course')],promt,keys_wod)
#print(promt)
# Configura el archivo de registro
log_filename = "log_promt_9_eval.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#
#train,promt_train,b,d=get_all_files('../data/splits/train_new_short',-1)
train,promt_train,b_2,d_2,classes=get_all_files('../data/splits/train_new_data',-1)
##train,promt_train,b,d
#path_data='../data/splits/train_new_short'
#classes=get_class(path_data)
#query=creating_promt_2(classes,b[0][0])
#query2=creating_promt(classes,b_2,b[0][0])
#
#print(query)
#logging.info(query)
#
#print(query2)
#logging.info(query2)

#


test,promt_test,a,c,_=get_all_files('../data/splits/test_new_data',-1)
#test,promt_test,a,c=get_all_files('../data/splits/train_new_data',-1)
class_selected='staff'
path_to_txt='./staff_predicted_student_files.txt'
# Cargar los nombres de archivo del archivo txt y almacenarlos en una lista
with open(path_to_txt, 'r') as f:
    archives_pobe = f.read().split(',')
  
#print(archives_pobe)


model_name='llama2-chat-ayb-13b.Q5_K_M'
model_path = "../llama_models/llama2-chat-ayb-13b.Q5_K_M.gguf"
llm = LlamaLLM(model_path=model_path,n_ctx=4096,n_gpu_layers=25)

#
#


#archives_pobe={'../data/splits/train_new_data/staff/varuyxnz.json'}
#print(promt_train['staff'][0])

#select_example={'student':968,'department':137,'faculty':602,'course':[254,729],'project':186,'staff':37}
select_example={'student':968,'department':137,'faculty':106,'course':682,'project':186,'staff':37}
#select_example={'student':968,'department':137,'faculty':602,'course':254,'project':186,'staff':37}
#select_example={'student':346,'department':143,'faculty':602,'course':729,'project':186,'staff':24,'other':312}

#select_example={'student':557,'department':103,'faculty':212,'course':593,'project':13,'staff':61,'other':1549}
select_example={'student':1189,'department':126,'faculty':479,'course':600,'project':131,'staff':76,'other':498}


examples_to_promt=[]

for key in promt_train.keys():
  if key !='other':
    logging.info(f'\n{train[key][select_example[key]]}\n')
    examples_to_promt.append(promt_train[key][select_example[key]])

#print(len(get_examples),promt_train['staff'][i] )
logging.info(examples_to_promt)
#prompts=[(name,creating_promt_8(classes,examples_to_promt,text_class[0],keys_wod),
#creating_promt_2(classes,text_class[0])) for name,text_class in zip(c,a) if get_name_archive(name) in archives_pobe ]
# probamos con todods los archivos
prompts=[(name,creating_promt_7(classes,examples_to_promt,text_class,keys_wod),
creating_promt_2(classes,text_class[0])) for name,text_class in zip(c,a) ]
#print(prompts[0])
datos_a_guardar_1 = []
datos_no_promting=[]
name_examaple_text='-1'#get_name_archive(example)

path_rag='./result_2/key_promt_new_promt_9_eval_staff_llama2-chat-ayb-13b.Q5_K_M_with_other.csv'
sub_df = pd.read_csv(path_rag)
not_process_again=sub_df['file_name'].tolist()

for indx,prompt in tqdm.tqdm(enumerate(prompts),desc="promting"):
    name,prompt1,prompt2=prompt
    if get_name_archive(name) in not_process_again:
      continue
    try:
      logging.info(f'\n\n{name}\n\n')
      logging.info(f'\n\n1 {len(prompt1)}\n\n')
      logging.info(f'\n\n{prompt1}\n\n')
      ll_result1=llm(prompt1)
      logging.info(ll_result1)
      datos_a_guardar_1.append((name,ll_result1,indx))
    except Exception as e:  # Captura cualquier excepción
      # Registra la excepción para entender qué salió mal
      logging.error(f'Error: {e}')
      logging.info(f'\n\nfallo por {len(prompt1)}\n\n')
      datos_no_promting.append((name,'',1))
    guardar_en_json(f'./staff_student/{key_promt}_new_promt_9_eval_{class_selected}_{model_name}_with_other.json', datos_a_guardar_1)

#print(len(datos_no_promting))


"""En este codigo lo que vamos a probar cual es el mejor ejemplo para en el promt para poder clasificar la clase para ello probaremos distintos ejemplos 
de promt en el cual podamos mirar cual es e mejor ejemplo por cada clase luego comprobaremos combinaciones de los mismos

primero miraremos que archivos clasifica mal y usaremos estos archivos para ver si los clasifica bien luego probaremos los que consigan clasificar bien estos
archivos sobre el conjunto de archivos totales"""

"""Selecionamos los teoricamente mejores candidatos luego usamos un recorte de claves y mas cosas y cambio de orden de los nombres de las clases
obtenidos con

 print(ldamodel.print_topics(num_topics=20, num_words=20))
"""