import argparse
import ray
from llama_cpp import Llama

from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any


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

from langchain.chains import RetrievalQA

from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain import PromptTemplate, LLMChain

from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Embedding Model for converting text to numerical representations
embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)


import logging
from llama_cpp import Llama
import json
from tqdm import tqdm
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



# ./main -m models/speechless-llama2-13b.Q4_K_M.gguf -ngl 36 -n 3620 -c 3620
#-p "### System: You possess expertise in creating articles with markdown formatting. 
# ### User: Craft an article in encyclopedia-style on the topic 'tesla stock' in English. 
# The article length should range from 800 to 2400 words. Utilize markdown for styling, using 
# '#' for headers, '##' for sub-headers, and so on. Refrain from employing HTML; only markdown should be used.
# Enhance the article's readability with markdown features like bold, italics, and hyperlinks pointing to Wikipedia entries where relevant.
# Commence the article with '# [title]', 
#followed immediately by '## Table of Contents\n1. '..., and then start the first subheading with '## 1. '... and continue accordingly." --temp 0.3

# ### System: You possess expertise in web page classification
# ### User:
def get_file(path_file):
    with open(path_file) as f:
        data = json.load(f)
    print(data.keys())
    return ('\n').join([f"{data[key].split('Content-length')}" for key in data.keys() if key != 'ground_truth' and key != 'Título']),data['ground_truth']

def proces_file(key,text):
    if key=="Cuerpo":
      text.split(': ')

def get_file_2(path_file):  
  #print(path_file)
  with open(path_file) as f:
      data = json.load(f)
  #print(data.keys())
  return [f"{key} : {data[key]}" for key in data.keys() if key != 'ground_truth' ],data['ground_truth']
  

def get_class(train_path):
  print(train_path)
  if os.path.exists(train_path) and os.path.isdir(train_path):
    carpetas = [nombre for nombre in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, nombre))]
    return carpetas
  else:
    print("El directorio especificado no existe o no es un directorio válido.")



# ### System: You possess expertise in web page classification
# ### User:
def get_file(path_file):
    with open(path_file) as f:
        data = json.load(f)
    print(data.keys())
    return ('\n').join([f"{data[key].split('Content-length')}" for key in data.keys() if key != 'ground_truth' and key != 'Título']),data['ground_truth']

def proces_file(key,text):
    if key=="Cuerpo":
      text.split(': ')

def get_file_2(path_file):  
  #print(path_file)
  with open(path_file) as f:
      data = json.load(f)
  #print(data.keys())
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

  return f" Classify the text in this class : {class_name}. Reply with only one word:  {class_name}. \n\
  Examples: \n\
  {text_traindata} \n\n\
  Text: '{text_classification}' \n\
  Classication: "

def creating_promt_2(class_name,text_traindata,text_classification):
  class_name=(', ').join(class_name)
  text_traindata = ('\n\n').join([f" Text:'{text}' \n Classication: {label}" for text,label in text_traindata])
  return f"Classify the text in this class : [{class_name}]. Reply with only one of these words: [{class_name}]. \n\
  Text: '{text}' \n\
  Classication: "


def creating_promt_3(class_name,text_traindata,text_classification):
  class_name=(', ').join(class_name)
  text_traindata = ('\n\n').join([f" Text:'{text}' \n Classication: {label}" for text,label in text_traindata])
  return f"<s>[INST] <<SYS>> You are a helpful assistant. <</SYS>> Classify the text in this class : {class_name}. Reply with only one word:  {class_name}. \n\
  Examples: \n\
  {text_traindata} \n\n\
  Text: '{text_classification}' \n\
  Classication:  \n \
  [/INST] "

def creating_RAG(class_name,text_traindata):

    text_traindata = ('\n\n').join([f"Text:'{text}' \nClassication: {label}" for text,label in text_traindata])
    return f"Examples: \n{text_traindata} \n\n "

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

        #print(len(archiver_per_clases[class_name]),promt_all_reson)
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
    #logging.info("Respuesta: %s", output['choices'])
    logging.info("ALL_INFO: %s", output)
    #print(output['choices'])
    #print("Prompt: %s", prompt)
    #print(output['choices'][0]['text'])
    return output

# Configura el archivo de registro
log_filename = "llama_promt_log.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Inicializa Llama
# Inicializa Llama y guarda la información del modelo en el registro

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="./llama_models/llama2-chat-ayb-13b.Q5_K_M.gguf")
args = parser.parse_args()
logging.info("Modelo: %s", args.model)
# Load the model
llm = LlamaLLM(model_path=args.model,n_ctx=4096)

text,label=get_file('./data/dummy.json')
text_2,label_2=get_file('./data/dummy_2.json')
class_name=get_class('./data/splits/train')
template=creating_promt(class_name,[[text,label],[text_2,label_2]],text_2 )
#print(template)
train,promt_train,b,d=get_all_files('./data/splits/train_new',-1)
#test,promt_test,a,c=get_all_files('./data/splits/test_new',-1)
test,promt_test,a,c=get_all_files('./data/splits/train_new',1)


RAG=creating_RAG(class_name,b)
with open('knowledge_train.txt', 'w') as fp:
    fp.write(''.join(RAG))

train,promt_train,b,d=get_all_files('./data/splits/train_new',1)
print(class_name,train.keys())
prompts=[(name,creating_promt_2(class_name,b,text_class[0] )) for name,text_class in zip(c,a)]

datos_a_guardar = []

# Load documents and split them
documents = TextLoader("knowledge_train.txt").load()
#text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_size=["\n\n"],chunk_overlap=0)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, separators=["\n\n"],chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create local vector database
db = FAISS.from_documents(docs, embedding_model)


rag = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=db.as_retriever()
)

for prompt in tqdm(prompts,desc="promting"):
    name,prompt=prompt
    logging.info("Prompt: %s", prompt)
    result=promting(rag,prompt,logging)
    datos_a_guardar.append((name,result))
    guardar_en_csv('resultados.csv', datos_a_guardar)   
# Imprime la respuesta
#print(output)

# Cierra el archivo de registro
logging.shutdown()


