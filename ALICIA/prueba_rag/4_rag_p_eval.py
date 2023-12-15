from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from llama_cpp import Llama
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
import pinecone
import time
from langchain.vectorstores import Pinecone
import json
import csv
import os
from langchain.chains import RetrievalQA
import logging
import csv
import tqdm

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


def get_file_2(path_file):  
  #print(path_file)
  with open(path_file) as f:
      data = json.load(f)
  #print(data.keys())
  return {key: data[key] for key in data.keys() if key != 'ground_truth' },data['ground_truth']

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
  text_traindata = ('\n\n').join([f" Text:'{text}' \n Classification: {label}" for text,label in text_traindata])

  return f" Classify the text in this class : {class_name}. Reply with only one word:  {class_name}. \n\
  Examples: \n\
  {text_traindata} \n\n\
  Text: '{text_classification}' \n\
  Classification: "

# Configura el archivo de registro
log_filename = "llama_eval_log.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


train,promt_train,b,d=get_all_files('../data/splits/train_new_short',-1)
train,promt_train,b_2,d_2=get_all_files('../data/splits/train_new_short',2)
#train,promt_train,b,d
path_data='../data/splits/train_new_short'
classes=get_class(path_data)
query=creating_promt_2(classes,b[0][0])
query2=creating_promt(classes,b_2,b[0][0])

print(query)
logging.info(query)

print(query2)
logging.info(query2)

model_path = "../llama_models/llama-2-13b-chat.Q4_K_M.gguf"
# total VRAM used: 6782.52 MB (model: 6424.51 MB, context: 358.00 MB)

llm = LlamaLLM(model_path=model_path,n_ctx=4096,n_gpu_layers=30)

# get API key from app.pinecone.io and environment from console
PINECONE_API_KEY='bc795b6a-d0aa-4877-bdc6-414135eb4bef'
PINECONE_ENV='gcp-starter'
pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY') or PINECONE_API_KEY,
    environment=os.environ.get('PINECONE_ENVIRONMENT') or PINECONE_ENV
)


index_name = 'llama-2-rag'
index = pinecone.Index(index_name)
logging.info(index.describe_index_stats())

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 16}
)


text_field = 'text'  # field in metadata that contains text content

vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)

vectorstore.similarity_search(
    query,  # the search query
    k=5  # returns top 3 most relevant chunks of text
)



rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=vectorstore.as_retriever()
)


test,promt_test,a,c=get_all_files('../data/splits/train_new_short',13)
#test,promt_test,a,c=get_all_files('./data/splits/train_new',2)
prompts=[(name,creating_promt(classes,b_2,text_class[0]),creating_promt_2(classes,text_class[0])) for name,text_class in zip(c,a)]

datos_a_guardar_1 = []
datos_a_guardar_2 = []
datos_a_guardar_3 = []
datos_a_guardar_4 = []

for prompt in tqdm.tqdm(prompts,desc="promting"):
    name,prompt1,prompt2=prompt
    logging.info(f'\n\n{name}\n\n')
    logging.info('\n\n1\n\n')
    ll_result1=llm(prompt1)
    logging.info(ll_result1)
    #rag_result1=rag_pipeline(query)['result']
    #logging.info(rag_result1)

    logging.info('\n\n2\n\n')

    ll_result2=llm(prompt2)
    logging.info(ll_result2)
    rag_result2=rag_pipeline(prompt2)
    logging.info(rag_result2)

    datos_a_guardar_1.append((name,ll_result1))
    datos_a_guardar_2.append((name,ll_result2))
    datos_a_guardar_3.append((name,rag_result2))
    #datos_a_guardar_4.append((name,rag_result2))
    guardar_en_csv('resultados_llm1_short.csv', datos_a_guardar_1)   
    guardar_en_csv('resultados_llm2_short.csv', datos_a_guardar_2)   
    guardar_en_csv('resultados_rag1_short.csv', datos_a_guardar_3)   
    #guardar_en_csv('resultados_rag2.csv', datos_a_guardar_4)   

