import logging
from llama_cpp import Llama

from langchain.chains import RetrievalQA

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain import PromptTemplate, LLMChain

from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Embedding Model for converting text to numerical representations
embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)


# Our tiny knowledge base
knowledge_base = [
    "On July 18, 2023, in partnership with Microsoft, Meta announced LLaMA-2, the next generation of LLaMA." ,
    "Llama 2, a collection of pretrained and fine-tuned large language models (LLMs) ",
    "The fine-tuned LLMs, called Llama 2-Chat, are optimized for dialogue use cases.",
    "Meta trained and released LLaMA-2 in three model sizes: 7, 13, and 70 billion parameters.",
    "The model architecture remains largely unchanged from that of LLaMA-1 models, but 40% more data was used to train the foundational models.",
    "The accompanying preprint also mentions a model with 34B parameters that might be released in the future upon satisfying safety targets."
]

with open('knowledge_base.txt', 'w') as fp:
    fp.write('\n'.join(knowledge_base))

# Load documents and split them
documents = TextLoader("knowledge_base.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Create local vector database
db = FAISS.from_documents(docs, embedding_model)




# Configura el archivo de registro
log_filename = "llama_promt_log.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Inicializa Llama
# Inicializa Llama y guarda la información del modelo en el registro
model_path = "./llama_models/llama2-chat-ayb-13b.Q5_K_M.gguf"
logging.info("Modelo: %s", model_path)
llm = Llama(model_path=model_path)

# Define el prompt
prompt = "What is Llama 2?"

prompt2 = PromptTemplate(template=prompt, input_variables=["question"])

# Genera la respuesta
output = llm(prompt, max_tokens=32, stop=["Q:", "\n"], echo=True)

#llm_chain = LLMChain(prompt=prompt2, llm=llm)
# RAG Pipeline
rag = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=db.as_retriever()
)

rag(prompt)

# Registra el prompt y la respuesta en el archivo de registro
logging.info("Prompt: %s", prompt)
logging.info("Respuesta: %s", output['choices'])
logging.info("ALL_INFO: %s", output)

# Imprime la respuesta
print(output)

# Cierra el archivo de registro
logging.shutdown()


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


            