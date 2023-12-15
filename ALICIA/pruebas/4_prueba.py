import os
import ray
import json
import argparse

import matplotlib.pyplot as plt

from pathlib import Path
from rag.data import extract_sections
from rag.config import EFS_DIR

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

from llama_cpp import Llama

from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any


from rag.embed import get_embedding_model
from rag.utils import get_num_tokens, trim
from rag.config import EMBEDDING_DIMENSIONS, MAX_CONTEXT_LENGTHS



class LlamaLLM(LLM):
    model_path: str
    llm: Llama

    @property
    def _llm_type(self) -> str:
        return "llama-cpp-python"

    def __init__(self, model_path: str, embedding_model_name="thenlper/gte-base", temperature=0.0, 
                 max_context_length=4096, system_content="", assistant_content="", **kwargs: Any):
        model_path = model_path
        llm = Llama(model_path=model_path,**kwargs)
        # Embedding model
        self.embedding_model = get_embedding_model(
            embedding_model_name=embedding_model_name, 
            model_kwargs={"device": "cuda"}, 
            encode_kwargs={"device": "cuda", "batch_size": 100})

        self.temperature = temperature
        self.context_length = max_context_length - get_num_tokens(system_content + assistant_content)
        self.system_content = system_content
        self.assistant_content = assistant_content
        super().__init__(model_path=model_path, llm=llm, **kwargs)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:

        contexts_results = semantic_search(
            query=prompt, 
            embedding_model=self.embedding_model, 
            num_chunks=num_chunks)

        # Generate response
        context = [item["text"] for item in context_results]
        sources = [item["source"] for item in context_results]
        user_content = f"query: {query}, context: {context}"
        answer = generate_response(
            llm=self.llm,
            temperature=self.temperature,
            stream=stream,
            system_content=self.system_content,
            assistant_content=self.assistant_content,
            user_content=trim(user_content, self.context_length))

        #response = self.llm(prompt, stop=stop or [])
        
        result = {
            "question": query,
            "sources": sources,
            "answer": answer,
            "llm": self.llm,
        }
        return result
        #return response["choices"][0]["text"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_path": self.model_path}


#
#DOCS_DIR =Path("./data/splits/train") #Path(EFS_DIR, "docs.ray.io/en/master/")
#ds = ray.data.from_items([{"path": path} for path in DOCS_DIR.rglob("*.html") if not path.is_dir()])
#print(f"{ds.count()} documents")
#
#sample_html_fp = Path("./data/splits/train/course/aaexyuw")#Path("./data/splits/train")
#extract_sections({"path": sample_html_fp})
#
#
## Extract sections
#sections_ds = ds.flat_map(extract_sections)
#sections_ds.count()
#
#
#
#section_lengths = []
#for section in sections_ds.take_all():
#    section_lengths.append(len(section["text"]))
#
def creating_RAG(class_name,text_traindata):

    text_traindata = ('\n\n').join([f"Text:'{text}' Classication: {label}" for text,label in text_traindata])
    return f"{text_traindata} \n\n "

def get_class(train_path):
  print(train_path)
  if os.path.exists(train_path) and os.path.isdir(train_path):
    carpetas = [nombre for nombre in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, nombre))]
    return carpetas
  else:
    print("El directorio especificado no existe o no es un directorio válido.")

def get_file_2(path_file):  
  #print(path_file)
  with open(path_file) as f:
      data = json.load(f)
  #print(data.keys())
  return [f"{key} : {data[key]}" for key in data.keys() if key != 'ground_truth' ],data['ground_truth']


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


class_name=get_class('./data/splits/train')
train,promt_train,b,d=get_all_files('./data/splits/train_new',-1)
RAG=creating_RAG(class_name,b)
with open('knowledge_train.txt', 'w') as fp:
    fp.write(''.join(RAG))

print(class_name,train.keys())

documents = TextLoader("knowledge_train.txt").load()
# Text splitter
#chunk_size = 300
#chunk_overlap = 50
#text_splitter = RecursiveCharacterTextSplitter(
#    separators=["\n\n"],
#    chunk_size=chunk_size,
#    chunk_overlap=chunk_overlap,
#    length_function=len)
#
#
#
## Chunk a sample section
#chunks = text_splitter.split_documents(documents)


embedding_model_name="thenlper/gte-base"
llm="meta-llama/Llama-2-7b-chat-hf"

query = "What is the default batch size for map_batches?"
system_content = "Answer the query using the context provided. Be succinct."


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="./llama_models/llama2-chat-ayb-13b.Q5_K_M.gguf")
args = parser.parse_args()
print("Modelo: %s", args.model)
# Load the model
llm = LlamaLLM(model_path=args.model,n_ctx=4096)

agent = LlamaLLM(model_path=args.model,
    embedding_model_name=embedding_model_name,
    llm=llm,
    max_context_length=MAX_CONTEXT_LENGTHS[llm],
    system_content=system_content)
result = agent(query=query, stream=False)
print("\n\n", json.dumps(result, indent=2))



def chunk_section(chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len)
    chunks = text_splitter.split_documents(documents)
    return [{"text": chunk.page_content, "source": chunk.metadata["source"]} for chunk in chunks]

print(chunk_section(chunk_size=300,chunk_overlap=50)[0])
