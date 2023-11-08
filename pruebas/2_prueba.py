import argparse

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
        llm = Llama(model_path=model_path)
        super().__init__(model_path=model_path, llm=llm, **kwargs)

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.llm(prompt, stop=stop or [])
        return response["choices"][0]["text"]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_path": self.model_path}


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="./llama_models/llama2-chat-ayb-13b.Q5_K_M.gguf")
args = parser.parse_args()

# Load the model
llm = LlamaLLM(model_path=args.model)

# Basic Q&A
answer = llm(
    "Question: What is the capital of France? Answer: ", stop=["Question:", "\n"]
)
print(f"Answer: {answer.strip()}")

# Using in a chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["product"],
    template="\n\n### Instruction:\nWrite a good name for a company that makes {product}\n\n### Response:\n",
)
chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
print(chain.run("colorful socks"))



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


rag = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=db.as_retriever()
)
prompt = "What is Llama 2?"
print(rag(prompt))
print(llm(prompt))




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
  return f"Classify the text in this class : [{class_name}]. Reply with only one of these words: [{class_name}. \n\
  Text: '{text}' \n\
  Classication: "


def creating_promt_3(class_name,text_traindata,text_classification):
  class_name=(', ').join(class_name)
  text_traindata = ('\n\n').join([f" Text:'{text}' \n Classication: {label}" for text,label in text_traindata])
  return f"<s>[INST] <<SYS>> You are a helpful assistant. <</SYS>> Classify the text in this class : {class_name}. Reply with only one word:  {class_name}. \n\
  Examples: \n\
  {text_traindata} \n\n\
  Text: '{text_classification}' \n\
  Classication: "
  [/INST] "

def creating_RAG(class_name,text_traindata):

    text_traindata = ('\n\n').join([f" Text:'{text}' \n Classication: {label}" for text,label in text_traindata])
    return f" Examples: \n\
    {text_traindata} \n\n "

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
    logging.info("Respuesta: %s", output['choices'])
    logging.info("ALL_INFO: %s", output)
    #print(output['choices'])
    #print("Prompt: %s", prompt)
    #print(output['choices'][0]['text'])
    return output['choices'][0]['text']

# Configura el archivo de registro
log_filename = "llama_promt_log.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Inicializa Llama
# Inicializa Llama y guarda la información del modelo en el registro
model_path = "./llama_models/llama2-chat-ayb-13b.Q5_K_M.gguf"
logging.info("Modelo: %s", model_path)
llm = Llama(model_path=model_path, n_ctx=4096)


# Define el prompt
#prompts = ["{ 'prompt': 'Can you explain what a transformer is (in a machine learning context)?','system_prompt': 'You are a pirate' }",
#         "{ 'prompt': 'Can you explain what a transformer is (in a machine learning context)?','system_prompt': 'You are responding to highly technical customers' }",
#         "[INST] Hi! [/INST]",
#         "Write a poem about a flower"
#        ]

template = "Classify the text into neutral, negative, or positive. Reply with only one word: Positive, Negative, or Neutral. \n  \
Examples: \n  \
Text: Big variety of snacks (sweet and savoury) and very good espresso Machiatto with reasonable prices,\
you can't get wrong if you choose the place for a quick meal or coffee. \n  \
Sentiment: Positive. \n  \
Text: I got food poisoning \n  \
Sentiment: Negative. \n  \
Text: {text} \n  \
Sentiment: "


template=""" Classify the text into neutral, negative, or positive. Reply with only one word: Positive, Negative, or Neutral.
  """


text,label=get_file('./data/dummy.json')
text_2,label_2=get_file('./data/dummy_2.json')
class_name=get_class('./data/splits/train')
template=creating_promt(class_name,[[text,label],[text_2,label_2]],text_2 )
#print(template)
train,promt_train,b,d=get_all_files('./data/splits/train_new',1)
test,promt_test,a,c=get_all_files('./data/splits/test_new',-1)
#test,promt_test,a,c=get_all_files('./data/splits/train_new',2)
RAG=creating_RAG(class_name,b)
with open('knowledge_train.txt', 'w') as fp:
    fp.write('\n'.join(RAG))

print(class_name,train.keys())
prompts=[(name,creating_promt_2(class_name,b,text_class[0] )) for name,text_class in zip(c,a)]

datos_a_guardar = []

# Load documents and split them
documents = TextLoader("knowledge_train.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
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
