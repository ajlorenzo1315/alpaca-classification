import logging
from llama_cpp import Llama
import json
from tqdm import tqdm
import os



def get_file(path_file):
    with open(path_file) as f:
        data = json.load(f)
    print(data.keys())
    return ('\n').join([f"{data[key].split('Content-length')}" for key in data.keys() if key != 'ground_truth' and key != 'Título']),data['ground_truth']

def get_file_2(path_file):
    with open(path_file) as f:
        data = json.load(f)
    print(data.keys())
    return [f"{key} : {data[key]}" for key in data.keys() if key != 'ground_truth' ],data['ground_truth']
  

def get_class(train_path):
  if os.path.exists(train_path) and os.path.isdir(train_path):
    carpetas = [nombre for nombre in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, nombre))]
    return carpetas
  else:
    print("El directorio especificado no existe o no es un directorio válido.")
  
def creating_promt(class_name,text_traindata,text_classification):
  class_name=(', ').join(class_name)
  text_traindata = ('\n\n').join([f" Text:'{text}' \n Classication: {label}" for text,label in text_traindata])
  return f"Classify the text in this class : {class_name}. Reply with only one word:  {class_name}. \n\
  Examples: \n\
  {text_traindata} \n\n\
  Text: '{text}' \n\
  Classication: "

def creating_promt_2(class_name,text_traindata,text_classification):
  class_name=(', ').join(class_name)
  text_traindata = ('\n\n').join([f" Text:'{text}' \n Classication: {label}" for text,label in text_traindata])
  return f"Classify the text in this class : {class_name}. Reply with only one word:  {class_name}. \n\
  Text: '{text}' \n\
  Classication: "

def promting(llm,prompt,logging):
    # Genera la respuesta
    output = llm(prompt)
    #, max_tokens=32, stop=["Q:", "\n"], echo=True)

    # Registra el prompt y la respuesta en el archivo de registro
    logging.info("Prompt: %s", prompt)
    logging.info("Respuesta: %s", output['choices'])
    logging.info("ALL_INFO: %s", output)
    print(output['choices'])

# Configura el archivo de registro
log_filename = "llama_promt_log.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Inicializa Llama
# Inicializa Llama y guarda la información del modelo en el registro
model_path = "./llama_models/llama-2-7b.Q8_0.gguf"
logging.info("Modelo: %s", model_path)
llm = Llama(model_path="./llama_models/llama-2-7b.Q8_0.gguf", n_ctx=2048)


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
template=creating_promt_2(class_name,[[text,label],[text_2,label_2]],text )
print(template)

prompts=[template]

for prompt in prompts:
    promting(llm,prompt,logging)

# Imprime la respuesta
#print(output)

# Cierra el archivo de registro
logging.shutdown()
