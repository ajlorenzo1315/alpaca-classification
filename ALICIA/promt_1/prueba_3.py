import logging
from llama_cpp import Llama
import json
from tqdm import tqdm
# Configura el archivo de registro
log_filename = "llama_promt_log.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Inicializa Llama
# Inicializa Llama y guarda la información del modelo en el registro
model_path = "./llama_models/llama-2-7b.Q8_0.gguf"
logging.info("Modelo: %s", model_path)
llm = Llama(model_path="./llama_models/llama-2-7b.Q8_0.gguf")

data_path='./samples/inputs_direct_ag_news.json'
with open(data_path) as f:
        data = json.load(f)

max_batch_size=1
for start_idx in tqdm(range(0, len(data), max_batch_size)):
    end_idx = min(start_idx + max_batch_size, len(data))
    batch = data[start_idx:end_idx]
    prompts, completions = [], []
    for example in batch:
        for label_word in example["label_words"]:
            print(label_word)
            prompts.append(example["prompt"].format(label_word=label_word))
            completions.append(example["completion"].format(text=example["text"]))
    log_probs = []
    for micro_start_idx in range(0, len(prompts), max_batch_size):
        micro_end_idx = min(micro_start_idx + max_batch_size, len(prompts))
        micro_prompts = prompts[micro_start_idx:micro_end_idx]
        micro_completions = completions[micro_start_idx:micro_end_idx]
        log_probs.extend(
            generator.compute_log_probs(micro_prompts, micro_completions)
        )
        
        # Genera la respuesta
        output = llm(micro_prompts, max_tokens=32, stop=["Q:", "\n"], echo=True)

        # Registra el prompt y la respuesta en el archivo de registro
        logging.info("Prompt: %s", prompt)
        logging.info("Respuesta: %s", output['choices'])
        logging.info("ALL_INFO: %s", output)




# Define el prompt
prompt = "Q: what it is your name? A: "

# Genera la respuesta
output = llm(prompt, max_tokens=32, stop=["Q:", "\n"], echo=True)

# Registra el prompt y la respuesta en el archivo de registro
logging.info("Prompt: %s", prompt)
logging.info("Respuesta: %s", output['choices'])
logging.info("ALL_INFO: %s", output)

# Imprime la respuesta
print(output)

# Cierra el archivo de registro
logging.shutdown()