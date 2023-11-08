import logging
from llama_cpp import Llama

# Configura el archivo de registro
log_filename = "llama_promt_log.txt"
logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Inicializa Llama
# Inicializa Llama y guarda la información del modelo en el registro
model_path = "./llama_models/llama-2-7b.Q8_0.gguf"
logging.info("Modelo: %s", model_path)
llm = Llama(model_path="./llama_models/llama-2-7b.Q8_0.gguf")

# Define el prompt
prompt = "Q: hi, how are you? A: "

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