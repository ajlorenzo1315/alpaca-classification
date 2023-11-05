import logging
from llama_cpp import Llama
import json
from tqdm import tqdm

def promting(llm,prompt,logging):
    # Genera la respuesta
    output = llm(prompt)#, max_tokens=32, stop=["Q:", "\n"], echo=True)

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
llm = Llama(model_path="./llama_models/llama-2-7b.Q8_0.gguf")


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

template = """Classify the text into neutral, negative, or positive. Reply with only one word: Positive, Negative, or Neutral.

Examples:
Text: Big variety of snacks (sweet and savoury) and very good espresso Machiatto with reasonable prices, you can't get wrong if you choose the place for a quick meal or coffee.
Sentiment: Positive.

Text: I got food poisoning
Sentiment: Negative.

Text: {text}
Sentiment:"""


template = """Classify the text into neutral, negative, or positive. Reply with only one word: Positive, Negative, or Neutral.

Examples:
Text: Big variety of snacks (sweet and savoury) and very good espresso Machiatto with reasonable prices, you can't get wrong if you choose the place for a quick meal or coffee.
Sentiment: Positive.

Text: I got food poisoning
Sentiment: Negative.

Text: {text}
Sentiment:"""

template="""
You are a classifier for web pages.
  You are given informations about a web page and you have to classify it as one of the following categories: 
  e-commerce product list/catalog, e-commerce product details, e-commerce cart, e-commerce shipping/delivery information, e-commerce customer reviews/testimonial, e-commerce gift cards, e-commerce returns/refunds and exchanges, physical store address/location/direction, customer support and assistance/frequently asked questions, contact us/get a pricing quote, blog, legal informations and terms and conditions, information about the company, press material, career opportunities, account login/register, other, page not found
  You will return a top 3 of the most probable categories for the given web page.
  Your response should be a list of comma separated values, example: e-commerce product details, e-commerce product list/catalog, legal informations and terms and conditions
  
  Here are the informations you have about the web page:
  The web page url is the more important to classify the web page: 
  https://c64audio.com/collections/8-bit-symphony

  The Web page open graph information provide a good insight about the web page content: 
  og:site_name: C64Audio.com
  og:url: https://c64audio.com/collections/8-bit-symphony
  og:title: 8-Bit Symphony
  og:type: product.group
  og:description: Imagine hearing your favourite C64 tunes as stunning, emotional orchestral masterpieces bursting from your player. Imagine yourself sitting in a concert hall overwhelmed with emotion, a tear in the corner of your eye welling up with pride that your C64 music has come this far. That's what this is about. Look below the 
  og:image: http://c64audio.com/cdn/shop/collections/symphony64_5332597f-a487-4fda-9d30-63d6a3002ddb_1200x1200.jpg?v=1542278865
  og:image:secure_url: https://c64audio.com/cdn/shop/collections/symphony64_5332597f-a487-4fda-9d30-63d6a3002ddb_1200x1200.jpg?v=1542278865
  

  The Web page main content text can contains some noise but it is still a good source of information: 
  -
  8-Bit Symphony
  Rob Hubbard
  FastLoaders
  Project Sidologie
  Back in Time
  Uncle and the Bacon
  About C64Audio.com

  1
  8-Bit Symphony
  2
  <a href="/products/8-bit-symphony-pro-first-half" class="product__image-wrapper" title="8-Bit Symphony Pro: First Half">
  <img src="//c64audio.com/cdn/shop/products/IMG_2576_grande.jpg?v=1606566955" alt="8-Bit Symphony Pro: First Half">
  </a>
  8-Bit Symphony Pro: First Half
  From £22.00
  <a href="/products/8-bit-symphony-pro-second-half-cd-digital-double-album" class="product__image-wrapper" title="8-Bit Symphony Pro: Second Half: CD preorder with immediate digital delivery">
  <img src="//c64audio.com/cdn/shop/products/8-BitSymphonyPro2_grande.png?v=1607122703" alt="8-Bit Symphony Pro: Second Half: CD preorder with immediate digital delivery">
  </a>
  8-Bit Symphony Pro: Second Half: CD preorder with immediate digital delivery
  From £22.00
  <a href="/products/8-bit-symphony-pro-first-half-blu-ray" class="product__image-wrapper" title="8-Bit Symphony Pro: First Half Surround-sound Blu-ray">
  <img src="//c64audio.com/cdn/shop/products/IMG_2564_grande.jpg?v=1606306879" alt="8-Bit Symphony Pro: First Half Surround-sound Blu-ray">
  </a>
  8-Bit Symphony Pro: First Half Surround-sound Blu-ray
  Regular price
  £28.99
  <a href="/products/8-bit-symphony-pro-second-half-surround-sound-blu-ray" class="product__image-wrapper" title="8-Bit Symphony Pro: Second Half Surround-sound Blu-ray (Pre-order)">
  <img src="//c64audio.com/cdn/shop/products/8-BitSymphonyPro2_f47816be-b9f4-4f88-b907-f3066e2b1d95_grande.png?v=1607193061" alt="8-Bit Symphony Pro: Second Half Surround-sound Blu-ray (Pre-order)">
  </a>
  8-Bit Symphony Pro: Second Half Surround-sound Blu-ray (Pre-order)
  Regular price
  £28.99
  <a href="/products/8-bit-symphony-pro-second-half-vip-package" class="product__image-wrapper" title="8-Bit Symphony Pro: VIP Packages">
  <img src="//c64audio.com/cdn/shop/products/8-BitSymphonyPro2_grande.jpg?v=1608309616" alt="8-Bit Symphony Pro: VIP Packages">
  </a>
  8-Bit Symphony Pro: VIP Packages
  From £99.00
  Choosing a selection results in a full page refresh.
  Press the space key then arrow keys to make a selection.
  """
prompts=[template]

for prompt in prompts:
    promting(llm,prompt,logging)

# Imprime la respuesta
#print(output)

# Cierra el archivo de registro
logging.shutdown()