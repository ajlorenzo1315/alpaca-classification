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
        return response
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_path": self.model_path}


model_path = "./llama_models/llama2-chat-ayb-13b.Q5_K_M.gguf"
print("Modelo: %s", model_path)
# Load the model

llm = LlamaLLM(model_path=model_path)


# Basic Q&A
answer = llm(   "Question: What is the capital of France? Answer: ", stop=["Question:", "\n"])
print(f"Answer: {answer.strip()}")

# Using in a chain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["product"],
    template="\n\n### Instruction:\nWrite a good name for a company that makes {product}\n\n### Response:\n",
)
#chain = LLMChain(llm=llm, prompt=prompt)

# Run the chain only specifying the input variable.
#print(chain.run("colorful socks"))


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
