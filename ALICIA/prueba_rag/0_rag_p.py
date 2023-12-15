

from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embed_model_id = 'sentence-transformers/all-MiniLM-L6-v2'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

embed_model = HuggingFaceEmbeddings(
    model_name=embed_model_id,
    model_kwargs={'device': device},
    encode_kwargs={'device': device, 'batch_size': 32}
)
     

import os
import pinecone

# get API key from app.pinecone.io and environment from console
PINECONE_API_KEY='bc795b6a-d0aa-4877-bdc6-414135eb4bef'
PINECONE_ENV='gcp-starter'
pinecone.init(
    api_key=os.environ.get('PINECONE_API_KEY') or PINECONE_API_KEY,
    environment=os.environ.get('PINECONE_ENVIRONMENT') or PINECONE_ENV
)
     
import time

index_name = 'llama-2-rag'

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=len(embeddings[0]),
        metric='cosine'
    )
    # wait for index to finish initialization
    while not pinecone.describe_index(index_name).status['ready']:
        time.sleep(1)
else:
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

from datasets import load_dataset

data = load_dataset(
    'jamescalam/llama-2-arxiv-papers-chunked',
    split='train'
)

data = data.to_pandas()

batch_size = 32

for i in range(0, len(data), batch_size):
    i_end = min(len(data), i+batch_size)
    batch = data.iloc[i:i_end]
    ids = [f"{x['doi']}-{x['chunk-id']}" for i, x in batch.iterrows()]
    texts = [x['chunk'] for i, x in batch.iterrows()]
    embeds = embed_model.embed_documents(texts)
    # get metadata to store in Pinecone
    metadata = [
        {'text': x['chunk'],
         'source': x['source'],
         'title': x['title']} for i, x in batch.iterrows()
    ]
    # add to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))


model_path = "./llama_models/llama2-chat-ayb-13b.Q5_K_M.gguf"
from langchain.vectorstores import Pinecone

text_field = 'text'  # field in metadata that contains text content

vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)

query = 'what makes llama 2 special?'

vectorstore.similarity_search(
    query,  # the search query
    k=3  # returns top 3 most relevant chunks of text
)

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

llm = LlamaLLM(model_path=model_path,n_ctx=4096)


from langchain.chains import RetrievalQA

rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm, chain_type='stuff',
    retriever=vectorstore.as_retriever()
)


rag_pipeline('what is so special about llama 2?')