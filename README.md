# alpaca-classification
-
base :

- [llama-clasification](https://github.com/sh0416/llama-classification)
- [llama](https://github.com/facebookresearch/llama)


create even :

```bash

conda create -y -n alpaca-classification python=3.8
# activate evn:

conda activate llama-classification 
# or
source activate alpaca-classification

# (optional, if don't have cuda in your host)
conda install cudatoolkit=11.7 -y -c nvidia
conda list cudatoolkit # to check what cuda version is installed (11.7)

# install requirements
pip3 install -r requirements.txt

```


## Train 

```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .

huggingface-cli login --token YOUR_TOKEN
```
[models_weigth](https://huggingface.co/meta-llama)

[pasos](https://github.com/facebookresearch/llama-recipes/blob/main/docs/Dataset.md#training-on-custom-data)

## Llama cpp short

[low_models_weigth](https://huggingface.co/TheBloke/Llama-2-7B-GGUF)
[llama cpp](https://github.com/abetlen/llama-cpp-python)

Pasos

```bash
git clone https://github.com/abetlen/llama-cpp-python.git

cd llama-cpp-python

conda create --name alpaca python=3.11
conda activate alpaca

pip install llama-cpp-python

#aceleration Hardware

CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS -DLLAMA_CUBLAS=on -DLLAMA_CLBLAST=on -DLLAMA_METAL=on -DLLAMA_HIPBLAS=on" pip install llama-cpp-python

```

opcional 

```bash
pip3 install jupyter
```

link

https://promptengineering.org/how-does-llama-2-compare-to-gpt-and-other-ai-language-models/

https://github.com/Aschen/web-classification

https://www.mlexpert.io/machine-learning/tutorials/alpaca-fine-tuning

https://github.com/georgian-io/LLM-Finetuning-Hub

https://www.maartengrootendorst.com/blog/improving-llms/
https://medium.com/unstructured-io/setting-up-a-private-retrieval-augmented-generation-rag-system-with-local-vector-database-d42f34692ca7

## Citation

It would be welcome citing my work if you use my codebase for your research.



```
@software{Lee_Simple_Text_Classification_2023,
    author = {Lee, Seonghyeon},
    month = {3},
    title = {{Simple Text Classification Codebase using LLaMA}},
    url = {https://github.com/github/sh0416/llama-classification},
    version = {1.1.0},
    year = {2023}
}
```