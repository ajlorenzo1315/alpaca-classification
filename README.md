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