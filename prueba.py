from transformers import AutoTokenizer

model_id = "meta-llama/Llama-2-13b-hf" # sharded weights
tokenizer = AutoTokenizer.from_pretrained(model_id,use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token