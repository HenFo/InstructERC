# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

login()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")


tokenizer.save_pretrained("./LLM_bases/LLaMA2-base/")
model.save_pretrained("./LLM_bases/LLaMA2-base/")