# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

login()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")


tokenizer.save_pretrained("./LLM_bases/LLaMA/")
model.save_pretrained("./LLM_bases/LLaMA/")