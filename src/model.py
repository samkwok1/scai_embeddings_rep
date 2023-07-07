#model.py

from transformers import AutoModelForCausalLM, AutoTokenizer

#Simplest Enlgish Tokenizer
model_1 = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-1M')
tokenizer_1 = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
