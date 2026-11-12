import os
from transformers import AutoTokenizer, AutoModel

model_name = "distilbert-base-uncased"

save_directory = "./model/pretrained_distilbert"

os.makedirs(save_directory, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModel.from_pretrained(model_name)

tokenizer.save_pretrained(save_directory)

model.save_pretrained(save_directory)

print("DistilBERT model downloaded successfully!")
print("Model saved at:", save_directory)