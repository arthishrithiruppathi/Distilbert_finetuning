import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments)

df = pd.read_csv("dataset/customer_reviews_clean.csv")
label_mapping = {"negative": 0,"neutral": 1,"positive": 2}

df["label"] = df["sentiment"].map(label_mapping)
train_df, temp_df = train_test_split(df,test_size=0.2,random_state=42,stratify=df["label"])
val_df, test_df = train_test_split(temp_df,test_size=0.5,random_state=42,stratify=temp_df["label"])

model_path = "./model/fine_tuned_distilbert"

tokenizer = AutoTokenizer.from_pretrained(model_path)
test_dataset = Dataset.from_pandas(test_df[["review", "label"]],preserve_index=False)

def tokenize_function(examples):
    return tokenizer(examples["review"],truncation=True,padding="max_length",max_length=128)

test_dataset = test_dataset.map(tokenize_function,batched=True,remove_columns=["review"])

test_dataset = test_dataset.rename_column("label", "labels")

model = AutoModelForSequenceClassification.from_pretrained(model_path)

training_args = TrainingArguments(output_dir="./test_results",per_device_eval_batch_size=8,report_to="none")

trainer = Trainer(model=model,args=training_args,processing_class=tokenizer)

results = trainer.predict(test_dataset)

predictions = np.argmax(results.predictions, axis=1)

actual_labels = results.label_ids

print("\nCLASSIFICATION REPORT")
print(classification_report(actual_labels,predictions,labels=[0, 1, 2],target_names=["Negative", "Neutral", "Positive"],zero_division=0))

print("\nCONFUSION MATRIX")
print(confusion_matrix( actual_labels,predictions, labels=[0, 1, 2]))
