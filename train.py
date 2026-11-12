import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

# 1. Load cleaned data and convert sentiment words to numbers
df = pd.read_csv("dataset/customer_reviews_clean.csv")
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {v: k for k, v in label2id.items()}
df["label"] = df["sentiment"].map(label2id)

# 2. Split: 80% train, 10% validation, 10% test (same settings as test_model.py)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

# 3. Tokenize
model_path = "./model/pretrained_distilbert"
tokenizer = AutoTokenizer.from_pretrained(model_path)

def to_dataset(frame):
    ds = Dataset.from_pandas(frame[["review", "label"]], preserve_index=False)
    ds = ds.map(lambda x: tokenizer(x["review"], truncation=True,
                                    padding="max_length", max_length=128),
                batched=True, remove_columns=["review"])
    return ds.rename_column("label", "labels")

train_dataset = to_dataset(train_df)
val_dataset = to_dataset(val_df)

# 4. Load DistilBERT with a new 3-class classification layer
model = AutoModelForSequenceClassification.from_pretrained(
    model_path, num_labels=3, id2label=id2label, label2id=label2id)

# 5. Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average="macro")}

# 6. Train
args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to="none",
)

trainer = Trainer(model=model, args=args, train_dataset=train_dataset,
                  eval_dataset=val_dataset, processing_class=tokenizer,
                  compute_metrics=compute_metrics)
trainer.train()

# 7. Save
trainer.save_model("./model/fine_tuned_distilbert")
tokenizer.save_pretrained("./model/fine_tuned_distilbert")
print("Fine-tuned model saved.")