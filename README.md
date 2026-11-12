# DistilBERT Fine-Tuning: Customer Review Sentiment Classification

Fine-tuning `distilbert-base-uncased` (Hugging Face Transformers) to classify customer reviews into three sentiment classes: **Positive**, **Neutral** and **Negative**.

## Overview

| Item | Details |
|---|---|
| Base model | `distilbert-base-uncased` |
| Task | Text classification (3-class sentiment) |
| Framework | Hugging Face Transformers, PyTorch |
| Dataset | Customer reviews CSV (generated with ChatGPT) |
| Environment | Python virtual environment (venv) |

## Project Structure

```
DistilBERT_FineTuning/
├── download_model.py    
├── inspect_dataset.py    
├── clean_dataset.py     
├── train.py             
├── test_model.py        
├── requirements.txt   
└── README.md
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate  
```

2. Install dependencies:

```bash
pip install transformers datasets torch scikit-learn pandas accelerate
```

## How to Run

Run the scripts in this order:

```bash
python download_model.py    
python inspect_dataset.py   
python clean_dataset.py     
python train.py             
python test_model.py       
```

## Workflow

1. **Download model**: Load `distilbert-base-uncased` and its tokenizer from Hugging Face.
2. **Inspect dataset**: Review the CSV structure, label distribution and sample rows.
3. **Clean dataset**: Remove duplicates and null values, and make sure labels are consistent.
4. **Tokenize**: Convert review text into token IDs using the DistilBERT tokenizer, with padding and truncation.
5. **Train**: Fine-tune the model with a classification head for 3 labels, using the Hugging Face `Trainer`.
6. **Evaluate**: Measure accuracy and F1-score on a held-out test split.
7. **Test**: Predict sentiment for new, unseen reviews.

## What Is Fine-Tuning?

DistilBERT is already pre-trained on a large amount of general English text, so it understands language structure. Fine-tuning continues training on a small, task-specific dataset (here, customer reviews) so the model learns to map review text to sentiment labels. This needs far less data and compute than training from scratch.

## Example Usage

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="./sentiment_model")

print(classifier("The product quality is excellent and delivery was fast."))
# [{'label': 'Positive', 'score': 0.98}]
```

## Results

| Metric | Value |
|---|---|
| Accuracy | _add your value_ |
| F1-score | _add your value_ |

## Limitations

- The dataset is ChatGPT-generated, so it may not capture the variety of real customer reviews.
- The model is trained on a small dataset; results on real-world data may differ.
- The Neutral class is typically the hardest to separate from Positive and Negative.

## Author

**ArthiShri T.**
LinkedIn: [linkedin.com/in/arthishri-thiruppathi](https://linkedin.com/in/arthishri-thiruppathi)
