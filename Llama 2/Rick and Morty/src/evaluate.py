# evaluate.py
import torch
from datasets import load_metric
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Function to calculate perplexity
def calculate_perplexity(model, tokenizer, dataset, device, batch_size=2, max_length=100):
    model.to(device)
    model.eval()
    total_loss = 0
    total_length = 0

    # DataLoader for batch processing
    loader = DataLoader(dataset, batch_size=batch_size)

    for batch in loader:
        inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        # Create a mask for non-padding tokens
        non_pad_mask = input_ids != tokenizer.pad_token_id

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss.item()
            loss = loss * non_pad_mask.float()
            loss = loss.sum() / non_pad_mask.sum()

        total_loss += loss * input_ids.size(0)
        total_length += input_ids.size(0)

    average_loss = total_loss / total_length
    perplexity = torch.exp(torch.tensor(average_loss))
    return perplexity.item()

# Function to calculate BLEU score
def calculate_bleu_score(model, tokenizer, dataset, device, batch_size=2, max_length=100):
    bleu = load_metric("bleu")
    model.to(device)
    model.eval()

    predictions = []
    references = []

    # DataLoader for batch processing
    loader = DataLoader(dataset, batch_size=batch_size)

    for batch in loader:
        inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=max_length)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        with torch.no_grad():
            outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=max_length)
        
        # Decode generated tokens to texts
        gen_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ref_texts = [text.split() for text in batch['text']]
        
        predictions.extend(gen_texts)
        references.extend([[ref] for ref in ref_texts])  # BLEU expects a list of list for references

    return bleu.compute(predictions=predictions, references=references)["bleu"]
