import os
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
max_token_len = 250

def process(examples):
    tokenized_data = tokenizer(text=examples['document'], padding="max_length", max_length=max_token_len, return_tensors="pt")
    tokenized_labels = tokenizer(text=examples['summary'], padding="max_length", max_length=max_token_len, return_tensors="pt")
    
    tokens = {
        'input_ids': tokenized_data['input_ids'], 
        'input_attention_mask': tokenized_data['attention_mask'].bool(), 
        'label_ids': tokenized_labels['input_ids'],
        'label_attention_mask': tokenized_labels['attention_mask'].bool()
    }
    
    return tokens


if __name__ == "__main__":
    num_proc = 4
    dataset = load_dataset("gigaword", num_proc=num_proc)

    tokenized = dataset.map(
        process,
        remove_columns=['document', 'summary'],
        desc="tokenizing the splits",
        num_proc=num_proc, 
        batched=True
    )

    dset_filename = os.path.join(os.path.dirname(__file__), f'dataset')
    tokenized.save_to_disk(dset_filename)