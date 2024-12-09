import json
import random
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
import torch.nn as nn
import torch.nn.functional as F
import preprocess as pp

# Create a custom Dataset for Triplet Loss
class ChunkQuestionTripletDataset(Dataset):
    def __init__(self, chunks, questions, tokenizer, max_length=4000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        num_chunks = len(chunks)
        for idx, chunk in enumerate(chunks):
            anchor = chunk
            positive_questions = questions[idx]
            # For each positive question, create a triplet
            for positive in positive_questions:
                # Sample a negative question from other chunks
                negative_indices = list(range(num_chunks))
                negative_indices.remove(idx)
                negative_idx = random.choice(negative_indices)
                negative_questions = questions[negative_idx]
                negative = random.choice(negative_questions)
                self.samples.append((anchor, positive, negative))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        anchor_text, positive_text, negative_text = self.samples[idx]        
        x =  {
            'anchor_text': anchor_text,
            'positive_text': positive_text,
            'negative_text': negative_text,
        }        
        return x


class DataCollatorForTripletLoss:
    def __init__(self, tokenizer, max_length=4000):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, features):
        #print("features:", features)
        anchor_texts = [f['anchor_text'] for f in features]
        positive_texts = [f['positive_text'] for f in features]
        negative_texts = [f['negative_text'] for f in features]
        
        # Tokenize and pad the sequences
        anchor_encodings = self.tokenizer(
            anchor_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        positive_encodings = self.tokenizer(
            positive_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        negative_encodings = self.tokenizer(
            negative_texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        
        batch = {
            'input_ids_anchor': anchor_encodings['input_ids'],
            'attention_mask_anchor': anchor_encodings['attention_mask'],
            'input_ids_positive': positive_encodings['input_ids'],
            'attention_mask_positive': positive_encodings['attention_mask'],
            'input_ids_negative': negative_encodings['input_ids'],
            'attention_mask_negative': negative_encodings['attention_mask'],
        }        
        return batch



# Define the Siamese network model with Triplet Loss
class SiameseModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
        self.encoder = AutoModel.from_pretrained(model_name)

    def mean_pooling(self, token_embeddings, attention_mask):
        # Mean Pooling - Take attention mask into account
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, **inputs):        
        # Encode anchor
        outputs_anchor = self.encoder(
            input_ids=inputs['input_ids_anchor'],
            attention_mask=inputs['attention_mask_anchor'],
        )
        anchor_embedding = self.mean_pooling(
            outputs_anchor.last_hidden_state,
            inputs['attention_mask_anchor'],
        )

        # Encode positive
        outputs_positive = self.encoder(
            input_ids=inputs['input_ids_positive'],
            attention_mask=inputs['attention_mask_positive'],
        )
        positive_embedding = self.mean_pooling(
            outputs_positive.last_hidden_state,
            inputs['attention_mask_positive'],
        )

        # Encode negative
        outputs_negative = self.encoder(
            input_ids=inputs['input_ids_negative'],
            attention_mask=inputs['attention_mask_negative'],
        )
        negative_embedding = self.mean_pooling(
            outputs_negative.last_hidden_state,
            inputs['attention_mask_negative'],
        )

        loss = self.loss_fn(anchor_embedding, positive_embedding, negative_embedding)
        return {'loss': loss}        


def prepare_dataset():    
    chunks, questions = [], []
    d = json.loads(pp.file_get_contents("./temp/chunks3.json"))
    for key in d:
        a = d[key]
        for chunk in a:
            text = "#"+chunk["keywords"]+"\n"+chunk["content"]
            chunks.append(text)
            questions.append(chunk["questions_list"])
    return (chunks, questions)


# =================================================

# Initialize tokenizer and models
random.seed(42)
torch.manual_seed(42)
model_name = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
siamese_model = SiameseModel(model_name=model_name)

# Create the dataset
"""
chunks = [
    "This is the text of chunk1. It is about topic A.",
    "Here is chunk2. It discusses topic B.",
    "Text of chunk3 goes here. Topic C is covered.",
]

questions = [
    ["What is topic A?", "Explain topic A in detail."],
    ["Can you tell me about topic B?", "What does chunk2 discuss?"],
    ["Give me information on topic C.", "Describe the subject of chunk3."],
]
"""
chunks, questions = prepare_dataset()
train_dataset = ChunkQuestionTripletDataset(chunks, questions, tokenizer)
print( len(train_dataset.samples) )


# Start training
data_collator = DataCollatorForTripletLoss(tokenizer=tokenizer)

training_args = TrainingArguments(
    output_dir='./model_temp',
    num_train_epochs=30,
    per_device_train_batch_size=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=1,
    save_steps=3,
    remove_unused_columns=False
)

trainer = Trainer(
    model=siamese_model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()