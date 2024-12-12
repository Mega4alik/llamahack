import json
import random
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from scipy import spatial
from typing import Any, Dict, List, Optional, Union, Tuple
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from safetensors.torch import load_file
import preprocess as pp

#=======================================================
class ChunkQuestionTripletDataset(Dataset):
    def __init__(self, chunks, questions, tokenizer, max_length=4000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        num_chunks = len(chunks)
        for idx, chunk in enumerate(chunks):
            anchor = chunk
            positive_questions = questions[idx]            
            for positive in positive_questions:                
                negative_indices = list(range(num_chunks))
                negative_indices.remove(idx)
                for _ in range(3): #for each positive, pair 3 negatives
                    negative_idx = random.choice(negative_indices)
                    negative_indices.remove(negative_idx)
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


#=======================================================
class SiameseModel(nn.Module):
    def __init__(self, model_name=None):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
        self.encoder = AutoModel.from_pretrained(model_name) if model_name else None

    def mean_pooling(self, token_embeddings, attention_mask):
        # Mean Pooling - Take attention mask into account
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


    def forward(self, **inputs):
        # Encode anchor
        outputs_anchor = self.encoder(input_ids=inputs['input_ids_anchor'], attention_mask=inputs['attention_mask_anchor'])
        anchor_embedding = F.normalize(self.mean_pooling(outputs_anchor.last_hidden_state, inputs['attention_mask_anchor']))
        #print(anchor_embedding.shape, inputs["attention_mask_anchor"])
        
        # Encode positive
        outputs_positive = self.encoder(input_ids=inputs['input_ids_positive'], attention_mask=inputs['attention_mask_positive'])
        positive_embedding = F.normalize(self.mean_pooling(outputs_positive.last_hidden_state,inputs['attention_mask_positive']))

        # Encode negative
        outputs_negative = self.encoder(input_ids=inputs['input_ids_negative'], attention_mask=inputs['attention_mask_negative'])
        negative_embedding = F.normalize(self.mean_pooling(outputs_negative.last_hidden_state, inputs['attention_mask_negative']))

        loss = self.loss_fn(anchor_embedding, positive_embedding, negative_embedding)
        return {'loss': loss}

    
    def generate(self, texts):
        inputs = tokenizer(texts, return_tensors='pt').to(device)
        outputs = self.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        embeddings = F.normalize(self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])).detach().cpu().numpy()
        return embeddings


#=======================================================
def cosine_similarity(v1, v2):
  return 1 - spatial.distance.cosine(v1, v2)

def get_embedding(text):
    with torch.no_grad():
        embedding = siamese_model.generate([text])[0]
    return embedding

def prepare_dataset():  #questions - [[Q1_1,Q1_2,...]]
    chunks, questions = [], []
    d = json.loads(pp.file_get_contents("./temp/chunks3.json"))
    for key in d:
        a = d[key]
        for chunk in a:
            text = "#"+chunk["keywords"]+"\n"+chunk["content"]
            chunks.append(text)
            questions.append( ["Q: "+question for question in chunk["questions_list"]] )
    return (chunks, questions)

def test():
    chunks, questions = prepare_dataset()
    n, chunks_emb = len(chunks), []
    for i in range(n): chunks_emb.append( get_embedding(chunks[i]) )
    for i in range(n):
        for question in questions[i]:
            qe, a = get_embedding(question), []
            for j in range(n): a.append((j, cosine_similarity(qe, chunks_emb[j])))
            a = sorted(a, key=lambda x: x[1], reverse=True)
            top = [x[0] for x in a[:5]]
            print(i, "--", top, ("YES" if i in top else "NO") )

    
# =================================================

model_name = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)

if 1==2: #evaluate trained model
    device = torch.device("cuda:0")
    siamese_model = SiameseModel(model_name=model_name)
    #state_dict = load_file("./model_temp/checkpoint-6600/model.safetensors")
    #siamese_model.load_state_dict(state_dict)
    siamese_model.cuda()
    siamese_model.eval()
    test()
else: #train
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
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=siamese_model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    