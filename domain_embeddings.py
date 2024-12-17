#sources: PC4 - asr_p3.8, US1 - asr3.12
import json
import random
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass, field
#from typing import Any, Dict, List, Optional, Union, Tuple
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments
from safetensors.torch import load_file

from utils import file_get_contents, cosine_similarity, myOpenAI


#=======================================================
class ChunkQuestionTripletDataset(Dataset):    
    def __init__(self, chunks, questions, tokenizer, max_length=4000):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.test_samples = []
        num_chunks = len(chunks)
        for idx, chunk in enumerate(chunks):
            anchor = chunk
            positive_questions = questions[idx]                                
            self.test_samples.append((anchor, positive_questions[0], questions[self.randint_excluding_idx(num_chunks-1, idx)][0])) #each first positive_question goes to test set

            for positive in positive_questions[1:]:
                negative_indices = list(range(num_chunks))
                negative_indices.remove(idx)
                for _ in range(5): #for each positive, pair 3 negatives
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

    def randint_excluding_idx(self, n, idx):        
        rand = random.randint(0, n - 1)
        return rand if rand < idx else rand + 1


class TestDataset(ChunkQuestionTripletDataset):
    def __init__(self, test_samples):
        self.samples = test_samples

#=======================================================

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


class SiameseModel(nn.Module):
    def __init__(self, model_name=None):
        super().__init__()
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
        
        # Encode positive
        outputs_positive = self.encoder(input_ids=inputs['input_ids_positive'], attention_mask=inputs['attention_mask_positive'])
        positive_embedding = F.normalize(self.mean_pooling(outputs_positive.last_hidden_state,inputs['attention_mask_positive']))

        # Encode negative
        outputs_negative = self.encoder(input_ids=inputs['input_ids_negative'], attention_mask=inputs['attention_mask_negative'])
        negative_embedding = F.normalize(self.mean_pooling(outputs_negative.last_hidden_state, inputs['attention_mask_negative']))

        loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)
        return {'loss': loss}
    
    def generate(self, texts):
        inputs = tokenizer(texts, return_tensors='pt').to(device)
        outputs = self.encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        embeddings = F.normalize(self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])).detach().cpu().numpy()
        return embeddings

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)


#=======================================================
class JinaDataCollator:
    def __init__(self):
        pass        

    def __call__(self, features):
        anchor_texts = [f['anchor_text'] for f in features]
        positive_texts = [f['positive_text'] for f in features]
        negative_texts = [f['negative_text'] for f in features]
        batch = {
            'anchor_texts': anchor_texts,
            'positive_texts': positive_texts,
            'negative_texts': negative_texts
        }
        return batch


class JinaModel(nn.Module):
    def __init__(self, model_name=None):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_flash_attn=True)
        #self.encoder.config.use_flash_attn = False
        #self.encoder.roberta.config.use_flash_attn = False
        for param in self.encoder.roberta.parameters():
            param.requires_grad = True


    def encode(self, input_ids, attention_mask):
        #with torch.enable_grad(): embeddings = self.encoder.encode(texts, task="retrieval.passage", convert_to_tensor=True)
        with torch.enable_grad():
            #encoded_input = self.encoder.roberta.tokenizer(texts, return_tensors="pt") #.to(device)
            token_embs = self.encoder.roberta.forward(input_ids) 
            embeddings = self.encoder.roberta.mean_pooling(token_embs.last_hidden_state, attention_mask)
            embeddings = F.normalize(embeddings, p=2, dim=0)                
        return embeddings

    def forward(self, **inputs): #modeling_xlm_roberta comment @torch.inference_mode() on encode function
        anchor_embedding = self.encode(inputs['input_ids_anchor'], inputs['attention_mask_anchor']) 
        positive_embedding = self.encode(inputs['input_ids_positive'], inputs['attention_mask_positive'])
        negative_embedding = self.encode(inputs['input_ids_negative'], inputs['attention_mask_negative'])
        loss = loss_fn(anchor_embedding, positive_embedding, negative_embedding)
        #if self.training:
        return {'loss': loss}
        

    @torch.inference_mode()
    def generate(self, texts, task):
        encoded_input = tokenizer(texts, return_tensors="pt").to(device)
        embeddings = self.encode(encoded_input['input_ids'], encoded_input['attention_mask']).detach().cpu().numpy()
        #embeddings = self.encoder.encode(texts, task=task)
        return embeddings

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

#=======================================================

class OwnTrainer(Trainer):
    @torch.inference_mode()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        outputs = []
        for step, inputs in enumerate(eval_dataloader):
            out = self.model.forward(**inputs) #**input_ids=<>, attention_mast=<>
            x = out["loss"].item()
            #print("===eval:", x, out)
            outputs.append(x)

        average = sum(outputs) / len(outputs)
        r = {"eval_loss":average}
        print(r)
        return r

#=======================================================

def get_embedding(text, task='retrieval.query'):
    embedding = siamese_model.generate([text], task)[0]
    #embedding = opai.get_embedding(text)
    return embedding

def prepare_dataset():  #questions - [[Q1_1,Q1_2,...]]
    chunks, questions = [], []
    d = json.loads(file_get_contents("./temp/chunks3.json"))
    for key in d:
        a = d[key]
        for chunk in a:
            text = "#"+chunk["keywords"]+"\n"+chunk["content"]
            chunks.append(text)
            questions.append( ["Q: "+question for question in chunk["questions_list"]] )
    return (chunks, questions)

def test():
    chunks, questions = prepare_dataset()
    dataset = ChunkQuestionTripletDataset(chunks, questions, tokenizer)

    n, chunks_emb, yes_count = len(chunks), [], 0
    for i in range(n): chunks_emb.append( get_embedding(chunks[i], task='retrieval.passage') )

    for i in range(n):
        _, question, _  = dataset.test_samples[i]
        idx = i
        qe, a = get_embedding(question), []
        for j in range(n): a.append((j, cosine_similarity(qe, chunks_emb[j])))
        a = sorted(a, key=lambda x: x[1], reverse=True)
        top = [x[0] for x in a[:5]]
        if idx in top: yes_count+=1        
        print(idx, "--", top, ("YES" if idx in top else "NO") )        
    print(f"YES: {yes_count}, NO: {n-yes_count}")
    
# =================================================

model_name = "jinaai/jina-embeddings-v3"  #"Alibaba-NLP/gte-Qwen2-1.5B-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

if 1==1: #=== Evaluate trained model ===
    device = torch.device("cuda:0")
    siamese_model = JinaModel(model_name=model_name)
    #state_dict = load_file("./models/domain_emb_jina1/checkpoint-3500/model.safetensors")
    #siamese_model.load_state_dict(state_dict)
    siamese_model.cuda()
    siamese_model.eval()
    opai = myOpenAI()
    test()
else: #=== TRAIN ===
    # Create the dataset
    chunks, questions = prepare_dataset()
    train_dataset = ChunkQuestionTripletDataset(chunks, questions, tokenizer)
    test_dataset = TestDataset(train_dataset.test_samples)
    print("train, test len:", len(train_dataset.samples), len(train_dataset.test_samples))

    # Start training
    siamese_model = JinaModel(model_name=model_name) #SiameseModel     
    data_collator = DataCollatorForTripletLoss(tokenizer=tokenizer) #JinaDataCollator()
    loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

    training_args = TrainingArguments(
        output_dir='./model_temp',
        num_train_epochs=30,
        per_device_train_batch_size=1, #16,
        gradient_accumulation_steps=8,
        #gradient_checkpointing=True, #default False - slows down the training        
        learning_rate=1e-6,
        logging_steps=20,
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_eval_batch_size=1,
        metric_for_best_model='eval_loss',
        remove_unused_columns=False,
        logging_dir="./logs/",
        report_to="tensorboard",
        #weight_decay=0.01,
    )

    trainer = OwnTrainer(
        model=siamese_model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset    
    )

    trainer.train()
    