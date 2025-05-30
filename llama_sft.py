#venv PC4 - asr3.8
import json
import os
import numpy as np
import random
import time
import datetime
import evaluate
from datasets import load_dataset, Dataset
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaModel, LlamaForSequenceClassification #, LlamaForCausalLM
from utils import file_get_contents, file_put_contents
from webapi_data import prepare_data

def messages_to_prompt(messages):
	return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def preprocess(batch):	#messages, label, event
	prompts, labels = [], []
	for i in range(len(batch["label"])):		
		messages = [ {"role":"system",  "content": gp} ]
		messages.extend(batch["messages"][i])
		label = batch["label"][i]
		prompt = messages_to_prompt(messages)
		#file_put_contents("./temp/temp.txt",  prompt + "\n\n-- label:" + label);exit()
		prompts.append(prompt)
		labels.append(label)
	return {"prompts":prompts, "labels":labels}


class myDataCollator:
    def __call__(self, features):
        input_ids, labels = [], []

        for sample in features:
            prompt = sample["prompts"]
            answer = sample["labels"]

            # Compose full text
            full = f"{prompt.strip()}{answer.strip()}<|eot_id|>"            

            full_tokens = tokenizer(full, truncation=True, max_length=3500).input_ids
            prompt_tokens = tokenizer(prompt, truncation=True, max_length=3300).input_ids

            label_ids = [-100] * len(prompt_tokens) + full_tokens[len(prompt_tokens):]

            input_ids.append(torch.tensor(full_tokens))
            labels.append(torch.tensor(label_ids))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
        print(attention_mask, input_ids.shape, labels.shape, attention_mask.shape)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}



def compute_metrics(eval_pred):	
	generated_ids, labels = eval_pred
	pred = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)	
	labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
	for p,l in zip(pred, labels):
		print(p, " -- LABEL:", l[-10:].replace("\n",""), "  -- gen_ids:", len(generated_ids[0]), "\n#==================\n")
	return {"accuracy": 1.0}



if __name__=="__main__":
	model_id = "meta-llama/Llama-3.2-1B-Instruct"  #"meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8" #"meta-llama/Llama-3.2-1B" #"Qwen/Qwen2-0.5B-Instruct"
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	tokenizer.pad_token = tokenizer.eos_token #'!' #'<|finetune_right_pad_id|>' 
	tokenizer.pad_token_id = tokenizer.eos_token_id
	tokenizer.truncation_side = 'left'
	print("tokenizer:", tokenizer.pad_token_id, tokenizer.truncation_side)

	d, gp = prepare_data()
	dataset = Dataset.from_dict(d)		
	dataset = dataset.map(preprocess, batched=True)
	dataset = dataset.train_test_split(test_size=0.03, seed=42)
	train_dataset = dataset["train"]
	test_dataset = dataset["test"]
	print("Dataset train, test sizes:",  len(train_dataset), len(test_dataset))
	
	model = AutoModelForCausalLM.from_pretrained(model_id) #"./model_temp/checkpoint-6144"
	model.config.pad_token_id = tokenizer.pad_token_id #0	
	#looping L=16/k
	model.config.num_hidden_layers=4
	model.model.layers = model.model.layers[:4]
	#device = torch.device("cuda:0")
	#model.cuda()			
	#print(model, model.config)	

	# Start training    
	data_collator = myDataCollator()
	training_args = TrainingArguments(
		output_dir='./model_temp',
		num_train_epochs=100,
		per_device_train_batch_size=1,
		gradient_accumulation_steps=1,
		#gradient_checkpointing=True, - slows down the training
		learning_rate=1e-6,
		logging_steps=20,
		save_steps=500,
		save_total_limit=2,
		load_best_model_at_end=True,
		eval_strategy="steps", #evaluation_strategy
		eval_steps=500,
		per_device_eval_batch_size=1,
		#metric_for_best_model="eval_accuracy",
		remove_unused_columns=False,
		#logging_dir="./logs/",
		#report_to="tensorboard",
		#weight_decay=0.01,
	)

	trainer = Trainer(
		model=model,
		data_collator=data_collator,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
		compute_metrics=compute_metrics
	)
	
	trainer.train() #"./model_temp/checkpoint-84000"
	#print( trainer.evaluate() )

