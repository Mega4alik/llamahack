#math reasoning -- reward + decoder?
import json
import os
import numpy as np
import random
import time
import datetime

import evaluate
from datasets import load_dataset #, load_metric
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler #, Dataset
from datasets import Dataset
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaModel, LlamaForSequenceClassification #, LlamaForCausalLM

from utils import file_get_contents, file_put_contents


def messages_to_prompt(messages):
	prompt = "<|begin_of_text|>"
	for msg in messages:
		if msg['content']:
			prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n{msg['content']}<|eot_id|>"
	#prompt+="<|start_header_id|>assistant<|end_header_id|>\n"
	return prompt



def preprocess(batch):
	prompts, labels = [], []
	for i in range(len(batch["system"])):
		messages = [ {"role":"system",  "content": batch["system"][i]} ]
		messages.append({"role":"user", "content":batch["conversations"][i][0]["value"]})
		label = batch["conversations"][i][-1]['value']
		prompt = messages_to_prompt(messages)
		#file_put_contents("./temp/temp.txt",  prompt + "\n\n-- label:" + label)
		prompts.append(prompt)
		labels.append(label)
	return {"prompts":prompts, "labels":labels}


class myDataCollator:
	def __call__(self, features):
		prompts = [x["prompts"] for x in features]
		labels = [x["labels"] for x in features]
		#print(prompts, labels, "\n\n")
		batch = tokenizer(prompts, padding=True, truncation=True, max_length=1600, return_tensors='pt')
		labels = tokenizer(prompts, padding=True, truncation=True, max_length=1600, return_tensors='pt').input_ids
		batch["labels"] = labels
		return batch


def compute_metrics(eval_pred):	
	pred, labels = eval_pred
	print(pred[:3], labels[:3])
	return {"accuracy": 1.0}


if __name__=="__main__":
	model_id =  "meta-llama/Llama-3.2-1B-Instruct"  #"meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8" #"meta-llama/Llama-3.2-1B"
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	tokenizer.pad_token = tokenizer.eos_token	
	tokenizer.truncation_side = 'left'
	print("tokenizer:", tokenizer.pad_token_id, tokenizer.truncation_side)

	dataset = load_dataset("NovaSky-AI/Sky-T1_data_17k")
	dataset = dataset.map(preprocess, batched=True)
	dataset = dataset["train"].train_test_split(test_size=0.001)
	train_dataset = dataset["train"]
	test_dataset = dataset["test"]
	print("Dataset train, test sizes:",  len(train_dataset), len(test_dataset))
	
	model = AutoModelForCausalLM.from_pretrained(model_id)
	model.config.pad_token_id = tokenizer.pad_token_id #0	
	#device = torch.device("cuda:0")
	#model.cuda()			

	# Start training    
	data_collator = myDataCollator()
	training_args = TrainingArguments(
		output_dir='./model_temp',
		num_train_epochs=3,
		per_device_train_batch_size=1, #4
		gradient_accumulation_steps=1,
		#gradient_checkpointing=True, - slows down the training
		learning_rate=1e-6,
		logging_steps=100,
		save_steps=2000,
		save_total_limit=3,
		load_best_model_at_end=True,
		evaluation_strategy="steps",
		eval_steps=2000,
		per_device_eval_batch_size=1,
		metric_for_best_model="eval_accuracy",
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

