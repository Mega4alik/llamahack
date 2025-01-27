#math reasoning -- sky1 (thoughts, solution)
#https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k?row=0 - main data
#https://huggingface.co/datasets/NovaSky-AI/Sky-T1_preference_data_10k?row=0 - preference data
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


def messages_to_prompt(messages):
	return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	"""
	prompt = "<|begin_of_text|>"
	for msg in messages:
		if msg['content']:
			prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n{msg['content']}<|eot_id|>"
	#prompt+="<|start_header_id|>assistant<|end_header_id|>\n"
	return prompt
	"""


def preprocess(batch):	
	prompts, labels = [], []
	for i in range(len(batch["system"])):
		assert len(batch["conversations"][i]) == 2
		messages = [ {"role":"system",  "content": batch["system"][i]} ]
		messages.append({"role":"user", "content":batch["conversations"][i][0]["value"]})
		label = batch["conversations"][i][1]['value']
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
		required_mxlen = len(tokenizer(labels, padding='longest',  truncation=True, return_tensors='pt').input_ids[0])
		labels = tokenizer(labels, padding='longest',  max_length=5000, truncation=True, return_tensors='pt').input_ids #max_length?
		mxlen = len(labels[0])
		batch = tokenizer(prompts, padding='max_length', truncation=True, max_length=mxlen, return_tensors='pt')
		batch["labels"] = labels
		print(required_mxlen, batch["input_ids"].shape, batch["labels"].shape)
		return batch


def compute_metrics(eval_pred):		
	pred, labels = eval_pred
	input_ids = torch.argmax(torch.tensor(pred), dim=-1)
	pred = tokenizer.batch_decode(input_ids, skip_special_tokens=False)
	labels = [[token if token != -100 else tokenizer.pad_token_id for token in seq] for seq in labels]
	labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
	print(pred[0]) #, "\n\n -- labels:", labels[0]
	return {"accuracy": 1.0}



if __name__=="__main__":
	model_id = "Qwen/Qwen2-0.5B-Instruct"   #"meta-llama/Llama-3.2-1B-Instruct"  #"meta-llama/Llama-3.2-1B-Instruct-QLORA_INT4_EO8" #"meta-llama/Llama-3.2-1B"
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	#tokenizer.pad_token = tokenizer.eos_token #'!' #'<|finetune_right_pad_id|>' 
	tokenizer.truncation_side = 'left'
	print("tokenizer:", tokenizer.pad_token_id, tokenizer.truncation_side)

	dataset = load_dataset("NovaSky-AI/Sky-T1_data_17k")
	dataset = dataset.map(preprocess, batched=True)
	dataset = dataset["train"].train_test_split(test_size=0.0002)
	train_dataset = dataset["train"]
	test_dataset = dataset["test"]
	print("Dataset train, test sizes:",  len(train_dataset), len(test_dataset))

	model = AutoModelForCausalLM.from_pretrained(model_id) #"./model_temp/checkpoint-6144"
	model.config.pad_token_id = tokenizer.pad_token_id #0	
	#device = torch.device("cuda:0")
	#model.cuda()			

	# Start training    
	data_collator = myDataCollator()
	training_args = TrainingArguments(
		output_dir='./model_temp',
		num_train_epochs=3,
		per_device_train_batch_size=1,#2,
		gradient_accumulation_steps=8,
		#gradient_checkpointing=True, - slows down the training
		learning_rate=1e-6,
		logging_steps=20,
		save_steps=500,
		save_total_limit=3,
		load_best_model_at_end=True,
		evaluation_strategy="steps",
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

