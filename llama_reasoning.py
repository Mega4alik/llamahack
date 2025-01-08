#math reasoning -- reward + decoder?
import json
import os
import numpy as np
import soundfile as sf
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
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaModel, LlamaForSequenceClassification #, LlamaForCausalLM

from utils import file_get_contents
#from transformers1.models.llama.modeling_llama import LlamaForCausalLM
#from prm800k.grading import grader


def messages_to_prompt(messages):  
	prompt = "<|begin_of_text|>"
	for msg in messages:
			prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n{msg['content']}<|eot_id|>"
	#prompt+="<|start_header_id|>assistant<|end_header_id|>\n"
	return prompt


def prm800k_preprocess():
	lines = file_get_contents("./prm800k/data/phase2_train.jsonl").split("\n")
	data = []
	for line in lines[:-1]:
		x, a = json.loads(line), []
		for step in x["label"]["steps"]:
			for completion in step["completions"]:
				data.append( {"question":x["question"], "steps": a + [completion["text"]], "rating":completion["rating"]} ) #
			if step["chosen_completion"] is None: break 
			chosen_completion = step["completions"][step["chosen_completion"]]
			a.append(chosen_completion["text"])

	print( "dataset size = ", len(data) )
	return data


def data_to_prompt(data):
	prompts, labels = [], []
	for x in data:
		steps = x["steps"]
		st = ""
		for i in range(len(steps)-1): st+=f"Step {i+1}: {steps[i]}\n"
		st+=f"Last step: {steps[-1]}"
		messages = [
			{"role":"system", "content":"Given a math question and step-by-step solution, grade the last step."},
			{"role":"user", "content":x["question"]["problem"]},
			{"role":"assistant", "content":st}
		]
		prompt = messages_to_prompt(messages)
		prompts.append(prompt)
		labels.append(2 if x["rating"]==-1 else x["rating"] ) #normalize 2 - as -1
		#print(x, "\n", prompt)

	return {"prompts":prompts, "labels":labels}


def reward_preprocess(batch):	
	return tokenizer(
			batch["prompts"],
			#padding="max_length",
			truncation=True,
			max_length=256,
	)


class prm800kRewardDataset(Dataset):
	def __init__(self, data):		
		self.samples = data
		
	def __len__(self):
		return len(self.samples)
	
	def __getitem__(self, idx):
		x = self.samples[idx]
		return x



if __name__=="__main__":
	#print( grader.grade_answer("Right. So the integer $n$ that we're looking for is $-48$.\n\n# Answer\n\n-48", "-48") )	
	data = prm800k_preprocess()[-500:] #500000
	#dict of prompts, labels
	dataset = Dataset.from_dict(  data_to_prompt(data) )
	#dataset = prm800kRewardDataset(data)
	dataset = dataset.train_test_split(test_size=0.01)
	train_dataset = dataset["train"]
	test_dataset = dataset["test"]
	
	model_id =  "meta-llama/Llama-3.2-1B-Instruct" #"meta-llama/Llama-3.2-1B"
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	train_dataset = train_dataset.map(reward_preprocess, batched=True)
	test_dataset = test_dataset.map(reward_preprocess, batched=True)	
	
	#device = torch.device("cuda:0")
	model =  LlamaForSequenceClassification.from_pretrained(model_id, num_labels=3) # LlamaModel
	#model.cuda()

	# Start training    
	training_args = TrainingArguments(
			output_dir='./model_temp',
			num_train_epochs=30,
			per_device_train_batch_size=1, #16,
			gradient_accumulation_steps=1,
			#gradient_checkpointing=True, - slows down the training
			learning_rate=1e-6,
			logging_steps=20,
			save_steps=500,
			save_total_limit=3,
			load_best_model_at_end=True,
			evaluation_strategy="steps",
			eval_steps=50, #500
			per_device_eval_batch_size=1,
			#metric_for_best_model='eval_loss',
			#remove_unused_columns=False,
			#logging_dir="./logs/",
			#report_to="tensorboard",
			#weight_decay=0.01,
	)

	trainer = Trainer(
			model=model,
			#data_collator=data_collator,
			args=training_args,
			train_dataset=train_dataset,
			eval_dataset=test_dataset    
	)
	
	trainer.train()


	exit()
	#=======================	
	#playground
	messages = [
		{"role":"system", "content":"generate next single step to solve this math problem"},
		{"role":"user", "content":"What is the greatest common factor of $20 !$ and $200,\\!000$? (Reminder: If $n$ is a positive integer, then $n!$ stands for the product $1\\cdot 2\\cdot 3\\cdot \\cdots \\cdot (n-1)\\cdot n$.)"},
		{"role":"assistant", "content":"I want to find the largest positive integer that divides both $20 !$ and $200,\\!000$ evenly."}
	]
	prompt = messages_to_prompt(messages)  #"Who is the president of US?"	
	input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
	with torch.no_grad():
		out = model.generate(input_ids=input_ids, max_length=800) #.detach().cpu() #stopping_criteria=stopping_criteria
		#logits = out[0]
		#print(logits.shape, logits, "\n")
		#print(out.last_hidden_state.shape, out.last_hidden_state)
	print(  tokenizer.batch_decode(out) )



