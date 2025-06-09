# venv PC4 - asr3.8, US1 - asr3.12
# landing1(ch-19k) was trained on Qwen2 instruct without any model modifications
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
        input_ids, labels, prompt_lens = [], [], []

        for sample in features:
            prompt = sample["prompts"]
            answer = sample["labels"]

            # Compose full text
            full = f"{prompt}{answer}<|im_end|>" #<|eot_id|>

            full_tokens = tokenizer(full, truncation=True, max_length=4000).input_ids
            prompt_tokens = tokenizer(prompt, truncation=True, max_length=3800).input_ids

            label_ids = [-100] * len(prompt_tokens) + full_tokens[len(prompt_tokens):]

            input_ids.append(torch.tensor(full_tokens if mode==1 else prompt_tokens))
            labels.append(torch.tensor(label_ids))
            prompt_lens.append(len(prompt_tokens))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
        #print(input_ids.shape, labels.shape, attention_mask.shape)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask, "prompt_lens":prompt_lens}



class OwnTrainer(Trainer):
	def predict(self, test_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
		preds, lables = None, None
		eval_dataloader  = self.get_eval_dataloader(test_dataset)
		for step, inputs in enumerate(eval_dataloader):
			input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
			with torch.no_grad():
				generated_ids = self.model.generate(input_ids=input_ids,  max_new_tokens=50, do_sample=True, num_beams=10)
				generated_ids = [output_ids[plen-50:] for plen, output_ids in zip(inputs["prompt_lens"], generated_ids)] #remove input from output
				compute_metrics({"predictions":generated_ids, "labels":inputs["labels"]})
		return {"accuracy": 1.0}


def compute_metrics(p):
	if mode==1:
		labels, generated_ids = torch.tensor(p.label_ids), np.argmax(p.predictions, axis=-1)
	else:
		generated_ids, labels = p["predictions"], p["labels"] #predict, eval
	labels[labels == -100] = tokenizer.pad_token_id	
	pred = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)	
	labels = tokenizer.batch_decode(labels, skip_special_tokens=True)	
	for p,l in zip(pred, labels):
		print(p[:], " -- LABEL:", l, "\n#==================\n")
	wer = wer_metric.compute(predictions=pred, references=labels)
	return {"eval_accuracy": wer}



if __name__=="__main__":
	mode = 2 #1-train, 2-test
	model_id = "Qwen/Qwen2-0.5B-Instruct"
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	tokenizer.pad_token = tokenizer.eos_token #'!' #'<|finetune_right_pad_id|>' 
	tokenizer.pad_token_id = tokenizer.eos_token_id
	tokenizer.truncation_side = 'left'
	print("tokenizer:", tokenizer.pad_token_id, tokenizer.truncation_side)
	wer_metric = evaluate.load("wer")

	d, gp = prepare_data(mode)
	dataset = Dataset.from_dict(d)
	dataset = dataset.map(preprocess, batched=True)
	dataset = dataset.train_test_split(test_size=0.001 if mode==1 else 12, seed=42)
	train_dataset = dataset["train"]
	test_dataset = dataset["test"]
	print("Dataset train, test sizes:",  len(train_dataset), len(test_dataset))
	
	model = AutoModelForCausalLM.from_pretrained(model_id if mode==1 else "./model_temp/checkpoint-19000")
	model.config.pad_token_id = tokenizer.pad_token_id
	# looping L=16/k
	#model.config.num_hidden_layers=4
	#model.model.layers = model.model.layers[:4]
	#endOf looping	
	#print(model, model.config)

	# Start training    
	data_collator = myDataCollator()
	training_args = TrainingArguments(
		output_dir='./model_temp',
		num_train_epochs=100,
		per_device_train_batch_size=1,
		gradient_accumulation_steps=8,
		#gradient_checkpointing=True, - slows down the training
		learning_rate=1e-6,
		logging_steps=20,
		save_steps=500,
		save_total_limit=2,
		load_best_model_at_end=True,
		eval_strategy="steps", #evaluation_strategy
		eval_steps=500,
		per_device_eval_batch_size=1,
		metric_for_best_model="eval_loss",
		greater_is_better=False,
		remove_unused_columns=False,
		#logging_dir="./logs/",
		#report_to="tensorboard",
		#weight_decay=0.01,
	)

	trainer = OwnTrainer(
		model=model,
		data_collator=data_collator,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=test_dataset,
		#compute_metrics=compute_metrics
	)
	
	if mode==1: trainer.train("./model_temp/checkpoint-17070")
	else: 
		#print( trainer.evaluate() )		
		predictions = trainer.predict(test_dataset)
