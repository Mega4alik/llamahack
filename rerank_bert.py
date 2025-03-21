# RAG Rerank with BERT -- use chunk embeddings(ex, SONAR) instead of tokens
import numpy as np
import json
import os
import random
#import multiprocessing
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
#from torch.utils.data import Dataset, DataLoader
from datasets import Dataset
from typing import Any, Dict, List, Optional, Union
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModel
from utils import file_get_contents, file_put_contents, pickle_load, pickle_save

G_CHUNK_SIZE = 200
COHERE_EVAL = True #whether we are evaluationg cohere rerank now

def split_article_with_fact(article_text, fact_text, chunk_size=200):	
	def _chunk_words(words, chunk_size):	    
		result = []
		start = 0
		while start < len(words):
			result.append(" ".join(words[start:start + chunk_size]))
			start += chunk_size
		return result    

	# Helper function to locate fact in article (naive approach)
	def find_fact_indices(words, fact):
		for start_index in range(len(words) - len(fact) + 1):
			if words[start_index:start_index + len(fact)] == fact:
				return start_index, start_index + len(fact)
		return None, None

	# Split both article and fact into lists of words
	article_words = article_text.split()
	if fact_text is None: return _chunk_words(article_words, chunk_size)
	fact_words = fact_text.split()
	fact_len = len(fact_words)

	fact_start, fact_end = find_fact_indices(article_words, fact_words)

	# If the fact isn’t found, just chunk normally
	if fact_start is None:
		return _chunk_words(article_words, chunk_size)

	# Otherwise, chunk in three parts:
	# 1. Everything before the fact
	# 2. A single chunk that contains the entire fact
	# 3. Everything after the fact

	chunks = []

	# Chunk everything before the fact
	start = 0
	while start + chunk_size <= fact_start:
		chunks.append(" ".join(article_words[start:start + chunk_size]))
		start += chunk_size

	# Place the fact-containing chunk (could be bigger than chunk_size if needed)
	# The chunk starts where we left off above and ends at least at fact_end
	# (You can extend chunk_end further if you prefer balancing chunk sizes)
	chunk_end = max(fact_end, start + chunk_size)
	chunks.append(" ".join(article_words[start:chunk_end]))

	# Chunk everything after the fact
	start = chunk_end
	while start < len(article_words):
		chunks.append(" ".join(article_words[start:start + chunk_size]))
		start += chunk_size

	return chunks

def multihop_qa_prepare_data():
	dataset = []
	qas = json.loads(file_get_contents("./data/MultiHopRAG.json"))
	articles = json.loads(file_get_contents("./data/corpus.json"))	
	#print(len(qas), len(articles), qas[0], articles[0])
	for step, qa in enumerate(qas[:]):
		if len(qa["evidence_list"])==0: continue
		question = qa["query"]
		chunks_list, labels_list, urls, chunks_n = [], [], [], 0
		#add evidence chunks(fact, url)
		for ev in qa["evidence_list"]:
			urls.append(ev["url"])
			fact = ev["fact"]
			article_text = next((x for x in articles if x["url"] == ev["url"]), None)['body']
			chunks, labels = split_article_with_fact(article_text, fact, chunk_size=G_CHUNK_SIZE), []			
			chunks_list.append(chunks)
			chunks_n+=len(chunks)
			# double check
			found = False
			for chunk in chunks:
				if fact in chunk:
					found = True
					labels.append(1)
				else:
					labels.append(0)
			labels_list.append(labels)
			assert found == True
		#./endOf add evidence chunks(fact, url)
		#add more chunks
		articles2 = [x for x in articles if x["url"] not in urls]
		random.shuffle(articles2)
		for article in articles2:
			chunks = split_article_with_fact(article["body"], None, chunk_size=G_CHUNK_SIZE)
			if chunks_n + len(chunks) > 511: break
			chunks_list.append(chunks)
			labels_list.append([0] * len(chunks))
			chunks_n+=len(chunks)			
		#./endOf add more chunks

		dataset.append( (question, chunks_list, labels_list) )
	return dataset

def hashf(st):
	if len(st) < 120: return st
	else: return st[:50] + "..." + st[-50:]

@torch.inference_mode()
def make_embeddings(dataset):
	global emb_cache
	path = "./temp/rerank_cache.pkl"
	if os.path.exists(path):
		emb_cache = pickle_load(path)
		print("emb_cache len:", len(emb_cache))
		return
	
	embedding_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
	embedding_model.eval()
	embedding_model.cuda()
	for step, (question, chunks_list, labels_list) in enumerate(dataset):
		print(f"\rEmb: {step}/2555", end="", flush=True)
		h = hashf(question)
		if h not in emb_cache: emb_cache[h] = embedding_model.encode([question], task="retrieval.query")[0]
		for chunks in chunks_list:
			for chunk in chunks:
				h = hashf(chunk)
				if h not in emb_cache: emb_cache[h] = embedding_model.encode([chunk], task="retrieval.passage")[0]
	
	#torch.save(emb_cache, path, _use_new_zipfile_serialization=False)
	pickle_save(path, emb_cache)


def dataset_to_dict(dataset):	
	d = {}
	for (question, chunks_list, labels_list) in dataset:
		for o in [ ("question", question), ("chunks_list", chunks_list), ("labels_list", labels_list), ]: #("urls_list", urls_list), ("question_emb", question_emb), ("chunks_emb", chunks_emb) 
			k, v = o[0], o[1]
			if k not in d: d[k] = []
			d[k].append(v)
	return d


class DataCollator:
	def shuffle_lists(self, a, b):
		combined = list(zip(a, b))  # Pair corresponding elements
		random.shuffle(combined)  # Shuffle the pairs
		a_shuffled, b_shuffled = zip(*combined)  # Unzip after shuffling
		return list(a_shuffled), list(b_shuffled)
	

	def __call__(self, features) -> Dict[str, torch.Tensor]:
		#features: question_emb, chunks_emb, labels, question, chunks_list, urls_list
		"""
		question_emb = [feature["question_emb"] for feature in features] #b,[emb]
		urls_list = [feature["urls_list"] for feature in features] #b,[url]		
		labels_list = [feature["labels_list"] for feature in features] #b,[ [label1, label2], [..] ]
		"""
		batch = {"input_values": [], "labels":[], "chunks_list":[], "labels_list":[], "question":[]}
		for x in features: #for each batch
			question, labels_list, chunks_list = x["question"], x["labels_list"], x["chunks_list"]
			question_emb, chunks_emb = emb_cache[hashf(question)], []
			for chunks in chunks_list:
				chunks_emb.append( [emb_cache[hashf(chunk)] for chunk in chunks] )
			if not COHERE_EVAL: chunks_emb, labels_list = self.shuffle_lists(chunks_emb, labels_list)
			input_values, labels = [question_emb], [0] #input value: list of embeddings(list), labels: list of 0/1
			for i, embs in enumerate(chunks_emb):
				for emb in embs: input_values.append(emb)
				labels += labels_list[i] #seq

			input_values, labels = torch.tensor(input_values), torch.tensor(labels) #, dtype=torch.int32
			batch["input_values"].append(input_values)
			batch["labels"].append(labels)
			if COHERE_EVAL:
				batch["chunks_list"].append(chunks_list) #temp for cohere
				batch["labels_list"].append(labels_list) #temp for cohere
				batch["question"].append(question) #temp for cohere

		batch["input_values"] = pad_sequence(batch["input_values"], batch_first=True, padding_value=0) #B,S,C
		batch["labels"] = pad_sequence(batch["labels"], batch_first=True, padding_value=0) #B,S -100
		#print("batch shapes:", batch["input_values"].shape, batch["labels"].shape)
		return batch


class MyModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.bert_hidden_dim = 768 #bert emb dim
		self.embedding_dim = 1024 #jina emb dim
		self.llm_model = llm_model
		self.fc1 = nn.Linear(self.embedding_dim, self.bert_hidden_dim)
		self.fc2 = nn.Linear(self.bert_hidden_dim, 1)

	def trans(self, a):
		x = self.fc1(a)
		return x

	def forward(self,
		input_values: Optional[torch.Tensor],
		attention_mask: Optional[torch.Tensor] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		labels: Optional[torch.Tensor] = None,
		#target_lengths: Optional = None
	):
		out = self.llm_model(inputs_embeds=self.trans(input_values), output_hidden_states=True)
		pred = self.fc2(out.hidden_states[-1]).squeeze(-1) #B, S
		pred = torch.sigmoid(pred) #0..1

		if labels is None: #inference
			return pred
		else:
			#print("forward pred, labels:", pred.shape, labels.shape)
			loss = bce_loss(pred, labels.float())
			return {"loss":loss}


	def generate(self, x): #x-processed speech array
		pred = self.forward(input_values=x)
		return pred

	
	def _load_from_checkpoint(self, load_directory):
		load_path = os.path.join(load_directory, 'state_dict.pt')
		checkpoint = torch.load(load_path)
		self.fc1.load_state_dict(checkpoint['fc1_state_dict'])
		self.fc2.load_state_dict(checkpoint['fc2_state_dict'])
		self.llm_model.load_state_dict(checkpoint['llm_state_dict'])


class OwnTrainer(Trainer):
	def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
		preds, lables = None, None
		eval_dataloader = self.get_eval_dataloader(eval_dataset)
		for step, inputs in enumerate(eval_dataloader):
			if COHERE_EVAL: return cohere_rerank(inputs["question"], inputs["chunks_list"], inputs["labels_list"])
			with torch.no_grad():
				pred = self.model.generate(inputs['input_values']) #B,S				
			return compute_metrics({"predictions":pred, "labels":inputs['labels']})	
			

	def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False): #called from Trainer._save_checkpoint	
		save_directory, model = output_dir, self.model
		os.makedirs(save_directory, exist_ok=True)		
		save_path = os.path.join(save_directory, 'state_dict.pt')
		torch.save({
			'fc1_state_dict': model.fc1.state_dict(),
			'fc2_state_dict': model.fc2.state_dict(),
			'llm_state_dict': model.llm_model.state_dict()
		}, save_path)

	def _load_optimizer_and_scheduler(self, checkpoint):
		print("OPTIMIZER loading on train()!\n\n")
		#super()._load_optimizer_and_scheduler(checkpoint)
	
	def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
		self.model._load_from_checkpoint(resume_from_checkpoint)		
		return self.model


def compute_metrics(x):
	batch_preds, batch_labels = x["predictions"], x["labels"]
	correct, n = 0, 0
	for i in range(len(batch_preds)):
		probs, labels = batch_preds[i], batch_labels[i]
		top_indices = torch.topk(probs, 10).indices.detach().tolist()
		labels_ones = torch.nonzero(labels == 1).squeeze().detach().tolist()
		for lidx in labels_ones:
			if lidx in top_indices: correct+=1
			n+=1
		print(labels_ones, top_indices, correct, n)
	return {"eval_accuracy": (correct / n) }


#==== cohere ========
if COHERE_EVAL:
	from config import COHERE_API_KEY
	import cohere
	co = cohere.ClientV2(COHERE_API_KEY)
	def cohere_rerank(b_question, b_chunks_list, b_labels_list):		
		correct, n = 0, 0
		for i, chunks_list in enumerate(b_chunks_list):
			question, labels_list, documents, labels  = b_question[i], b_labels_list[i], [], []
			for chunks in chunks_list:
				for chunk in chunks: documents.append(chunk)
			for a in labels_list: labels+=a
			print("docs, labels lengths:", len(documents), len(labels))
			labels_ones = torch.nonzero(torch.tensor(labels) == 1).squeeze().detach().tolist()
			documents_reranked = co.rerank(model="rerank-v3.5", query=question, documents=documents, top_n=10)
			top_indices = [r.index for r in documents_reranked.results]
			for lidx in labels_ones:
				if lidx in top_indices: correct+=1
				n+=1
			print(labels_ones, top_indices, correct, n)
		return {"eval_accuracy": (correct / n) }
#==== endOf cohere ========	


###################### __main__ ###########################
gpu, device = True, torch.device("cuda")
llm_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
llm_model = AutoModel.from_pretrained("google-bert/bert-base-uncased")
mymodel = None

if 1==2: #Evaluate
	tester = MyTester()
	#tester.T248_evaluate()
else: #Train
	#prepare data
	emb_cache = {}
	dataset = multihop_qa_prepare_data()
	make_embeddings(dataset)
	d = dataset_to_dict(dataset)
	del dataset
	mydataset = Dataset.from_dict(d)
	del d
	mydataset = mydataset.train_test_split(test_size=0.01, seed=42)
	train_dataset, val_dataset = mydataset["train"], mydataset["test"]
	#endOf prepare data
	
	mymodel = MyModel()
	bce_loss = nn.BCELoss()
	data_collator = DataCollator()
	training_args = TrainingArguments(
	  output_dir="./model_temp/",
	  #group_by_length=True, length_column_name="len",
	  per_device_train_batch_size=16,
	  gradient_accumulation_steps=1, #update each 2 * batch_size
	  #fp16=True,
	  evaluation_strategy="steps",
	  num_train_epochs=100,
	  logging_steps=50,
	  save_steps=500,
	  eval_steps=500,
	  per_device_eval_batch_size=23,
	  learning_rate=1e-5,
	  dataloader_num_workers=4,
	  weight_decay=0.005,
	  warmup_steps=1000,
	  save_total_limit=3,
	  ignore_data_skip=True,
	  remove_unused_columns=False,
	  #label_names=["labels"], #attempt to solve eval problem
	  metric_for_best_model="eval_accuracy",
	  #load_best_model_at_end=True,
	)
	print("\n\nstarting training", len(train_dataset), len(val_dataset))
	trainer = OwnTrainer(
		model=mymodel,
		data_collator=data_collator,
		args=training_args,
		compute_metrics=compute_metrics,
		train_dataset=train_dataset,
		eval_dataset=val_dataset,
		#tokenizer=processor.feature_extractor,
	)
	#trainer.train("./model_temp/checkpoint-1000")
	trainer._load_from_checkpoint("./model_temp/checkpoint-11500")
	trainer.evaluate()

