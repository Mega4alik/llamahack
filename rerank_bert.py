# RAG Rerank with BERT -- use chunk embeddings(ex, SONAR) instead of tokens
import numpy as np
import json
import os
import random
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
#from torch.utils.data import Dataset, DataLoader
from datasets import Dataset
from typing import Any, Dict, List, Optional, Union
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModel
from utils import file_get_contents, file_put_contents


class DataCollator:
	def shuffle_lists(self, a, b):
		combined = list(zip(a, b))  # Pair corresponding elements
		random.shuffle(combined)  # Shuffle the pairs
		a_shuffled, b_shuffled = zip(*combined)  # Unzip after shuffling
		return list(a_shuffled), list(b_shuffled)

	def __call__(self, features) -> Dict[str, torch.Tensor]:
		#features: question_emb, chunks_emb, labels, question, chunks_list
		question_emb = [feature["question_emb"] for feature in features] #[emb]
		chunks_emb = [feature["chunks_emb"] for feature in features] #[ [emb1, emb2, ..], [..] ]
		labels_list = [feature["labels"] for feature in features] #[ [label1, label2], [..] ]
		chunks_emb, labels_list = self.shuffle_lists(chunks_emb, labels_list)
		input_values, labels = [question_emb[0]], [0]
		for i, embs in enumerate(chunks_emb):
			input_values += embs
			labels += labels_list[i]

		input_values = pad_sequence([torch.tensor(seq) for seq in input_values], batch_first=True, padding_value=0)
		labels = pad_sequence([torch.tensor(seq) for seq in labels], batch_first=True, padding_value=0)
		batch = {"input_values": input_values, "labels":labels}
		return batch



class MyModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.embedding_dim = 768
		self.llm_model = llm_model
		self.llm_tokenizer = llm_tokenizer
		#self.fc1 = nn.Linear(768,  self.embedding_dim)
		self.fc2 = nn.Linear(self.embedding_dim, 1)


	def trans(self, a):
		#x = self.llm_model(a, output_hidden_states=True).hidden_states[-1] #.last_hidden_state | hidden_states[-1]  (B, S/20ms, 768)		
		#z = self.fc(x)		
		return a


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
		#print(out.hidden_states[-1].shape, labels)  #B, S/20ms, 768
		pred = self.fc2(out.hidden_states[-1]) #B, S/20ms, 1

		if labels is None: #inference
			return pred
		else:
			loss = bce_loss(pred, labels)
			return {"loss":loss}


	def generate(self, x): #x-processed speech array
		pred = self.forward(input_values=x)
		return pred

	
	def _load_from_checkpoint(self, load_directory):				
		load_path = os.path.join(load_directory, 'state_dict.pt')
		checkpoint = torch.load(load_path)
		self.fc.load_state_dict(checkpoint['fc_state_dict'])		
		self.llm_model.load_state_dict(checkpoint['llm_state_dict'])


class OwnTrainer(Trainer):
	def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
		eval_dataloader = self.get_eval_dataloader(eval_dataset)
		for step, inputs in enumerate(eval_dataloader):
			# Move inputs to the appropriate device
			#inputs = {k: v.to(self.args.device) for k, v in inputs.items()}
			# Disable gradient calculation
			with torch.no_grad():
				pred = self.model.generate(inputs['input_values'])
			return compute_metrics({"predictions":pred, "labels": inputs["labels"]})
			
	def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False): #called from Trainer._save_checkpoint	
		save_directory, model = output_dir, self.model
		os.makedirs(save_directory, exist_ok=True)		
		save_path = os.path.join(save_directory, 'state_dict.pt')
		torch.save({
			'fc2_state_dict': model.fc2.state_dict(),			
			'llm_state_dict': model.llm_model.state_dict(),
		}, save_path)

	def _load_optimizer_and_scheduler(self, checkpoint):
		print("OPTIMIZER loading on train()!\n\n")
		#super()._load_optimizer_and_scheduler(checkpoint)
	
	def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
		self.model._load_from_checkpoint(resume_from_checkpoint)		
		return self.model


def compute_metrics(x):
	batch_preds, batch_labels = x["predictions"], x["labels"]
	for i in range(len(batch_preds)):
		probs, labels = batch_preds[i], batch_labels[i]
		top_indices = torch.topk(probs, 10).indices
		labels_ones = torch.nonzero(labels == 1).squeeze()
		print(labels_ones, top_indices)
	return {"eval_wer": 1}


#====================================================================

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
	global chunks_emb_cache
	model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
	model.cuda()
	dataset = []
	qas = json.loads(file_get_contents("./data/MultiHopRAG.json"))
	articles = json.loads(file_get_contents("./data/corpus.json"))
	#print(len(qas), len(articles), qas[0], articles[0])
	for step, qa in enumerate(qas[:]):
		print(f"\rstep/2600: {step}", end="", flush=True)
		if len(qa["evidence_list"])==0: continue
		question = qa["query"]
		chunks_list, chunks_emb, labels_list, urls, chunks_n = [], [], [], [], 0
		#add evidence chunks(fact, url)
		for ev in qa["evidence_list"]:
			urls.append(ev["url"])
			fact = ev["fact"]
			article_text = next((x for x in articles if x["url"] == ev["url"]), None)['body']
			chunks, labels = split_article_with_fact(article_text, fact, chunk_size=200), []
			chunks_list.append(chunks)			
			chunks_n+=len(chunks)
			#print(len(article_text) // len(chunks)) #average number of characters in single chunk
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

		#add more chunks		
		articles2 = [x for x in articles if x["url"] not in urls]
		random.shuffle(articles2)
		for article in articles2:
			chunks, url = split_article_with_fact(article["body"], None, chunk_size=200), article["url"]
			if chunks_n + len(chunks) > 512: break
			chunks_list.append(chunks)
			labels_list.append([0] * len(chunks))
			chunks_n+=len(chunks)
			if url not in chunks_emb_cache: chunks_emb_cache[url] = model.encode(chunks, task="retrieval.passage")			
		#./endOf add more chunks

		question_emb = model.encode([question], task="retrieval.query")
		dataset.append( (question, chunks_list, labels_list, question_emb) )

	return dataset


def dataset_to_dict(dataset):
	d = {}
	for (question, chunks_list, labels_list, question_emb) in dataset:
		for o in [ ("question", question), ("chunks_list", chunks_list), ("labels_list", labels_list), ("question_emb", question_emb) ]:
			k, v = o[0], o[1]
			if k not in d: d[k] = []
			d[k].append(v)
	return d


class MyDataset(Dataset):
	def __init__(self, dataset):
		self.samples = []
		self.test_samples = []
		for (question, chunks_list, labels_list) in dataset:
			chunks_list, labels_list = self.shuffle_lists(chunks_list, labels_list)
			chunks, labels = ["Q: "+question], [0]
			for i in range(len(chunks_list)):        		
				for j, chunk in enumerate(chunks_list[i]):
					chunks.append(chunk)
					labels.append(labels_list[i][j])
			self.samples.append( (chunks, labels) )

	def __len__(self):
		return len(self.samples)
	
	def __getitem__(self, idx):
		chunks, labels = self.samples[idx]
		x = {'chunks': chunks, 'labels': labels} 
		return x

	def shuffle_lists(self, a, b):
		combined = list(zip(a, b))  # Pair corresponding elements
		random.shuffle(combined)  # Shuffle the pairs
		a_shuffled, b_shuffled = zip(*combined)  # Unzip after shuffling
		return list(a_shuffled), list(b_shuffled)



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
	path = "./temp/rerank_dataset.pt"
	if not os.path.exists(path):
		chunks_emb_cache = {}
		dataset = multihop_qa_prepare_data()
		d = dataset_to_dict(dataset)
		mydataset = Dataset.from_dict(d)
		#mydataset = MyDataset(dataset)
		mydataset = mydataset.train_test_split(test_size=0.01)
		torch.save(mydataset, path)
		torch.save(chunks_emb_cache, "./temp/rerank_cache.pt")
	else:
		mydataset = torch.load(path)
		chunks_emb_cache = torch.load("./temp/rerank_cache.pt")
	#mydataset = mydataset.filter(lambda x: x["len"] <= 30 * 16000, num_proc=2) #less than 30 secs
	train_dataset = mydataset["train"]
	val_dataset = mydataset["test"]
	#endOf prepare data
	
	mymodel = MyModel()
	#mymodel.freeze_encoder_layers() #freezing wav2vec for initial training
	bce_loss = nn.BCELoss()
	data_collator = DataCollator()
	training_args = TrainingArguments(
	  output_dir="./model_temp/",
	  #group_by_length=True, length_column_name="len",
	  per_device_train_batch_size=8,
	  gradient_accumulation_steps=2, #update each 2 * batch_size
	  fp16=True,
	  evaluation_strategy="steps",
	  num_train_epochs=50,
	  logging_steps=100,
	  save_steps=5000,
	  eval_steps=50,
	  learning_rate=1e-5,
	  dataloader_num_workers=4,
	  weight_decay=0.005,
	  warmup_steps=1000,
	  save_total_limit=3,
	  ignore_data_skip=True,
	  remove_unused_columns=False,
	  #label_names=["labels"], #attempt to solve eval problem
	  #metric_for_best_model="eval_wer",
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
	trainer.train()

