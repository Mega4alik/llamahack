import json
import os
import re
import numpy as np
import soundfile as sf
import random
import time
import datetime
import webrtcvad
from pydub import AudioSegment
from pydub.utils import make_chunks
from datasets import load_dataset
import evaluate
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModel, AdamW, get_linear_schedule_with_warmup
from transformers import Trainer, TrainingArguments
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
import torch
import torch as T
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import transformers as TRF

import preprocess as pp


# example: T5 small
basetrf = TRF.T5ForConditionalGeneration.from_pretrained("google/mt5-small") #t5-small
basetrfconfig =  basetrf.config   # TRF.T5Config()
#print(basetrf.config)
basetokenizer = TRF.T5Tokenizer.from_pretrained("google/mt5-small")


class CFG:
	trf = basetrf
	trf_hiddim = basetrfconfig.d_model # T5's embedding dimension (e.g., 768 for 't5-base')
	maximum_length_seq = 150
	input_shape = (trf_hiddim, 45) #768-base,512-small


class Modified_TRF(nn.Module):
	"""
	Input shape: (X,32)
	"""
	def __init__(self, CFG=CFG):
		super().__init__()
		self.transformer = CFG.trf
		self.maxlen = CFG.maximum_length_seq

		#v1
		#self.fc = nn.Linear(CFG.input_shape[-1], CFG.trf_hiddim)
		
		#v2
		#self.cnn = nn.Conv1d(CFG.input_shape[-1], CFG.input_shape[0], 4, stride=2)
		self.cnn = nn.Sequential( #v2.1
			nn.Conv1d(in_channels=CFG.input_shape[-1], out_channels=CFG.input_shape[0], kernel_size=4, padding=2),
			nn.ReLU(),
			nn.BatchNorm1d(CFG.input_shape[0])
		)

		
		#v3		
		num_filters = 128
		kernel_sizes = [3, 5, 7]
		self.convs = nn.ModuleList([
			 nn.Sequential(
				 nn.Conv1d(in_channels=CFG.input_shape[-1], out_channels=num_filters, kernel_size=k, padding=k//2),
				 nn.ReLU(),
				 nn.BatchNorm1d(num_filters)
			 )
			 for k in kernel_sizes
		])
		self.fc = nn.Linear(num_filters * len(kernel_sizes), CFG.trf_hiddim)
		

	def trans2(self, x):
		#print(x.size(), z.size(), (2,75), "x[1] =",  x.size()[1])
		x = x.transpose(2,1)
		z = self.cnn(x)
		z = z.transpose(2,1)
		return z


	def trans3(self, x):
		 # x shape: (batch_size, seq_length, vocab_size)
		 x = x.permute(0, 2, 1)  # Shape: (batch_size, vocab_size, seq_length)

		 conv_outputs = []
		 for conv in self.convs:
			 conv_out = conv(x)  # Shape: (batch_size, num_filters, seq_length)
			 conv_outputs.append(conv_out)

		 # Concatenate along the channel dimension
		 x = T.cat(conv_outputs, dim=1)  # Shape: (batch_size, num_filters * len(kernel_sizes), seq_length)

		 # Permute back to (batch_size, seq_length, features)
		 x = x.permute(0, 2, 1)  # Shape: (batch_size, seq_length, num_filters * len(kernel_sizes))

		 embeddings = self.fc(x)  # Shape: (batch_size, seq_length, embedding_dim)
		 return embeddings		


	def generate(self, x):
		#return self.transformer.generate(inputs_embeds=self.trans1(x), max_new_tokens=self.maxlen) #stable for validation swbd
		return self.transformer.generate(inputs_embeds=self.trans3(x), max_new_tokens=self.maxlen, do_sample=False) #for inference-		

	def forward(self, x, y):									
		return self.transformer(inputs_embeds=self.trans3(x), labels=y) #self.fc(x) 



def format_time(elapsed):	
	elapsed_rounded = int(round((elapsed)))
	return str(datetime.timedelta(seconds=elapsed_rounded)) 	# Format as hh:mm:ss



def flat_accuracy2(preds, labels, tokenizer):      
	wers = 0.0
	for i in range(len(preds)):
		l = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(labels[i], skip_special_tokens=True)).lower()
		try:
			p = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(preds[i], skip_special_tokens=True)).lower()
		except Exception as e:
			print(preds[i])
			p = ""				
		wer = min( wer_metric.compute(predictions=[p], references=[l]), 1.0)
		wers+=wer
		print(p, "---- label:", l, "WER:", wer)
	return wers




def get_batch(X2, Y2, batch_size, j, indexes):
	if indexes is None:
		xs = X2[j*batch_size : (j+1)*batch_size]
		ys = Y2[j*batch_size : (j+1)*batch_size]
		anms = audio_names[j*batch_size : (j+1)*batch_size]
	else:
		arr = indexes[j*batch_size : (j+1)*batch_size]
		xs = [X2[idx] for idx in arr]
		ys = [Y2[idx] for idx in arr]
		anms = [audio_names[idx] for idx in arr]

	xs = pad_sequence(xs, batch_first=True)
	ys = pad_sequence(ys, batch_first=True)
	batch = (xs, ys, anms)
	return batch



def save_logits(audio_path, logits_path):	
	if os.path.exists(logits_path): return
	speech_array, sr = sf.read(audio_path)	
	input_values = processor2([speech_array], sampling_rate=sr, return_tensors="pt", padding=True, return_attention_mask=False).input_values.to(device)
	with torch.no_grad():
		logits = model2(input_values).logits.to("cpu")
	torch.save(logits, logits_path)
	predicted_ids = torch.argmax(logits, dim=-1)
	print("Prediction:", processor2.batch_decode(predicted_ids))
	#endOf temp
	

def prepare_data(maxlen):
	global X
	global Y
	global audio_names	
	pre = '/home/mega4alik/Desktop/data/cv_kk/'
	lines = []
	for setname in ["train", "dev", "other", "test"]: lines += pp.file_get_contents(pre+setname+".tsv").split("\n")[1:]
	texts = []

	for line in lines[:maxlen]:
		if not line: continue
		q = line.split("\t")		
		audio_name, text = q[1], q[3].lower()
		text = re.sub(r'[^a-zA-Zа-яА-ЯәғқңөұүіӘҒҚҢӨҰҮІ ]', '', text)
		logit_name = audio_name.replace(".mp3","")		
		logits_path = "./temp/logits/cv_kk/"+logit_name+".pt"
		#print("\t",audio_name, text)
		#save_logits(pre+"16k/"+audio_name, logits_path)		
		#continue
		logits = torch.load(logits_path)
		x = torch.squeeze(logits)
		if x.size(dim=0)>600: continue  #remove longer than 30 seconds
		X.append(x)
		texts.append(text.strip())
		audio_names.append(audio_name)
		
	Y = []
	for text in texts:
		 y = basetokenizer(text, add_special_tokens=True, return_tensors='pt', truncation=False, padding=False)['input_ids'][0]
		 #y = y.numpy().tolist() # [:-1] + [32100, 1] #append special element?
		 Y.append(y)
	
	print(Y[0], len(Y[0]))
	print(len(X), len(Y), " -----  dataset size")	
	



def train():
	testlen, maxlen = 150, 200000
	prepare_data(maxlen) #X,Y
	testX = X[:testlen]
	testY = Y[:testlen]
	trainX = X[testlen:]
	trainY = Y[testlen:]

	# ========================================
	#               Training
	# ========================================
	print('-'*20 + ' Starting Trainig ... ' + '-'*20)
	DO_VALIDATE = True
	epochs = 300
	batch_size = 4 #32
	best_val_accuracy = 999
	stepsN = int(len(trainX) / batch_size) #bs  --  steps per epoch
	total_steps = stepsN * epochs  #len(train_dataloader) * epochs

	model = Modified_TRF(CFG)
	optimizer = AdamW(model.parameters(), lr = 5e-5, eps = 1e-8 )	

	#load from checkpoint
	#checkpoint = torch.load("./model_temp/checkpoints/16_36.533_model.pt", map_location='cpu')
	#model.load_state_dict(checkpoint['model_state_dict'])
	#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])	
	#endOf load from checkpoint	

	scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)
	if gpu: model.cuda()	

	# Set the seed value all over the place to make this reproducible.
	seed_val = 42
	random.seed(seed_val)
	np.random.seed(seed_val)
	torch.manual_seed(seed_val)
	torch.cuda.manual_seed_all(seed_val)
	loss_values = []
	verbose_step = 1
	if gpu: verbose_step = 40

	# For each epoch...	
	for epoch_i in range(1, epochs+1):	  
		print('======== Epoch {:} / {:} ========'.format(epoch_i, epochs), flush=True)        
		t0 = time.time()
		total_loss = 0    
		model.train()					
		shuffledIndexes = list(range(len(trainX)))
		random.shuffle(shuffledIndexes)

		# For each batch of training data...
		for step in range(stepsN):   #step, batch in enumerate(train_dataloader):
			if step % verbose_step == 0 and not step == 0: print('  Batch {:>5,}  of  {:>5,}.'.format(step, stepsN), flush=True)
			batch = get_batch(trainX, trainY, batch_size, step, shuffledIndexes)
			if gpu: #transforming in forward, only then putting into GPU
				b_inputs = batch[0].to(device)
				b_labels = batch[1].to(device)            
			else:
				b_inputs = batch[0]
				b_labels = batch[1]            
			model.zero_grad()			
			outputs = model(b_inputs, b_labels)  #model(input_ids=b_input_ids, labels=b_labels)		  
			loss = outputs[0]
			total_loss += loss.item()			  
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
			optimizer.step()
			scheduler.step()

		
		avg_train_loss = total_loss  / stepsN  #/ len(train_dataloader)		
		loss_values.append(avg_train_loss) 	  
		print("  Average training loss: {0:.7f}".format(avg_train_loss))
		print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

		# ========================================
		#               Validation
		# ========================================
		if DO_VALIDATE==True:
			print("Running Validation...")
			t0 = time.time()        			
			model.eval() # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
			eval_loss, eval_accuracy = 0, 0
			nb_eval_steps, nb_eval_examples = 0, 0			

			for j in range(int(len(testX)/batch_size)):
				batch = get_batch(testX, testY, batch_size, j, None)
				if gpu: b_inputs = batch[0].to(device) #putting into GPU in forward
				else: b_inputs = batch[0]
				b_labels = batch[1]
				with torch.no_grad():
					logits =  model.generate(b_inputs)
					#logits = outputs #[0]							
				if gpu: logits = logits.detach().cpu().numpy()              
				else: logits = logits.numpy()
				label_ids = b_labels.numpy()
				tmp_eval_accuracy = flat_accuracy2(logits, label_ids, basetokenizer)
				eval_accuracy += tmp_eval_accuracy            
				nb_eval_steps += len(label_ids) #1
			
			val_acc = round(eval_accuracy/len(testX)*100, 3)
			print("  Accuracy: {0:.7f}".format(val_acc), flush=True) #nb_eval_steps
			print("  Validation took: {:}".format(format_time(time.time() - t0)))        
			
			if best_val_accuracy > val_acc: #WER >, others <
				save_dir = "./model_temp/checkpoints/"+str(epoch_i)+"_"+str(val_acc)+"_model.pt" 
				torch.save({'epoch': epoch_i, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': avg_train_loss}, save_dir)			
				best_val_accuracy = val_acc
		#./ ============= end of VALIDATION ===============



		
def test():
	model = Modified_TRF(CFG)
	path = "./model_temp/checkpoints1_038WER/223_37.731_model.pt"
	checkpoint = torch.load(path, map_location='cpu')
	model.load_state_dict(checkpoint['model_state_dict'])	
	#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	if gpu: model.cuda()
	model.eval()
	print("model initialized")

	prepare_data(2000000) #X,Y 
	testlen = 150
	testX = X[:testlen]
	testY = Y[:testlen]
	trans_map = {}

	batch_size, eval_accuracy = 4, 0

	for j in range(int(len(testX)/batch_size)):
		batch = get_batch(testX, testY, batch_size, j, None)
		if gpu: b_inputs = batch[0].to(device)
		else: b_inputs = batch[0]
		b_labels = batch[1]
		b_anms = batch[2]
		
		with torch.no_grad():
			logits =  model.generate(b_inputs)			
		if gpu: logits = logits.detach().cpu().numpy()
		else: logits = logits.numpy()

		#label_ids = b_labels.numpy()
		#tmp_eval_accuracy = flat_accuracy2(logits, label_ids, basetokenizer)	
		#eval_accuracy += tmp_eval_accuracy
		
		for i in range(batch_size):
			audio_name = b_anms[i]
			text = basetokenizer.convert_tokens_to_string(basetokenizer.convert_ids_to_tokens(logits[i], skip_special_tokens=True))
			print(audio_name, text)
			if audio_name not in trans_map:  trans_map[audio_name] = ""
			trans_map[audio_name]+=text+" "					

		pp.file_put_contents("./temp/_t5.json", json.dumps(trans_map))

	#print("Test WER:", eval_accuracy / len(testX))




def generate_for_one(x): #(B, S, C)
	global model
	if not model:
		model = Modified_TRF(CFG)
		path = "./model_temp/checkpoints1_038WER/223_37.731_model.pt"
		checkpoint = torch.load(path, map_location='cpu')
		model.load_state_dict(checkpoint['model_state_dict'])
		if gpu: model.cuda()
		model.eval()
	
	if x.size()[1] < 4: return ""

	time1 = time.time()
	#x = torch.from_numpy(x) #x is one dimensional numpy array
	#x = x[None, :] #add batch dimension
	#if gpu: x = x.to(device)	
	with torch.no_grad():
		logits =  model.generate(x)
	if gpu: logits = logits.detach().cpu().numpy()
	else: logits = logits.numpy()
	text = basetokenizer.convert_tokens_to_string(basetokenizer.convert_ids_to_tokens(logits[0], skip_special_tokens=True))	
	
	execution_time1 = int((time.time() - time1) * 1000)
	print(execution_time1, "ms generate:", text)
	return text


def temp():
	tokenizer = basetokenizer
	text = "бас бергенге ас бер"
	y = tokenizer(text, add_special_tokens=True, return_tensors='pt', truncation=False, padding=False)['input_ids'][0]
	tokens = tokenizer.convert_ids_to_tokens(y, skip_special_tokens=True)
	st = tokenizer.convert_tokens_to_string(tokens)
	print(st)


def demo():
	import gradio as gr
	demo = gr.Interface(
			demo_transcribe,
			gr.Audio(sources="upload"),
			"text",
			examples="./temp/examples/"
	)
	demo.launch(share=True)	



def demo_transcribe(audio):
	sr, speech_array = audio
	float32_array = speech_array.astype(np.float32)
	# Normalize the values to the range [-1.0, 1.0]
	float32_array /= 32768.0
	speech_array = float32_array	
	input_values = processor2([speech_array], sampling_rate=sr, return_tensors="pt", padding=True, return_attention_mask=False).input_values.to(device)
	with torch.no_grad():
		logits = model2(input_values).logits.to(device)	
	text = generate_for_one(logits)
	print("demo_transcribe:", text)
	return text



##########################################################################################3


# If there's a GPU available...
gpu = False
if 1==1 and torch.cuda.is_available():
	device = torch.device("cuda:0")
	gpu = True
	print('There are %d GPU(s) available.' % torch.cuda.device_count())
	print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
	print('No GPU available, using the CPU instead.')
	device = torch.device("cpu")
print('-'*60)


#global variables
model = None
X, Y, audio_names = [], [], []
wer_metric = evaluate.load("wer") 

#wav2vec
if 1==1:	
	processor2 = Wav2Vec2Processor.from_pretrained("aismlv/wav2vec2-large-xlsr-kazakh") #wav2vec2-large-xlsr-kazakh
	vocab = processor2.tokenizer.get_vocab()
	vocab = [x[0] for x in list(vocab.items())]
	print(vocab) #size=45
	model2 = Wav2Vec2ForCTC.from_pretrained("aismlv/wav2vec2-large-xlsr-kazakh")
	model2.to(device)



if __name__=="__main__":
	#prepare_data(2000000)
	#train()
	#test()
	demo()
	#temp()
	
	
	