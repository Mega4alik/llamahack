import json
import os
import numpy as np
import soundfile as sf
import random
import time
import datetime
import webrtcvad
from pydub import AudioSegment
from pydub.utils import make_chunks
from datasets import load_dataset #, load_metric
import evaluate
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModel, AdamW, get_linear_schedule_with_warmup
from transformers import Trainer, TrainingArguments
from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
import torch
import torch as T
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import preprocess as pp


llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
llama_model =  LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

config = llama_model.config


class CFG:  
  maximum_length_seq = 150
  input_shape = (512, 32) #2048


class Modified(nn.Module):
	"""
	Input shape: (X,32)
	"""
	def __init__(self, CFG=CFG):
		super().__init__()
		self.transformer = llama_model

		#v1
		#self.fc = nn.Linear(CFG.input_shape[-1], CFG.trf_hiddim)
		
		#v2
		#self.cnn = nn.Conv1d(CFG.input_shape[-1], CFG.input_shape[0], 4, stride=2)
		self.cnn = nn.Sequential( #v2.1
			nn.Conv1d(in_channels=CFG.input_shape[-1], out_channels=CFG.input_shape[0], kernel_size=4, padding=2),
			nn.ReLU(),
			nn.BatchNorm1d(CFG.input_shape[0])
		)
		self.loss_fn = nn.CrossEntropyLoss()
		#self.model.model.embed_tokens = self.cnn	

	def trans(self, x):
		print(x.size(), x[0].size())
		x = x.transpose(2,1)
		z = self.cnn(x)
		z = z.transpose(2,1)
		return x


	def generate(self, x):
		#return self.transformer.generate(inputs_embeds=self.trans1(x), max_new_tokens=self.maxlen) #stable for validation swbd
		return self.transformer.generate(x)
		

	def forward(self, x, y):		
		out =  self.transformer.forward(x) #labels=y
		#logits = torch.argmax(out.logits, dim=-1)\
		logits = out.logits.view(-1, out.logits.size(-1))  # Flatten to shape (3, 128001)		
		y = y.view(-1)  # Flatten to shape (2,)
		print(out.loss, out.logits.shape, logits.shape, y.shape)
		loss = self.loss_fn(logits, y)
		print(loss)		



def temp():
	model = Modified(CFG)
	#model.to(device)
	b_inputs = T.tensor([[128000,   5269,    527,    499,     30]])
	#b_inputs = llama_tokenizer(["how are you?"], return_tensors='pt')["input_ids"]	
	b_labels = T.tensor([[128000,   5269,   30]])  #T.randint(0, 999,(4, 3), dtype=torch.long) #.to(device)	
	print(b_inputs.size(), b_labels.size())	
	out = model(b_inputs, b_labels)	
 


def format_time(elapsed):	
	elapsed_rounded = int(round((elapsed)))
	return str(datetime.timedelta(seconds=elapsed_rounded)) 	# Format as hh:mm:ss


def flat_accuracy2(preds, labels, tokenizer):      
	wers = 0.0
	for i in range(len(preds)):
		l = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(labels[i], skip_special_tokens=True)).lower()
		p = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(preds[i], skip_special_tokens=True)).lower()
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


def prepare_data(dataset, maxlen): #1-swbd, 2-T122
	global X
	global Y
	global audio_names
	if dataset==1:
		pre = "/media/sda/ASR/data/swbd"  #"/hdd/ASR/data/swbd"
		lines = pp.file_get_contents(pre+'/test.jsonl').split("\n") + pp.file_get_contents(pre+'/train.jsonl').split("\n") #swbd
	else:
		lines = pp.file_get_contents('./data/T122/test.jsonl').split("\n") #T122 long 60 second audios
	
	texts = []	
	a = {}
	for line in lines[:maxlen]:
		if not line: continue
		q = json.loads(line)				
		a = q["file"].split("/")
		audio_name = a[-2]
		if dataset==1: logit_name =  a[-1].replace(".wav","") #swbd sw2001A-ms98-a-0019.pt	
		else: logit_name =  a[-2] + "--" + a[-1].replace(".wav","") #T122
		
		logits = torch.load("./temp/logits/"+("swbd" if dataset==1 else "T122")+"/"+logit_name+".pt") #swbd | T122 
		x = torch.squeeze(logits)
		
		#filter
		n = x.size(dim=0)
		if dataset==1 and n>600: continue  #swbd remove longer than 30 seconds
		
		X.append(x)
		texts.append(q["text"].lower().strip())
		audio_names.append(audio_name)


	#Y = basetokenizer([q for q in texts], add_special_tokens=True, return_tensors='pt', truncation=True, padding=True)['input_ids']
	#no padding case
	Y = []
	for q in texts:
		 y = llama_tokenizer(q, add_special_tokens=True, return_tensors='pt', truncation=False, padding=False)['input_ids'][0]		 
		 #y = y.numpy().tolist() # [:-1] + [32100, 1] #append special element?
		 Y.append(y)
	
	print(Y[0], len(Y[0]))
	print(len(X), len(Y), " -----  dataset size")	
	

def train():
	testlen, maxlen = 150, 200 #000
	prepare_data(1, maxlen) #X,Y
	testX = X[:testlen]
	testY = Y[:testlen]
	trainX = X[testlen:]
	trainY = Y[testlen:]

	# Create the DataLoader for our training set.
	#train_data = TensorDataset(X, Y['input_ids'])
	#train_sampler = RandomSampler(train_data)
	#train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=1)


	# ========================================
	#               Training
	# ========================================
	print('-'*20 + ' Starting Trainig ... ' + '-'*20)
	DO_VALIDATE = True
	epochs = 100
	batch_size = 16 #32
	best_val_accuracy = 999
	stepsN = int(len(trainX) / batch_size) #bs  --  steps per epoch
	total_steps = stepsN * epochs  #len(train_dataloader) * epochs

	model = Modified(CFG)
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
				tmp_eval_accuracy = flat_accuracy2(logits, label_ids, llama_tokenizer)
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
	path = "./model_te mp/checkpoints/12_13.843_model.pt"
	checkpoint = torch.load(path, map_location='cpu')
	model.load_state_dict(checkpoint['model_state_dict'])	
	#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	if gpu: model.cuda()
	model.eval()
	
	prepare_data(2) #X,Y 
	testlen = 200000
	testX = X[:testlen]
	testY = Y[:testlen]
	trans_map = {}

	batch_size, eval_accuracy = 2, 0

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

	print("Test WER:", eval_accuracy / len(testX))


def generate_for_one(x): #(B, S, C)
	global model
	if not model:
		model = Modified_TRF(CFG)
		path = "./model_temp/checkpoints/12_13.843_model.pt"
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
	print(execution_time1, "ms t5_generate:", text)
	return text


def split_by_vad(input_wav, aggressiveness=1, chunk_size=30):
	transcription = ""
	audio = AudioSegment.from_wav(input_wav).set_channels(1)  # Ensure mono
	chunks = make_chunks(audio, chunk_size)  # Create chunks of 30ms
	vad = webrtcvad.Vad(aggressiveness)
	
	# Iterate through chunks
	x, ns, speech_cnt, file_id = AudioSegment.empty(), 0, 0, 0
	chunks = chunks[:-1]    
	for i, chunk in enumerate(chunks):
		pcm_data = chunk.raw_data
		is_speech = vad.is_speech(pcm_data, chunk.frame_rate)                
		
		if is_speech==False:
			ns+=1
			if ns<=5: x+=chunk

		if (is_speech==False and ns>2 and speech_cnt>33*2) or (i==len(chunks)-1 and speech_cnt>2) : #end utterance -- at least 2s long
			x.export("./temp/temp.wav", format="wav")      		
			#generate
			speech_array, sampling_rate = sf.read("./temp/temp.wav")
			input_values = processor2([speech_array], sampling_rate=sampling_rate, return_tensors="pt", padding=True, return_attention_mask=False).input_values.to(device)
			with torch.no_grad():
				logits = model2(input_values).logits #.to("cpu")
			out = generate_for_one(logits)
			#endOf generate
			if out: transcription+=out+" "
			x = AudioSegment.empty()
			speech_cnt = 0
		
		if is_speech==True:
			ns = 0
			speech_cnt+=1
			x+=chunk

	return transcription.strip()


def T248_evaluate():
	folder_path = "/home/mega4alik/Desktop/python/ASR_W2V/data/T248/audios16k/"  # Replace with the path to your folder
	files = os.listdir(folder_path)
	files = [os.path.join(folder_path, f) for f in files if f.endswith(".wav")]
	d = {}
	for filename in files:
		audio_name = filename.split("/")[-1].replace(".wav","")
		print(audio_name)
		transcription = split_by_vad(filename)
		d[audio_name] = transcription
		pp.file_put_contents("./temp/t5.json", json.dumps(d))



##########################################################################################3


# If there's a GPU available...
gpu = False
if 1==2 and torch.cuda.is_available():
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
wer_metric = evaluate.load("wer") #load_metric("wer")

if 1==2: #test: wav2vec
	model2 = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
	processor2 = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
	model2.to(device)


if __name__=="__main__":
	#prepare_data(1, 200)
	#train()
	#T248_evaluate()
	#test()
	#generate_for_one([999])
	temp()


