import json
import os
import numpy as np
import soundfile as sf
import random
import time
import datetime
from datasets import load_dataset #, load_metric
import evaluate
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModel, AdamW, get_linear_schedule_with_warmup
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM


#from transformers import Wav2Vec2ForCTC #Wav2Vec2FeatureExtractor,
from transformers1.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForCTC
from transformers1.models.wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor 
from transformers.models.wav2vec2.processing_wav2vec2 import Wav2Vec2Processor


import torch
import torch as T
from torch import nn
from torch.nn.utils.rnn import pad_sequence

import preprocess as pp


sampling_rate = 8000 #16000


def temp():
	summ = 0
	for p in model.parameters():
		#print(p.numel(), p.shape)
		summ+=p.numel()
	print("params sum:", summ)	
	
	#print(model.wav2vec2.feature_extractor.conv_layers[0])
	speech_array, sr = sf.read("./temp/"+("8" if sampling_rate==8000 else "16")+"k.wav")
	input_values = feature_extractor([speech_array], sampling_rate=sampling_rate, return_tensors="pt", padding=True, return_attention_mask=False).input_values.to(device)
	#print(speech_array[:5], len(speech_array))
	print("input_values:", input_values.shape, input_values[0])

	with torch.no_grad(): 
		logits = model(input_values).logits.to(device)

	predicted_ids = torch.argmax(logits, dim=-1)
	print(processor.batch_decode(predicted_ids))
	

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

if 1==1: #test: wav2vec
	model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
	processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
	feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h") 
	feature_extractor.sampling_rate = sampling_rate
	#model.config.conv_kernel[0]	= 10
	#model.config.conv_stride[0]	= 2
	model.config.num_codevectors_per_group = 160
	model.to(device)


if __name__== "__main__":
	temp()



