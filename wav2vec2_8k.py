import soundfile as sf
import numpy as np
import json
import os
import time
from datasets import load_dataset
from dataclasses import dataclass, field
import evaluate
from transformers import Trainer, TrainingArguments
#from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, HubertForCTC
from transformers import Wav2Vec2Processor
#from transformers1.models.wav2vec2.feature_extraction_wav2vec2 import Wav2Vec2FeatureExtractor 
#from transformers1.models.wav2vec2.processing_wav2vec2 import Wav2Vec2Processor
from transformers1.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2ForCTC
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List, Optional, Union
import preprocess as pp

#sampling_rate = 16000
sampling_rate = 8000
data_path = "/home/mega4alik/Desktop/data"

def temp():
	device = "cpu"
	summ = 0
	for p in model.parameters():
		#print(p.numel(), p.shape)
		summ+=p.numel()
	print("params sum:", summ)	
	
	#print(model.wav2vec2.feature_extractor.conv_layers[0])
	speech_array, sr = sf.read("./temp/"+("8" if sampling_rate==8000 else "16")+"k.wav")
	input_values = processor([speech_array], sampling_rate=sampling_rate, return_tensors="pt", padding=True, return_attention_mask=False).input_values.to(device)
	#print(speech_array[:5], len(speech_array))
	print("input_values:", input_values.shape, input_values[0])

	t1 = time.time()
	with torch.no_grad():
		logits = model(input_values).logits.to(device)
	print(time.time() - t1)

	predicted_ids = torch.argmax(logits, dim=-1)
	print(processor.batch_decode(predicted_ids))
	

def compute_metrics(pred):
	pred_logits = pred.predictions
	pred_ids = np.argmax(pred_logits, axis=-1)
	pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
	pred_str = processor.batch_decode(pred_ids)
	label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
	wer = wer_metric.compute(predictions=pred_str, references=label_str)
	return {"wer": wer}


@dataclass
class DataCollatorCTCWithPadding:
	processor: Wav2Vec2Processor
	padding: Union[bool, str] = True
	max_length: Optional[int] = None
	max_length_labels: Optional[int] = None
	pad_to_multiple_of: Optional[int] = None
	pad_to_multiple_of_labels: Optional[int] = None

	def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
		# split inputs and labels since they have to be of different lenghts and need different padding methods

		input_features = [{"input_values": feature["input_values"]} for feature in features]
		label_features = [{"input_ids": feature["labels"]} for feature in features]

		batch = self.processor.pad(
			input_features,
			padding=self.padding,
			max_length=self.max_length,
			pad_to_multiple_of=self.pad_to_multiple_of,
			return_tensors="pt",
		)

		with self.processor.as_target_processor():
			labels_batch = self.processor.pad(
				label_features,
				padding=self.padding,
				max_length=self.max_length_labels,
				pad_to_multiple_of=self.pad_to_multiple_of_labels,
				return_tensors="pt",
			)

		# replace padding with -100 to ignore loss correctly
		labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

		batch["labels"] = labels
		#print(self.padding, "- self_padding", self.max_length_labels, "- max_length_labels", self.pad_to_multiple_of_labels, "- pad_to_multiple_of_labels")
		return batch



def speech_file_to_array_fn_batched(o):
	files = o["file"]
	texts = o["text"]
	#lens = o["len"]
	d = {"speech":[], "sampling_rate":[], "target_text":[], "len":[]}
	for i in range(len(files)):
		file_path = files[i].replace("/media/sda/ASR/data", data_path)		
		file_path = file_path.replace("/cut", "/8k")
		speech_array, sampling_rate = sf.read(file_path)
		d["speech"].append(speech_array)
		d["sampling_rate"].append(sampling_rate)
		d["target_text"].append(texts[i])
		d["len"].append(len(speech_array)) #lens[i]
	return d


def prepare_dataset(batch):
	batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values
	#batch["len"] = [len(x) for x in batch["input_values"]] #if len is not given
	with processor.as_target_processor():
		batch["labels"] = processor(batch["target_text"]).input_ids
	return batch




###################### __main__ ##########################
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h")
#feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-robust-ft-swbd-300h") 
processor.feature_extractor.sampling_rate = sampling_rate

#processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-xlarge-ls960-ft")  #facebook/wav2vec2-large-robust-ft-swbd-300h(main)  facebook/hubert-xlarge-ls960-ft(hubert
print(len(processor.tokenizer), "= processor.tokenizer.len.", processor.tokenizer.get_vocab())

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
wer_metric = evaluate.load("wer") 

model = Wav2Vec2ForCTC.from_pretrained( # Wav2Vec2ForCTC HubertForCTC
	"facebook/wav2vec2-large-robust-ft-swbd-300h",
	gradient_checkpointing=True,
	ctc_loss_reduction="mean",
	pad_token_id=processor.tokenizer.pad_token_id
)
model.config.ctc_zero_infinity = True
#model.freeze_feature_extractor() #unfreeze in the end
#print( model.wav2vec2.encoder.get_params_count() ) #310M / 315M
model.wav2vec2.encoder._freeze_parameters()
print("model loaded")
#temp()
#exit() #stop

training_args = TrainingArguments(
  output_dir="./model_temp/",
  group_by_length=True, length_column_name="len",
  per_device_train_batch_size=8, #8-only encoder, 6- full 
  evaluation_strategy="steps",
  num_train_epochs=15,
  fp16=True,
  save_steps=20000, 
  eval_steps=20000, #1000
  logging_steps=20000,
  learning_rate=1e-6, #1e-4 -> 1e-6
  dataloader_num_workers=2,
  ignore_data_skip=True,
  weight_decay=0.005,
  warmup_steps=1000,
  load_best_model_at_end=True, #save only best model
)


#prepare data
train_list = [ f"{data_path}/swbd/train.jsonl"]
test_list =  [f"{data_path}/swbd/test.jsonl"]

mydataset = load_dataset('json', data_files={'train':train_list, 'test':test_list})
print("\nrunning speech_file_to_array_fn")
mydataset = mydataset.map(speech_file_to_array_fn_batched, num_proc=6, batch_size=32, batched=True)
print("\nrunning prepare_dataset")
mydataset = mydataset.map(prepare_dataset,  batch_size=16, num_proc=6, batched=True)
train_dataset = mydataset["train"]
val_dataset = mydataset["test"]
#endOf prepare data


#print("configuration = ", model.config)
print("\n\nstarting training")
trainer = Trainer(
	model=model,
	data_collator=data_collator,
	args=training_args,
	compute_metrics=compute_metrics,
	train_dataset=train_dataset,
	eval_dataset=val_dataset,
	tokenizer=processor.feature_extractor,
)

trainer.train("./model_temp/checkpoint-120000") #checkpoint to start from  "./model_temp/checkpoint-140000"
trainer.save_model()
