# venv: US1 asr3.12
# landing RL training locally with WER reward

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, GRPOTrainer, GRPOConfig
from datasets import Dataset
import evaluate
import torch
from webapi_data import prepare_data


def preprocess(sample):
	#global labels_map
	messages = sample["messages"] #[{"role":"system", "content":gp}] + 
	prompt = str(messages) #tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	#labels_map[prompt[-1000:]] = sample["label"]
	#encoded = tokenizer(prompt, truncation=True, padding="max_length", max_length=4000, return_tensors="pt")
	#return {"input_ids": encoded["input_ids"][0], "attention_mask": encoded["attention_mask"][0], "label": sample["label"]}
	return {"prompt":prompt}

def wer_reward(reference: str, generated: str) -> float:
	wer_score = wer_metric.compute(predictions=[generated], references=[reference])
	return 1.0 - wer_score  # higher is better

def reward_fn(prompts, completions, completion_ids):
	rewards = []	
	for prompt, generated in zip(prompts, completions):
		ref = "aa" #labels_map[prompt[-1000:]]		
		rewards.append(wer_reward(ref, generated))
		#print(generated, "ref:", ref, "\n===============\n")
	print("rewards:", rewards)
	return rewards

# ------------------------------
mode = 1 #1-train, 2-test
print("Loading model and tokenizer...")
model_name = "gpt2" #"Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation = 'left'
wer_metric = evaluate.load("wer")

#base_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True) #, torch_dtype=torch.float16
#model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)

# --------- Dataset -------------
d, gp = prepare_data(mode)
labels_map = {}
dataset = Dataset.from_dict(d)
dataset = dataset.map(preprocess)

# -----------------------------
training_args = GRPOConfig(
	output_dir="./model_temp/",    
	learning_rate=5e-6,
	#generation_batch_size=2,
	num_generations=2,
	per_device_train_batch_size=2, #8
	gradient_checkpointing=True,
	num_train_epochs=100,
	bf16=True,
	remove_unused_columns=True,
)

trainer = GRPOTrainer(
	model=model_name,
	args=training_args,
	train_dataset=dataset,
	reward_funcs=reward_fn
)

trainer.train()
