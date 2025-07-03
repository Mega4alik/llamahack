# asr3.8
# !pip install predibase
import json
from transformers import AutoTokenizer
from predibase import Predibase, SFTConfig
from webapi_data import prepare_data
from config import PREDIBASE_API_KEY


def prepare_dataset(): #my_dataset1
	d, gprompt = prepare_data(1)
	for i in range(len(d["messages"])):
		messages, label = d["messages"][i], d["label"][i]
		#print(messages, "\n-- Label:", label);exit()
		messages = [{"role":"system","content":gprompt}] + messages + [{"role":"assistant", "content":label}]
		q = {"messages":messages}
		print(json.dumps(q))


def train():
	pb = Predibase(api_token=PREDIBASE_API_KEY)
	#create dataset only once 
	#dataset = pb.datasets.from_file("./temp/predibase.jsonl", name="my_dataset1"); exit()
	
	# Create an adapter repository
	repo = pb.repos.create(name="landing-adapter-sft1", description="SFT on landing. my_dataset1: 4.5k rows of messages", exists_ok=True)

	adapter = pb.adapters.create(
		config=SFTConfig(
			base_model="qwen2-5-1-5b-instruct",
			apply_chat_template=True  # Set to True if your dataset doesn't already have the chat template applied
		),
		dataset="my_dataset1",
		repo="landing-adapter-sft1",
		description="SFT on landing. my_dataset1: 4.5k rows of messages",
	)


def evaluate():
	pb = Predibase(api_token=PREDIBASE_API_KEY)
	client = pb.deployments.client("qwen2-5-1-5b-instruct")
	#client = pb.deployments.client("qwen3-8b")
	d, gprompt = prepare_data(2)
	tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

	for i in range(len(d["messages"])):
		messages, label = d["messages"][i], d["label"][i]
		messages = [{"role":"system", "content":gprompt}] + messages
		prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
		resp = client.generate(prompt, adapter_id="landing-adapter-sft1/2")
		print(resp.generated_text, "Label:", label, "\n===========\n")


#====================================
#prepare_dataset()
#train()
evaluate()
