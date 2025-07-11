# Reward function for landing RL training that scores assistant answer from 1 to 5
# modal deploy <name>.py
# curl -X POST https://anuarsh--rl-reward-web-inference.modal.run -H "Content-Type: application/json" -d '{"user_id":1, "messages":[{"role":"user","content":"hi"}, {"role":"assistant","content":"hi, how may I help you?"}]}'

import json, os, copy
from typing import Dict
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen2ForCausalLM, LlamaForCausalLM
import modal


gp = """
Intro:
- If user wants to connect with an agent or manager, suggest switching to a human agent
- Make sure that user provided a valid email and name before calling the "just-try" action
- Feel free to speak other languages
- If an answer is not given below, say that you don't know it

=== FAQ ===
Q: How do I integrate with external services or APIs?
1) First of all, make sure you have an API that receives GET/POST parameters and responds in JSON format. 
2) Secondly, go to "Slots and Actions" page to configure your input/output slots (parameters), and actions.  
- Input slots are the variables sent to your API. Its values are collected during the conversation. 
- Output slots are variables collected from your API response. Its values are used to generate an answer. 
- Each action requires input/output slots that it's going to use, as well as the API parameters (Endpoint URL, Header, etc)
3) Then go to "Dialogue Scenarios" page to give examples of how the action that you just configured should be called. 
The following article should also be helpful - https://s1.webapi.ai/article/104


Q: Do you support other languages?
A: As a language model, it has been trained on large amounts of text data from various languages and can understand and respond in many different languages including, but not limited to English, Spanish, French, German, Italian, Chinese, Japanese, Korean, Arabic, Russian, and Portuguese. However, the level of proficiency may vary depending on the language and the amount of training data available for it.

Q: Who is your creator?
A: AILabs Technologies, Inc

Q: How to do Fine-tuning?
A: Fine-tuning is a complex process that requires 100+ (preferably) scenarios and engagement from our side. In most cases, fine-tuning is not needed. Please go ahead and book a quick call with our team member https://calendly.com/webapi so we can understand your needs and give our best recommendations

Q: I haven't received an email after registration
A: We activate new accounts manually, so it may take up to 12 hours. You will get your credentials soon

Q: The generated bot will be the same as chatGPT and Chatsonic? In that case, do I need an OpenAI API? Or is just a service chatbot that can aswer just product related questions like any other chatbot from any website?
A: You don't need the OpenAI API key to use our platform. The platform already has one. Yes, it is mainly built to help companies automate their customer support by answering product-related questions and performing actions (retrieving information from API, etc). The difference from other platforms is that it understands the context and is fast to get started. But since it's GPT4-based,  you may expect it to show expert knowledge in many areas as ChatGPT does. I suggest trying it for free by giving instructions on how it should act and providing sample dialogue.

Q: How do I connect human agents?
A: There's a built-in action called "connect-human-agent" that transfers the user to a human agent. The "Lead Qualifier" template has this capability. Detailed instructions are available at https://s1.webapi.ai/article/91

Q: Is the chatbot hosted on a domain of our choosing?
A: No, all chatbots are hosted on our webapi.ai subdomains

Q: What is the cost?
A: We charge based on service usage of $0.99-$4 per 100 bot responses. All new users are granted $5 or about 170 bot responses for free.
Basically, usage is counted in the number of responses or actions that an AI chatbot generates. Learn detailed pricing at https://www.webapi.ai/files/pricing.pdf

Q: Do you have a youtube video on this
A: Videos with different bot development examples can be found at https://www.webapi.ai/#samples

Q: Are there user roles?
A: Yes, the roles are: agent, admin, superadmin

Q: How do you sign in/sign up?
A: Go to https://accounts.webapi.ai/ to create a new account or sign in to your existing account

Q: Can you integrate into whmcs?
A: The are hundreds of ready integrations on Zapier and Pabbly. Learn more at https://s1.webapi.ai/article/104

Q: Will the chatbot search the internet for the answer?
A: No, it will only use information from Domain Knowledge and Dialogue Scenarios

Q: Where are your servers located physically? 
A: We use AWS servers located in the US

Q: Are you GDPR compliant?
A: Yes, check our GDPR compliance article at https://s1.webapi.ai/article/128

Q: What happens if OpenAI GPT-3/4 goes offline?
A: In that case, we will be moving users to human agents 

Q: Can I upload documents?
A: Yes, we have the "Documents" feature that allows uploading documents. The AI will search through these documents to find an answer to the user's question. 

Q: How many documents can I upload?
A: Up to 100 documents. The document size should not exceed 7000 tokens (~5000 words).

Q: How many chatbots can I make using 1 account
A: There's one chatbot per one webapi account. You may have multiple webapi accounts. Each webapi account has 1 chatbot and an unlimited number of agents and admins

Q: Can I see your roadmap?
A: Sure, here's our roadmap - https://s1.webapi.ai/article/102

Q: Can the chatbot initiate a dialogue?
A: No, it only responds when a user asks a question

Q: Can I change my chatbot name / icon?
A: Yes, you can change the name and icon of your chatbot. This can be done in the chatbot settings in webapi.ai platform. 

Current model: GPT-4o

Supported channels (instructions available at "Channels" page): Telegram, Web, Whatsapp, Facebook Messenger, Instagram, Twilio SMS, Twilio Whatsapp, API for messaging

We have ready integrations with Shopify, Zendesk, Pabbly, Zapier, Calendly, and more. Check out the Pabbly integration video at https://vimeo.com/797265571

Documents articles can be imported from: Zendesk Knowledge base, Intercom knowledge base, Webapi KB.

Regarding partnerships, white-labeling, customization, etc. -  learn about our partnership programs at https://s1.webapi.ai/article/105

# About the Lite version
The Lite version is the easiest to start with, yet very powerful.
It supports Domain Knowledge (Instructions, Knowledge base, FAQ) and Forms.
Forms is probably the most exciting part of the Lite version. You declare the user information that needs to be collected, like email, address, and phone number. Then add "show form [form name]" as an instruction to the Domain Knowledge. That's it, now the chatbot will start asking for the required user information based on your instruction. Once collected, it will be added into our built-in Spreadsheets, emailed to your email address, and sent via integrations (API/Pabbly/Zapier)
Check out the demo at https://www.webapi.ai/#video

<Example dialogue>
AI: Hi, I’m a smart chatbot. I can understand the context and respond in various ways. Thank you for your interest. Would you like to automate your conversations with your leads? BUTTONS= Yes, try webapi.ai for free# ask a question# talk to agent# 
customer: Yes, try webapi.ai for free
AI: Awesome, may I know what business are you in? Ex: educational services
customer: consulting services
AI: Excellent, are you open for a 15-minute demo with our team member? BUTTONS= Book a zoom call# try a demo on my own# 
customer: try a demo on my own
AI: Ok! may I get your name and email, so I can setup your account?
customer: Sure, Arman Sharafudinov, email is mega4alik@gmail.com
AI-action: just-try# name=Arman Sharafudinov# email=mega4alik@gmail.com# business_type=consulting services
outputs: action_status=success
AI: Ok! We are already setting up your account. You will get an email with login details in 12 hours. Thanks!
###

AI: Hi, I’m a smart chatbot. I can understand the context and respond in various ways. Thank you for your interest. Would you like to automate your conversations with your leads? BUTTONS= Yes, try webapi.ai for free# ask a question# talk to agent# 
customer: talk to agent
AI: May I know your name and email address before I connect you with one of our agents?
customer: Sure, it is Choe Long, tester@webapi.ai
AI-action: connect-human-agent# name=Choe Long# email=tester@webapi.ai
outputs: action_status=success
AI: You have been switched to our agent. Usually we respond in up to 5 hours
###

AI: Hi, I’m a smart chatbot. I can understand the context and respond in various ways. Thank you for your interest. Would you like to automate your conversations with your leads? BUTTONS= Yes, try webapi.ai for free# ask a question# talk to agent# 
customer: Yes, try webapi.ai for free
AI: Awesome, may I know what business are you in? Ex: educational services
customer: IT technologies
AI: Excellent, are you open for a 15-minute demo with our team member? BUTTONS= Book zoom call# try a demo on my own# 
customer: Book zoom call
AI: Great! Please book your slot at https://calendly.com/webapi
###

AI: Hi, I’m a smart chatbot. I can understand the context and respond in various ways. Thank you for your interest. Would you like to automate your conversations with your leads? BUTTONS= Yes, try webapi.ai for free# ask a question# talk to agent# 
customer: Can you make a Twilio integration?
AI: I'm not sure. Would like to switch to a human agent? BUTTONS= talk to agent# 
###

"""


class rewardModel(LlamaForCausalLM):
	@torch.inference_mode()
	def process_gp(self, tokenizer):
		#messages = [{"role":"system", "content":gp}]; prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
		prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"+gp+"<|eot_id|>" #llama3
		#print(prompt); exit()
		inputs = tokenizer(prompt, truncation=True, max_length=4000, return_tensors="pt").to("cuda") #NEED TO BE LARGER
		outputs = super().forward(**inputs, use_cache=True, output_hidden_states=True)
		self.past_key_values = outputs.past_key_values


	@torch.inference_mode()
	def generate(self, input_ids=None, max_new_tokens=32, **kwargs): #B=1, S
		cached_kv = copy.deepcopy(self.past_key_values)
		outputs = super().forward( #attention_mask not needed unless B=1
			input_ids=input_ids,
			past_key_values=cached_kv,
			use_cache=True,
			output_hidden_states=True
		)
		past_key_values = outputs.past_key_values
		last_token = input_ids[:, -1:]

		#generate
		generated, cur_token = [], last_token
		for _ in range(max_new_tokens):
			outputs = super().forward(
				input_ids=cur_token,
				past_key_values=past_key_values,
				use_cache=True
			)
			logits = outputs.logits[:, -1, :]
			past_key_values = outputs.past_key_values
			next_token = torch.argmax(logits, dim=-1, keepdim=True)
			generated.append(next_token)
			cur_token = next_token
			if next_token.item() == 128009: break #tokenizer.eos_token_id llama3:128009

		# Decode final output
		gen_ids = torch.cat(generated, dim=-1)
		return gen_ids


#=============== modal.com ====================
app = modal.App("rl-reward")

image = modal.Image.debian_slim().pip_install(
	"torch", "transformers", "accelerate", "fastapi[standard]"
)

@app.cls(gpu="A100", image=image, secrets=[modal.Secret.from_name("hf-token")]) #, keep_warm=1
class ModelRunner:
	@modal.enter()
	def setup(self):
		model_id = "meta-llama/Llama-3.2-3B-Instruct"
		tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=os.getenv("US1"))
		tokenizer.pad_token, tokenizer.pad_token_id = tokenizer.eos_token, tokenizer.eos_token_id
		model = rewardModel.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, token=os.getenv("US1"))
		model.eval()
		model.cuda()
		model.process_gp(tokenizer)
		self.tokenizer, self.model = tokenizer, model    		

	@modal.method()
	def score_answer(self, messages): #must not include system message
		#messages[-1]["content"]+="\nTask: rate this answer by correcteness and helpfulness on scale from 1 to 5, where 1 is bad and 5 is good answer. Ex, 4" #v1
		#messages.append({"role": "user", "content": "On a scale from 1 to 5, how correct and helpful was the assistant’s last response based on history of conversation? Only return a single number."}) #v2
		prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)[134:] #qwen2-57, llama3-134
		#print("---"+prompt+"---")
		input_ids = self.tokenizer([prompt], return_tensors="pt").input_ids.to("cuda")
		gen_ids = self.model.generate(input_ids, max_new_tokens=5).detach().cpu()
		del input_ids
		#torch.cuda.empty_cache() #?
		return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)

		
@app.function(image=image)
@modal.web_endpoint(method="POST")
def web_inference(req: Dict):
	messages = req["messages"]
	return ModelRunner().score_answer.remote(messages)

#=============== ./endOf modal.com ====================

"""
if __name__ == "__main__":
	device = torch.device("cuda")
	tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct") #"Qwen/Qwen2-0.5B-Instruct"
	tokenizer.pad_token = tokenizer.eos_token
	tokenizer.pad_token_id = tokenizer.eos_token_id
	tokenizer.truncation_side = 'left'

	model = rewardModel.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
	model.eval()
	model.cuda()
	model.process_gp()
	
	# sample
	messages = [{"role":"user", "content":"I need pricing"}, {"role":"assistant", "content":"We charge based on usage"}]
	messages[-1]["content"]+="\nTask: rate this answer by correcteness and helpfulness on scale from 1 to 10, where 1 is bad and 5 is good answer. Ex, 4"
	prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)[134:] #qwen2-57, llama3-
	#print("---"+prompt+"---");exit()
	input_ids = tokenizer([prompt], return_tensors="pt").input_ids.to(device)
	gen_ids = model.generate(input_ids, max_new_tokens=10)
	print(tokenizer.decode(gen_ids[0], skip_special_tokens=True))
"""