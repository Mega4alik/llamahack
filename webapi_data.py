import json
from utils import file_get_contents, file_put_contents

def is_unique(hmap, messages):
	if len(messages)>3: return True
	h = str(messages)
	if h in hmap: return False
	hmap[h] = True
	return True


def dataset_to_dict(dataset):
	d = {}
	for (project, event, messages, label) in dataset:
		for o in [ ("messages", messages), ("event", event), ("label", label)]:
			k, v = o[0], o[1]
			if k not in d: d[k] = []
			d[k].append(v)
	return d


def prepare_data(mode):
	hmap = {}
	jj = json.loads(file_get_contents("./temp/test_history.json" if mode==2 else "./temp/landing_history.json"))
	arr = sorted(jj["data"], key=lambda x: (x['user_id'], x['id']))	
	cuid = None
	dataset, messages = [], []
	for x in arr:
		event, props = x["event"], json.loads(x["props"])
		if x["user_id"]!=cuid or event=="BotClosedChat": messages = []
		cuid = x["user_id"]

		if event=="BotAction": #promt: AI-action: connect-human-agent# name=patto# email=pattocarvente@gmail.com
			ans = "AI-action: "+props["action"]
			for q in props["inputs"]: ans+="# "+q["name"]+"="+q["value"]
			if len(messages)>0 and is_unique(hmap, messages): dataset.append(("landing", event, messages[-10:], ans))

			ans+="\noutputs: action_status=success"	#\nAI
			#for q in props["inputs"]: ans+="# "+q["name"]+"="+q["value"]
			messages.append({"role":"assistant", "content":ans})

		elif event=="UserMessage":
			if "content" not in props: continue
			content = props["content"]
			if content=="/start": messages = []
			else: messages.append({"role":"user", "content":content})

		elif event=="BotMessage":
			content = props["content"]
			if len(messages)>0 and is_unique(hmap, messages): dataset.append(("landing", event, messages[-10:], "AI: "+content))
			messages.append({"role":"assistant", "content":content})
	print("dataset size", len(dataset))
	return ( dataset_to_dict(dataset), file_get_contents("./temp/landing_gprompt.txt"))



if __name__=="__main__":
	dataset, gp = prepare_data()
	file_put_contents("./temp/temp.json", json.dumps(dataset, indent=2))