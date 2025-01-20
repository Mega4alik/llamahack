import re
import io
from openai import OpenAI
from config import DEEPINFRA_API_KEY

openai = OpenAI(api_key=DEEPINFRA_API_KEY, base_url="https://api.deepinfra.com/v1/openai")
#client = OpenAI(api_key=OPENAI_API_KEY)

def file_put_contents(filename, st):
	file = codecs.open(filename, "w", "utf-8")
	file.write(st)
	file.close()

def file_get_contents(name):
	f = io.open(name, mode="r", encoding="utf-8") #utf-8 | Windows-1252
	return f.read()


def openai_run(system_prompt, user_message):
	messages = [{"role":"system", "content":system_prompt}, {"role":"user", "content":user_message}]    
	completion = client.chat.completions.create(
	  model="gpt-4o-mini", #"gpt-4o-2024-05-13",
	  temperature=0,
	  max_tokens=2000,
	  messages=messages
	)
	message = completion.choices[0].message
	return message.content    


def anthropic_run(system_prompt, user_message):
   import anthropic 
   client = anthropic.Anthropic(  
	api_key=ANTHROPIC_API_KEY,
   )
   message = client.messages.create(
	model="claude-3-sonnet-20240229", #"claude-3-opus-20240229",
	max_tokens=4096,
	system=system_prompt,
	messages=[
	 {"role": "user", "content": user_message}
	]
   )
   return message.content[0].text


def deepinfra_run(system_prompt, user_message):
	chat_completion = openai.chat.completions.create(
		model="meta-llama/Meta-Llama-3.1-405B-Instruct",
		messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
		max_tokens=4096
	)
	return chat_completion.choices[0].message.content



def get_llm_answer(chunks_content, user_message): #keywords + content
	gp = "Is answer is not given below, say that you don't know it. Make sure to copy answers from documents without changing them."+chunks_content
	answer = deepinfra_run(gp, user_message)
	return answer



def parse_keywords(content):
	result = []
	lines = content.strip().split('\n')
	current_chunk = None
	inline_pattern = re.compile(r'^\s*[^#:]+\s*:\s*(.+)$')  # Matches lines like "Chunk1: word1, word2"
	section_pattern = re.compile(r'^###\s*[^#]+\s*###$')   #re.compile(r'^###\s*[^#\n]+\s*\d*\s*###$')
	#section_pattern = re.compile(r'^###\s*[^#\n]+\s*\d*\s*###\s*$|^[^#]+', re.DOTALL)
 
	for line in lines:
		line = line.strip()
		if not line: continue
		inline_match = inline_pattern.match(line)

		if inline_pattern.match(line):
			words_str = inline_match.group(1)
			words = [word.strip() for word in words_str.split(',') if word.strip()]
			result.append(words)            

		elif section_pattern.match(line):
			if current_chunk: result.append(current_chunk)
			current_chunk = []

		elif current_chunk is not None: #section_pattern continuation
			words = [word.strip() for word in line.split(',') if word.strip()]
			current_chunk.extend(words)

	if current_chunk: result.append(current_chunk)
	return result



def generate_contextual_keywords(chunked_content):
	system_prompt = '''
	Each chunk is separated as ### Chunk [id] ###. For each chunk generate keywords required to fully understand the chunk without any need for looking at the previous chunks.
	Don't just say "List of services", because its unclear what services are you referring to. Make sure to cover all chunks.
	Sample output:
	Chunk 1: BMW X5, pricings in France
	Chunk 2: BMW X5, discounts
	'''
	keywords_st = deepinfra_run(system_prompt, chunked_content)
	print("Keywords_st:\n", keywords_st, "\n")
	keywords = parse_keywords(keywords_st)    
	return keywords


def generate_questions_bychunk(chunks):
	system_prompt = '''
 Given a chunk from document. Generate 3-5 questions related to the chunk. Each question must be full and not require additional context. 
 Example output:
 1. How to open new account?
 2. How much BMW X5 costs? 
	'''	
	for idx, chunk in enumerate(chunks):
	   text = "#"+chunk["keywords"]+"\n"+chunk["content"]
	   out =  deepinfra_run(system_prompt, text) #anthropic_run(system_prompt, text)
	   question_pattern = re.compile(r'^\s*\d+\.\s+(.*)', re.MULTILINE)
	   questions = question_pattern.findall(out)
	   chunk["questions"] = questions
	   chunk["idx"] = idx
	return chunks

	


if __name__=="__main__":
	st = '''
Here are the keywords required to fully understand each chunk:

### Chunk 1 ###
* 3M, business segments, health care, electronics, energy, products, services

### Chunk 2 ###
* 3M, electronics, energy, products, infrastructure protection, renewable energy, markets

### Chunk 3 ###
* 3M, consumer business, products, office supplies, home improvement, health care, brands (Scotch, Post-it, Filtrete, etc.)

### Chunk 4 ###
* 3M, research and development, patents, expenses, innovation, products

### Chunk 5 ###
* 3M, executive officers, leadership, management, organizational structure

### Chunk 6 ###
* 3M, executive officers, leadership, management, organizational structure (continued)

### Chunk 7 ###
* 3M, forward-looking statements, financial performance, market conditions, risks, uncertainties

### Chunk 8 ###
* 3M, forward-looking statements, risks, uncertainties, assumptions, expectations

### Chunk 9 ###
* 3M, risk factors, global economy, politics, capital markets, competition, credit ratings, funding costs 
'''
	print( parse_keywords(st) )

