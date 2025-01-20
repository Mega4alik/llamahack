#from ..utils import file_get_contents
import random
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import MarkdownElementNodeParser, MarkdownNodeParser

# bring in our LLAMA_CLOUD_API_KEY
#from dotenv import load_dotenv
#load_dotenv()

def temp():
	# set up parser
	parser = LlamaParse(
		api_key=LLAMAPARSE_API_KEY,
	    result_type="markdown"  # "markdown" and "text" are available
	)

	# use SimpleDirectoryReader to parse our file
	file_extractor = {".pdf": parser}
	documents = SimpleDirectoryReader(input_files=['data/3M_2018_10K.pdf'], file_extractor=file_extractor).load_data()
	
	node_parser = MarkdownNodeParser(
	    chunk_size=50*5,
	    include_metadata=False,
	    #llm=OpenAI(model="gpt-3.5-turbo-0125", api_key=OPENAI_API_KEY), num_workers=8    
	)
	
	k = 8
	nodes = node_parser.get_nodes_from_documents(documents[k:k+2])
	print(len(documents), len(nodes), )

	for node in nodes:
	    print("----- Node:", node.text, "\n\n")


if __name__=="__main__":
	temp()

