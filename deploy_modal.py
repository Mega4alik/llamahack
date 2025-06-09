#pip install modal
# modal deploy deploy_modal.py
"""

curl -X POST https://anuarsh--hf-transformer-model-web-inference.modal.run/web-inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Once upon a time"}'


curl -X POST https://anuarsh--hf-transformer-model-web-inference.modal.run -H "Content-Type: application/json" -d '{"prompt": "Hi there"}'

"""

import modal
from typing import Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

app = modal.App("hf-transformer-model")

image = modal.Image.debian_slim().pip_install(
    "torch", "transformers", "accelerate", "fastapi[standard]"
)

"""
# Define a function to run remotely
@app.function(image=image, gpu="any", timeout=600)
def run_inference(prompt: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    model_name = "Qwen/Qwen2-0.5B-Instruct"  # or your HF model like "meta-llama/Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(output[0], skip_special_tokens=True)

 

"""

@app.cls(gpu="any", image=image, timeout=600)
class ModelRunner:
    @modal.enter()
    def setup(self):
        model_name = "Qwen/Qwen2-0.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")

    @modal.method()
    def run(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**inputs, max_new_tokens=50)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)



#@app.function(image=image)
#@modal.fastapi_endpoint(docs=False)
#def web_inferface(prompt: str) -> str:
#    return ModelRunner().run.remote(prompt)


@app.function(image=image)
@modal.web_endpoint(method="POST")
def web_inference(req: Dict):
    prompt = req["prompt"]
    #return run_inference.remote(prompt)
    return ModelRunner().run.remote(prompt)  