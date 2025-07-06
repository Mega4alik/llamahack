"""
curl -X POST https://anuarsh--rl-reward-web-inference.modal.run -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"What is the price?"}, {"role":"assistant","content":"it is based on service usage from 2.99 to 4 USD"}, {"role": "user", "content": "On a scale from 1 to 5, how correct and helpful was the assistants last response based on history of conversation? Only return a single number. Ex, 4"} ]}'


curl -X POST https://anuarsh--rl-reward-web-inference.modal.run -H "Content-Type: application/json" -d '{"messages":[
{"role":"user","content":"What is the price?"}, 
{"role":"assistant","content":"it is based on service usage from 2.99 to 4 USD"}, 
{"role": "user", "content": "On a scale from 1 to 5, how correct and helpful was the assistants last response based on history of conversation? Only return a single number. Ex, 4"} ]}'


# gives 5
curl -X POST https://anuarsh--rl-reward-web-inference.modal.run -H "Content-Type: application/json" -d '{"messages":[
{"role":"assistant","content":"Hi, I am a smart chatbot. I can understand the context and respond in various ways. Thank you for your interest. Would you like to automate your conversations with your leads?"},
{"role":"user","content":"Yes, try webapi.ai for free"},
{"role":"assistant","content":"Awesome, may I know what business are you in?"},
{"role": "user", "content": "Task: On a scale from 1 to 5, how correct was the last assistant response? Only return a single number. Ex, 4"} ]}'


"""

curl -X POST https://anuarsh--rl-reward-web-inference.modal.run -H "Content-Type: application/json" -d '{"messages":[
{"role":"assistant","content":"Hi, I am a smart chatbot. I can understand the context and respond in various ways. Thank you for your interest. Would you like to automate your conversations with your leads?"},
{"role":"user","content":"Yes, try webapi.ai for free"},
{"role":"assistant","content":"Great, may I know what business you are in?"},
{"role": "user", "content": "Task: Is the last assistant response correct? Just answer Yes or No"} ]}'
