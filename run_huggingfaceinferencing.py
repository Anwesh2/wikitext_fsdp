import requests

API_TOKEN = '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'
headers = {"Authorization": f"Bearer {API_TOKEN}"}
API_URL = "https://api-inference.huggingface.co/models/Anwesh0127/textgeneration"
# jsoninput = {"inputs": "The answer to the universe is", 'max_new_tokens' : 250, 'num_return_sequences':5 }

jsoninput = {"inputs": "The answer to the universe is", 
             'max_new_tokens' : 250,
            'temperature': 0.1,
             'top_p': 1
            }


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload, timeout = 120)
    return response.json()
    
data = query(jsoninput)

print(data)