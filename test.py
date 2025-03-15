import requests

url = "http://127.0.0.1:5000/analyze"
data = {"text": "I love this product!"}
response = requests.post(url, json=data)

print(response.json())  # Should print sentiment result
