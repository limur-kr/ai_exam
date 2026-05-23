import requests

# api_key
apiKey = "85bd1238"

url = f"http://www.omdbapi.com/?s=movie&y=2026&apikey={apiKey}&page=1"
response = requests.get(url)

print(response.text)