import requests

# URL of the API
url = "http://127.0.0.1:8000"

# Example data to send to the API
data = {
    "features": [1, 2, 3, 4]  # Replace with your actual feature values
}

# Send a POST request to the API
response = requests.post(url, json=data)

# Print the response
print(response.json())