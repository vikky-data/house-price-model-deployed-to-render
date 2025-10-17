import requests

url = "http://127.0.0.1:5000/predict"


data = {
   "House_size":850,
   "Bedrooms":2,
   "Bathrooms":1,
   "Location":2,
   "Furnishing":0


}

response = requests.post(url,json=data)

print("status code:",response.status_code)

print("predictions:",response.text)