import requests

url = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=41.8781"
    "&longitude=-87.6298"
    "&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
)

response = requests.get(url)

data = response.json()
current = data["current"]

print("Temperature:", current["temperature_2m"])
print("Humidity:", current["relative_humidity_2m"])
print("Wind Speed:", current["wind_speed_10m"])