# 1.01
#Code Agent Exsample: Retrieve Weather Information
def get_weather(city):
    import requests
    api_url = f"https://api.weather.com/v1/location/{city}?apiKey=YOUR_API_KEY"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        return data.get("weather", "No weather information available")
    else:
        return "Error: Unable to fetch weather data."

# Execute the function and prepare the final answer
result = get_weather("New York")
final_answer = f"The current weather in New York is: {result}"
print(final_answer)
#-------------------------------------------------------------------------------------------------------------------------------------
#[in this exsample, the Code Agent]:
#•Retrives weather data via an API call
#•Processes the response
#•And uses the print() function to output a final answer.
#📌This method also follows the stop and parse approach by clearly delimiting the code block and signaling when execution is complete 
#(here, by printing the final_answer).

