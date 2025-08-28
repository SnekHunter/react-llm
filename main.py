
import os
import json
from dotenv import load_dotenv
import openai
import requests

load_dotenv()

# Get the API key from the environment
openai.api_key = os.getenv("OPENAI_API_KEY")

# Weather API key
WEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
MODEL = "gpt-3.5-turbo"

def get_weather(city: str) -> str:
    """
    Get the weather for a city.
    :param city: The city to get the weather for.
    :return: The weather for the city.
    """
    # Not a real API key, just a placeholder
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return json.dumps(
            {
                "temperature": data["main"]["temp"],
                "description": data["weather"][0]["description"],
            }
        )
    else:
        return json.dumps({"error": "Could not get weather"})


def run_conversation(user_input: str):
    # Step 1: send the conversation and available functions to GPT
    messages = [
        {
            "role": "user",
            "content": user_input,
        }
    ]
    functions = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                },
                "required": ["city"],
            },
        }
    ]
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        functions=functions,
        function_call="auto",  # auto is default, but we'll be explicit
    )
    response_message = response["choices"][0]["message"]

    # Step 2: check if GPT wanted to call a function
    if response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "get_weather": get_weather,
        }
        function_name = response_message["function_call"]["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = function_to_call(
            city=function_args.get("city"),
        )

        # Step 4: send the info on the function call and function response to GPT
        messages.append(response_message)  # extend conversation with assistant's reply
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )
        second_response = openai.ChatCompletion.create(
            model=MODEL,
            messages=messages,
        )
        return second_response
    return response


if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = run_conversation(user_input)
        print(f"AI: {response.choices[0].message.content}")
