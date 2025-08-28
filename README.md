# ReAct Weather Bot

This project is a simple demonstration of the ReAct (Reasoning + Acting) pattern with a Large Language Model (LLM). The bot can get the current weather for a given city using the OpenWeatherMap API.

## How it works

The user provides a prompt to the LLM. The LLM, in turn, has access to a `get_weather` function. If the LLM determines that it needs to call this function to answer the user's query, it will do so. The result of the function call is then fed back to the LLM, which then generates a final response to the user.

## Setup

1.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up your API keys:**

    - Rename the `.env.example` file to `.env`.
    - Open the `.env` file and add your OpenAI and OpenWeatherMap API keys.

3.  **Run the bot:**

    ```bash
    python main.py
    ```

## Example

```
You: What's the weather like in London?
LLM: The weather in London is currently 15 degrees Celsius with broken clouds.
```
