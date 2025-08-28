# How to Add New Tools

This document explains how to add new tools to the ReAct agent.

## 1. Define the Tool Function

First, you need to define a Python function in `main.py` that will perform the action of your tool. The function should take the necessary parameters and return a string.

For example, let's create a tool that gets the current time.

```python
import datetime

def get_current_time(*args, **kwargs) -> str:
    """
    Get the current time.
    :return: The current time.
    """
    return datetime.datetime.now().isoformat()
```

## 2. Add the Tool Definition to the `functions` List

Next, you need to add the definition of your tool to the `functions` list in the `run_conversation` function. This is how the LLM knows about the tool and its parameters.

```python
functions = [
    # ... existing get_weather function
    {
        "name": "get_current_time",
        "description": "Get the current time.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    }
]
```

## 3. Add the Tool to the `available_functions` Dictionary

Finally, you need to add your new function to the `available_functions` dictionary in the `run_conversation` function. This maps the function name to the actual Python function.

```python
available_functions = {
    "get_weather": get_weather,
    "get_current_time": get_current_time,
}
```

That's it! You have now added a new tool to the ReAct agent.
