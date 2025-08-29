import os, json, requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
WEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
MODEL = "gpt-3.5-turbo"

SYSTEM_PROMPT = """You have three skills and must pick only one per request:

1) Smart Fashion Advisor
- If the user asks weather, call get_weather(city).
- Then answer: state city weather (°C + description) AND give 1–2 sentence outfit advice.

2) Universal Converter (units)
- If the user asks a unit conversion (mi↔km, kg↔lb, C↔F), call convert_units(value, from_unit, to_unit).
- Then answer: “{value} {from_unit} is equal to {converted} {to_unit}.” with 2 decimals.
- If unsupported: “Sorry, I don’t support that conversion yet.”

3) Currency Converter
- If the user asks a currency conversion, call convert_currency(amount, from_currency, to_currency).
- Then answer: “{amount} {from_currency} is equal to {converted} {to_currency}.” with 2 decimals.
- If unsupported: “Sorry, I don’t support that conversion yet.”
"""

# ---- Weather ----
def get_weather(city: str) -> str:
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        d = r.json()
        return json.dumps({
            "city": city,
            "temperature_c": d["main"]["temp"],
            "description": d["weather"][0]["description"]
        })
    return json.dumps({"error": "Could not get weather"})

# ---- Simple unit conversions (mi↔km, kg↔lb, C↔F) ----
ALIASES = {
    "mi": "mile", "miles": "mile", "mile": "mile",
    "km": "kilometer", "kilometers": "kilometer", "kilometer": "kilometer",
    "kg": "kilogram", "kilograms": "kilogram", "kilogram": "kilogram",
    "lb": "pound", "lbs": "pound", "pounds": "pound", "pound": "pound",
    "c": "celsius", "°c": "celsius", "celsius": "celsius",
    "f": "fahrenheit", "°f": "fahrenheit", "fahrenheit": "fahrenheit",
}
FACTORS = {("mile", "kilometer"): 1.60934, ("kilogram", "pound"): 2.20462}

def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    v = float(value)
    f = ALIASES.get((from_unit or "").strip().lower(), "")
    t = ALIASES.get((to_unit or "").strip().lower(), "")
    if not f or not t:
        return json.dumps({"error": "unsupported"})
    if f == t:
        return json.dumps({"value": v, "from_unit": f, "to_unit": t, "converted_value": v})
    if {f, t} == {"celsius", "fahrenheit"}:
        out = (v * 9/5 + 32) if f == "celsius" else ((v - 32) * 5/9)
        return json.dumps({"value": v, "from_unit": f, "to_unit": t, "converted_value": out})
    if (f, t) in FACTORS:
        out = v * FACTORS[(f, t)]
    elif (t, f) in FACTORS:
        out = v / FACTORS[(t, f)]
    else:
        return json.dumps({"error": "unsupported"})
    return json.dumps({"value": v, "from_unit": f, "to_unit": t, "converted_value": out})

# ---- Currency conversion via Frankfurter (no API key) ----
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    amt = float(amount)
    frm = (from_currency or "").strip().upper()
    to  = (to_currency or "").strip().upper()
    if not frm or not to:
        return json.dumps({"error": "unsupported_currency"})
    if frm == to:
        return json.dumps({"amount": amt, "from": frm, "to": to, "rate": 1.0, "converted": amt})

    url = f"https://api.frankfurter.dev/v1/latest?base={frm}&symbols={to}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return json.dumps({"error": "api_error"})
        data = r.json()
        rate = (data.get("rates") or {}).get(to)
        if rate is None:
            return json.dumps({"error": "unsupported_currency"})
        converted = amt * float(rate)
        return json.dumps({
            "amount": amt, "from": frm, "to": to,
            "rate": rate, "converted": converted,
            "date": data.get("date"), "base": data.get("base")
        })
    except Exception:
        return json.dumps({"error": "api_error"})

# ---- Function registry ----
FUNCTIONS = [
    {
        "name": "get_weather",
        "description": "Get current weather in a city (metric).",
        "parameters": {"type": "object","properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
    {
        "name": "convert_units",
        "description": "Convert value between mi↔km, kg↔lb, C↔F.",
        "parameters": {"type": "object","properties": {
            "value": {"type": "number"},
            "from_unit": {"type": "string"},
            "to_unit": {"type": "string"}}, "required": ["value","from_unit","to_unit"]},
    },
    {
        "name": "convert_currency",
        "description": "Convert amount between currencies using latest rates from Frankfurter.",
        "parameters": {"type": "object","properties": {
            "amount": {"type": "number"},
            "from_currency": {"type": "string"},
            "to_currency": {"type": "string"}}, "required": ["amount","from_currency","to_currency"]},
    },
]

def run_conversation(user_input: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, functions=FUNCTIONS, function_call="auto"
    )
    msg = resp.choices[0].message

    if getattr(msg, "function_call", None):
        name = msg.function_call.name
        args = json.loads(msg.function_call.arguments or "{}")

        if name == "get_weather":
            tool_result = get_weather(city=args.get("city", ""))
        elif name == "convert_units":
            tool_result = convert_units(
                value=args.get("value", 0),
                from_unit=args.get("from_unit", ""),
                to_unit=args.get("to_unit", "")
            )
        elif name == "convert_currency":
            tool_result = convert_currency(
                amount=args.get("amount", 0),
                from_currency=args.get("from_currency", ""),
                to_currency=args.get("to_currency", "")
            )
        else:
            return "Unknown tool."

        messages.append({"role": "function", "name": name, "content": tool_result})
        final = client.chat.completions.create(model=MODEL, messages=messages)
        return final.choices[0].message.content or ""

    return msg.content or ""

if __name__ == "__main__":
    while True:
        q = input("You: ")
        if q.lower().strip() == "exit":
            break
        print("AI:", run_conversation(q))
