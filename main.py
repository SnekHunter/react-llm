import os, json, requests, re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
WEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
MODEL = "gpt-3.5-turbo"

# ---- Travel Planner instruction variants (kept minimal; select via env INSTRUCTION_VARIANT) ----
INSTRUCTION_VARIANTS = {
    "simple": """You have a tool named plan_trip(destination, duration_days, interests).

When the user asks to plan a trip, do ONLY this:
- Extract destination, duration in DAYS (interpret words: “a week”=7, “weekend”=2 or 3), and interests (default ["general"]).
- If the user asks to compare destinations for the same duration/interests, call plan_trip once per destination, then summarize both and highlight differences.
- If any of destination or duration is missing, ask one direct question for JUST the missing value.
- Do NOT call weather or any converter tools for travel requests.
- Keep the final answer concise: a short overview plus a Day 1..N plan (morning/afternoon/evening).""",

    "detailed": """Use ONLY the plan_trip(destination, duration_days, interests) tool for travel planning intents.

Detection:
- “Plan”, “itinerary”, “trip”, “things to do”, “3-day in X”, “week in X”, or “compare X vs Y” → plan_trip.
- Ignore weather or conversions even if mentioned alongside travel.

Parameters:
- destination: required (use exactly what the user wrote if ambiguous).
- duration_days: 1..30; words→days (day=1, weekend=3 unless clearly 2, week=7, fortnight=14).
- interests: from user text; default ["general"]; unknowns→"general".
- Comparative: same duration/interests unless stated otherwise.

Errors:
- If destination or duration is missing, ask ONE targeted question.
- If the tool errors, state what’s missing and ask one follow-up.

Output:
- “Itinerary: {destination}, {days} days, interests: {…}”
- Day 1..N (morning/afternoon/evening).
- For comparisons: two labeled blocks + “Key differences” bullets (pace, themes, highlights).

Examples:
- “Plan me a 3-day trip to Bali.” → plan_trip("Bali", 3, ["general"])
- “Rome for a week. Love food and history.” → plan_trip("Rome", 7, ["food","history"])
- “4 days hiking: New Zealand or Switzerland?” → call plan_trip twice with (4, ["hiking"]).""",

    "cot": """Think step-by-step SILENTLY to decide if the user intent is travel planning. Do NOT reveal your reasoning. If yes, use ONLY:

plan_trip(destination, duration_days, interests)

Private checklist (do not output):
1) Extract destination(s).
2) Normalize duration to days (weekend=3 unless asked otherwise).
3) Extract interests (default ["general"]; unknown→"general").
4) If comparative, call plan_trip twice (same days/interests).
5) If a required field is missing, ask one targeted question.

Public output:
- One-line summary (destination(s), days, interests).
- Day 1..N with morning/afternoon/evening from tool output.
- For comparisons: two mini-itineraries + “How they differ”.
- No weather or converters; keep tight and actionable."""
}

def _compose_system_prompt(variant_name: str) -> str:
    travel_variant = INSTRUCTION_VARIANTS.get(variant_name or "", INSTRUCTION_VARIANTS["cot"]).strip()
    base_prompt = """You have FOUR skills and must pick only one per request:

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

4) Master Travel Planner  ← NEW
- If the user asks to plan a trip, call plan_trip(destination, duration_days, interests).
- If no interests provided, use ["general"].
- If the user gives a duration in words (e.g., “a week”), interpret it (week=7, weekend=2/3).
- If the user asks to compare destinations for the same duration/interests, call plan_trip separately for EACH destination, then present both itineraries succinctly and highlight differences.
- Keep responses concise and actionable. If the tool returns an error, ask the user for the missing details (destination, days, interests).
- IMPORTANT: Only use ONE of the four skills per request. If user intent clearly matches Travel Planner, do NOT call weather or converters.

---- Travel Planner Instruction Variant (applies only when using skill #4) ----
"""
    return f"{base_prompt}{travel_variant}\n"

SYSTEM_PROMPT = _compose_system_prompt(os.getenv("INSTRUCTION_VARIANT", "detailed"))

# ---- Weather ----
def get_weather(city: str) -> str:
    api_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(api_url, timeout=10)
    except requests.RequestException:
        return json.dumps({"error": "network_error"})
    if response.status_code == 200:
        payload = response.json()
        weather_info = {
            "city": city,
            "temperature_c": payload.get("main", {}).get("temp"),
            "description": (payload.get("weather") or [{}])[0].get("description")
        }
        return json.dumps(weather_info)
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
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return json.dumps({"error": "invalid_value"})

    from_normalized = ALIASES.get((from_unit or "").strip().lower(), "")
    to_normalized = ALIASES.get((to_unit or "").strip().lower(), "")

    if not from_normalized or not to_normalized:
        return json.dumps({"error": "unsupported"})

    if from_normalized == to_normalized:
        return json.dumps({
            "value": value_float,
            "from_unit": from_normalized,
            "to_unit": to_normalized,
            "converted_value": value_float
        })

    if {from_normalized, to_normalized} == {"celsius", "fahrenheit"}:
        converted_value = (
            value_float * 9/5 + 32
            if from_normalized == "celsius"
            else (value_float - 32) * 5/9
        )
        return json.dumps({
            "value": value_float,
            "from_unit": from_normalized,
            "to_unit": to_normalized,
            "converted_value": converted_value
        })

    if (from_normalized, to_normalized) in FACTORS:
        converted_value = value_float * FACTORS[(from_normalized, to_normalized)]
    elif (to_normalized, from_normalized) in FACTORS:
        converted_value = value_float / FACTORS[(to_normalized, from_normalized)]
    else:
        return json.dumps({"error": "unsupported"})

    return json.dumps({
        "value": value_float,
        "from_unit": from_normalized,
        "to_unit": to_normalized,
        "converted_value": converted_value
    })

# ---- Currency conversion via Frankfurter (no API key) ----
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    try:
        amount_float = float(amount)
    except (TypeError, ValueError):
        return json.dumps({"error": "invalid_amount"})

    from_code = (from_currency or "").strip().upper()
    to_code = (to_currency or "").strip().upper()
    if not from_code or not to_code:
        return json.dumps({"error": "unsupported_currency"})
    if from_code == to_code:
        return json.dumps({
            "amount": amount_float, "from": from_code, "to": to_code,
            "rate": 1.0, "converted": amount_float
        })

    api_url = f"https://api.frankfurter.dev/v1/latest?base={from_code}&symbols={to_code}"
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code != 200:
            return json.dumps({"error": "api_error"})
        data = response.json()
        rate_value = (data.get("rates") or {}).get(to_code)
        if rate_value is None:
            return json.dumps({"error": "unsupported_currency"})
        converted_amount = amount_float * float(rate_value)
        return json.dumps({
            "amount": amount_float,
            "from": from_code,
            "to": to_code,
            "rate": rate_value,
            "converted": converted_amount,
            "date": data.get("date"),
            "base": data.get("base")
        })
    except Exception:
        return json.dumps({"error": "api_error"})

# ---- NEW: Travel Planner tool (offline generator) ----
_INTEREST_BANK = {
    "general": [
        "Orientation walk & iconic landmark",
        "Top-rated museum or cultural site",
        "Sunset viewpoint or waterfront stroll",
        "Local market + street snacks",
        "Signature district exploration"
    ],
    "art": [
        "Major art museum/gallery",
        "Street art / contemporary scene",
        "Artist studio or design district"
    ],
    "history": [
        "Old town & heritage sites",
        "Fortresses/ruins & guided history tour",
        "Archaeology or national museum"
    ],
    "food": [
        "Market/tasting tour",
        "Local cooking class",
        "Regional specialty dinner"
    ],
    "hiking": [
        "National park trail (half-day)",
        "Scenic ridge or lake loop",
        "Sunrise/sunset hike & viewpoints"
    ],
    "beach": [
        "Relax on popular beach",
        "Snorkel/boat excursion",
        "Beach sunset + seaside dinner"
    ]
}

def _sanitize_interests(interests):
    if not interests:
        return ["general"]
    normalized_interests = []
    for interest in interests:
        key = (interest or "").strip().lower()
        normalized_interests.append(key if key in _INTEREST_BANK else "general")
    return normalized_interests or ["general"]

def plan_trip(destination: str, duration_days: int, interests: list[str]) -> str:
    if not destination or not isinstance(destination, str):
        return json.dumps({"error": "missing_destination"})
    try:
        duration_int = int(duration_days)
    except Exception:
        return json.dumps({"error": "invalid_duration"})
    duration_int = max(1, min(duration_int, 30))

    normalized_interests = _sanitize_interests(interests)
    itinerary_days = []
    for day_index in range(1, duration_int + 1):
        theme = normalized_interests[(day_index - 1) % len(normalized_interests)]
        idea_list = _INTEREST_BANK.get(theme, _INTEREST_BANK["general"])
        morning_activity = idea_list[(day_index * 1) % len(idea_list)]
        afternoon_activity = idea_list[(day_index * 2) % len(idea_list)]
        evening_activity = idea_list[(day_index * 3) % len(idea_list)]
        itinerary_days.append({
            "day": day_index,
            "theme": theme,
            "morning": morning_activity,
            "afternoon": afternoon_activity,
            "evening": evening_activity
        })

    return json.dumps({
        "destination": destination,
        "duration_days": duration_int,
        "interests": normalized_interests,
        "itinerary": itinerary_days,
        "note": "Offline generator: swap with a real travel API if desired."
    })

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
    # NEW
    {
        "name": "plan_trip",
        "description": "Generate a basic travel itinerary for a destination, duration (days), and interests list.",
        "parameters": {"type": "object","properties": {
            "destination": {"type": "string", "description": "City/region/country name"},
            "duration_days": {"type": "integer", "minimum": 1, "maximum": 30},
            "interests": {"type": "array", "items": {"type": "string"}}
        }, "required": ["destination","duration_days","interests"]},
    },
]

# ---- Dispatcher that supports MULTIPLE tool calls (important for comparisons) ----
def _call_tool(name: str, args: dict) -> str:
    if name == "get_weather":
        return get_weather(city=args.get("city", ""))
    if name == "convert_units":
        return convert_units(
            value=args.get("value", 0),
            from_unit=args.get("from_unit", ""),
            to_unit=args.get("to_unit", "")
        )
    if name == "convert_currency":
        return convert_currency(
            amount=args.get("amount", 0),
            from_currency=args.get("from_currency", ""),
            to_currency=args.get("to_currency", "")
        )
    if name == "plan_trip":
        return plan_trip(
            destination=args.get("destination", ""),
            duration_days=args.get("duration_days", 0),
            interests=args.get("interests", []),
        )
    return json.dumps({"error": "unknown_tool"})

def run_conversation(user_input: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_input},
    ]

    max_tool_hops = 6  # safety cap
    for _ in range(max_tool_hops):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, functions=FUNCTIONS, function_call="auto"
        )
        message = response.choices[0].message

        if getattr(message, "function_call", None):
            tool_name = message.function_call.name
            tool_args = json.loads(message.function_call.arguments or "{}")
            tool_output = _call_tool(tool_name, tool_args)
            messages.append({"role": "function", "name": tool_name, "content": tool_output})
            # loop again to allow the model to make another tool call if it wants
            continue

        # no further tool calls; return final assistant message
        return message.content or ""

    # fallback if too many tool calls
    return "I made too many tool calls for this request. Please refine your question."

if __name__ == "__main__":
    while True:
        user_text = input("You: ")
        if user_text.lower().strip() == "exit":
            break
        print("AI:", run_conversation(user_text))
