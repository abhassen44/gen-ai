import os
import json
import requests
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize model
model = genai.GenerativeModel(model_name='gemini-1.5-flash-latest')

# Tool functions
def run_command(command):
    result = os.system(command=command)
    return result

def get_weather(city: str):
    print("🔨 Tool Called: get_weather", city)
    url = f"https://wttr.in/{city}?format=%C+%t"
    response = requests.get(url)
    if response.status_code == 200:
        return f"The weather in {city} is {response.text.strip()}."
    return "Something went wrong"

# Available tools
avaiable_tools = {
    "get_weather": {
        "fn": get_weather,
        "description": "Takes a city name as an input and returns the current weather for the city"
    },
    "run_command": {
        "fn": run_command,
        "description": "Takes a command as input to execute on system and returns output"
    }
}

# System prompt
system_prompt = """
You are a helpful AI Assistant who resolves user queries using tools. You work in four steps: plan, action, observe, output.

Rules:
- Output must be in JSON format.
- Perform one step at a time and wait for the next user input.
- Carefully analyze the user query and available tools.

JSON Output Format:
{
    "step": "string",
    "content": "string",
    "function": "The name of function if the step is action",
    "input": "The input parameter for the function"
}

Tools:
- get_weather: Takes a city name and returns weather.
- run_command: Takes a shell command and returns output.

Example for "What's the weather in Paris?":
Output: { "step": "plan", "content": "User is asking for weather in Paris" }
Output: { "step": "plan", "content": "Use get_weather tool" }
Output: { "step": "action", "function": "get_weather", "input": "Paris" }
Output: { "step": "observe", "output": "Sunny +22°C" }
Output: { "step": "output", "content": "The weather in Paris is Sunny with 22°C." }
"""

# Start conversation
messages = [system_prompt]

while True:
    user_query = input("> ")
    messages.append(f"User Query: {user_query}")

    while True:
        prompt = "\n".join(messages) + "\nRespond with the next JSON step:"
        response = model.generate_content(prompt)

        raw_text = response.text.strip()

        # Strip code block wrappers if present
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:-3].strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:-3].strip()

        try:
            parsed_output = json.loads(raw_text)
        except Exception as e:
            print("⚠️ JSON Parse Error:", e)
            print("🔴 Raw Output:", response.text)
            break

        messages.append(json.dumps(parsed_output))

        step = parsed_output.get("step")

        if step == "plan":
            print(f"🧠: {parsed_output['content']}")
            continue

        if step == "action":
            tool_name = parsed_output.get("function")
            tool_input = parsed_output.get("input")
            if tool_name in avaiable_tools:
                output = avaiable_tools[tool_name]["fn"](tool_input)
                messages.append(json.dumps({ "step": "observe", "output": output }))
                continue

        if step == "output":
            print(f"🤖: {parsed_output['content']}")
            break
