import os
import subprocess
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_community.tools.shell import ShellTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition

# --- Load Environment ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("❌ GOOGLE_API_KEY not found. Check your .env file.")
    
# --- Configure LLM ---
# genai.configure(api_key=GOOGLE_API_KEY)
llm = init_chat_model("google_genai:gemini-2.0-flash")

# --- Tool Definitions ---
@tool
def run_command(command: str):
    """Run a shell command on the terminal."""
    print(f"[🔧 run_command] Executing: {command}")
    return os.system(command)

@tool
def write_code(file_name: str, code: str):
    """Write the provided code to a file."""
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w') as f:
        f.write(code)
    return f"✅ Code written to {file_name}"

@tool
def read_code(file_name: str):
    """Read and return the content of a file."""
    try:
        with open(file_name, 'r') as f:
            return f.read()
    except Exception as e:
        return f"❌ {e}"

@tool
def execute_code(file_name: str):
    """Execute a Python file and return its output."""
    try:
        result = subprocess.run(["python", file_name], capture_output=True, text=True)
        output = result.stdout.strip()
        error = result.stderr.strip()
        return output + ("\n❌ " + error if error else "")
    except Exception as e:
        return f"❌ Error: {e}"


# --- LangGraph State Definition ---
class State(TypedDict):
    messages: Annotated[list, add_messages]

llm_with_tools = llm.bind_tools([run_command, write_code, read_code, execute_code])

# --- Chatbot Logic ---
def chatbot(state: State):
    system_message = SystemMessage(
        content="""You are an AI coding assistant. Use tools instead of explaining.
        
Available tools:
- run_command
- write_code
- read_code
- execute_code

Examples:
User: Create a Python file that prints "Hi".
→ write_code("gemini/hi.py", "print('Hi')")

User: Run the file.
→ execute_code("gemini/hi.py")

Rules:
- Do not describe. Call tools directly.
- All files go inside 'gemini/'. Create it if missing.
- if a file exists, ask user to what to do if not mentioned by user.
"""
    )
    messages = [system_message] + state["messages"]
    print(f"\n[🧠 Chatbot] Invoking LLM with {len(messages)} messages...\n")
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# --- Graph Building ---
tool_node = ToolNode(tools=[run_command, write_code, read_code, execute_code])
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges("chatbot", tools_condition, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "chatbot")
graph = graph_builder.compile()

def create_chat_graph(checkpointer):
    return graph_builder.compile(checkpointer=checkpointer)
