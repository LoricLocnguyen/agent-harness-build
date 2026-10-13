import os 
import sys 

sys.stdout.reconfigure(encoding="utf-8")

import json
from pathlib import Path 
from openai import OpenAI

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.environ["GEMINI_API_KEY"].strip(),
)
MODEL = "gemini-2.5-flash"

WORKSPACE = Path(__file__).parent/"workspace"
WORKSPACE.mkdir(exist_ok=True)


def list_files() -> str:
    return "\n".join(p.name for p in WORKSPACE.iterdir()) or "(empty)"



def read_file(filename: str) ->str:
    path = WORKSPACE / filename
    if not path.is_file():
        return f"error: no file named {filename}"
    return path.read_text(encoding="utf-8")

TOOLS = {"list_files": list_files, "read_file": read_file}

TOOL_SCHEMAS=[
    {
        "type":"function",
        "function":{
            "name":"list_files",
            "description":"List the files in the user's workspace folder.",
            "parameters":{"type":"object","properties":{}},
        },
    },
    {
        "type":"function",
        "function":{
            "name":"read_file",
            "description":"Read one file from the user's workspaces folder.",
            "parameters":{
                "type":"object",
                "properties":{"filename":{"type":"string"}},
                "required":["filename"],
            },
        },
    },
]

def chat(user_message:str) -> str:
    messages=[
        {"role":"system","content":"You are a helpful personal assistant"},
        {"role":"user","content":user_message},
    ]

    response = client.chat.completions.create( # type: ignore
        model = MODEL , messages = messages, tools = TOOL_SCHEMAS 
    )

    message = response.choices[0].message

    if not message.tool_calls: # tool calls có sẵn trong chatbot al , là  hộp đứng để chatbot sử dụng , nếu kiểm tra trong tools_calls ko  có yêu cầu chatbot sử dụng thì empty
        return message.content or ""

    messages.append(message)
    for call in message.tool_calls:
        args = json.loads(call.function.arguments or "{}")
        print(f"  [tool] {call.function.name}({args})")
        result = TOOLS[call.function.name](**args)
        messages.append(
            {"role": "tool", "tool_call_id": call.id, "content": result}
        )

    final = client.chat.completions.create(model=MODEL, messages=messages)
    return final.choices[0].message.content


if __name__ == "__main__":
    print(f"v1 assistant ({MODEL}) - ctrl+c to quit")
    try:
        while True:
            question = input("\nyou: ")
            print("\nassistant:", chat(question))
    except (EOFError, KeyboardInterrupt):
        print("\nbye!")
