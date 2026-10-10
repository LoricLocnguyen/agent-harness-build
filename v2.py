import os
import sys

sys.stdout.reconfigure(encoding="utf-8")  # so emoji don't crash the Windows console

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
    """List the files in the assistant's workspace folder."""
    return "\n".join(p.name for p in WORKSPACE.iterdir()) or "(empty)"


def read_file(filename: str) -> str:
    """Read a file from the workspace folder."""
    path = WORKSPACE / filename
    if not path.is_file():
        return f"error: no file named {filename}"
    return path.read_text(encoding="utf-8")


def write_file(filename: str, content: str) -> str:
    """Write (or overwrite) a file in the workspace folder."""
    (WORKSPACE / filename).write_text(content, encoding="utf-8")
    return f"wrote {filename}"


TOOLS = {"list_files": list_files, "read_file": read_file, "write_file": write_file}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files in the user's workspace folder.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one file from the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file in the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filename", "content"],
            },
        },
    },
]


def run_agent(user_message: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful personal assistant."},
        {"role": "user", "content": user_message},
    ]

    # THE agent loop. This is the whole trick.
    while True:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SCHEMAS
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content or ""  # done - it answered

        messages.append(message)
        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            print(f"  [tool] {call.function.name}({args})")
            result = TOOLS[call.function.name](**args)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )


if __name__ == "__main__":
    print(f"v2 agent ({MODEL}) - ctrl+c to quit")
    try:
        while True:
            question = input("\nyou: ")
            print("\nassistant:", run_agent(question))
    except (EOFError, KeyboardInterrupt):
        print("\nbye!")
