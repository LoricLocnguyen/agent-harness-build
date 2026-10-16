"""
v4 — Agentic AI Assistant with Long-Term Memory
================================================
Author  : LoricLocnguyen
Written : October 2026
License : MIT

Tự tay code từ đầu. v4 nâng cấp từ v2 bằng cách thêm hệ thống
bộ nhớ dài hạn (memory.md) để assistant nhớ thông tin người dùng
qua các phiên làm việc. Conversation history cũng được giữ lại
trong cùng một session.

Stack: Gemini 2.5 Flash + OpenAI SDK + Function Calling
"""

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
MODEL ="gemini-2.5-flash"
WORKSPACE = Path(__file__).parent / "workspace"
MEMORY_FILE = Path(__file__).parent / "memory.md"


# --- memory: a markdown file, that's it -------------------------------------

def load_memory() -> str:
    if MEMORY_FILE.is_file():
        return MEMORY_FILE.read_text(encoding="utf-8")
    return "(nothing saved yet)"


def save_memory(fact: str) -> str:
    """Append one fact about the user to long-term memory."""
    with MEMORY_FILE.open("a", encoding="utf-8") as f:
        f.write(f"- {fact}\n")
    return f"saved: {fact}"


# --- tools ------------------------------------------------------------------

def list_files() -> str:
    return "\n".join(p.name for p in WORKSPACE.iterdir()) or "(empty)"


def read_file(filename: str) -> str:
    path = WORKSPACE / filename
    return path.read_text(encoding="utf-8") if path.is_file() else f"error: no {filename}"


TOOLS = {"list_files": list_files, "read_file": read_file, "save_memory": save_memory}

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
            "name": "save_memory",
            "description": (
                "Save one short fact about the user to long-term memory. "
                "Use whenever you learn something worth remembering: their "
                "name, preferences, projects, recurring tasks."
            ),
            "parameters": {
                "type": "object",
                "properties": {"fact": {"type": "string"}},
                "required": ["fact"],
            },
        },
    },
]


SYSTEM_PROMPT = """You are a helpful personal assistant.

Here is what you remember about the user from previous sessions:
{memory}

When you learn a new lasting fact about the user, save it with save_memory."""


def run_agent(messages: list) -> str:
    while True:
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOL_SCHEMAS
        )
        message = response.choices[0].message
        if not message.tool_calls:
            messages.append({"role": "assistant", "content": message.content})
            return message.content or ""

        messages.append(message)
        for call in message.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            print(f"  [tool] {call.function.name}({args})")
            result = TOOLS[call.function.name](**args)
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )


if __name__ == "__main__":
    # Conversation history persists across turns now, too.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(memory=load_memory())}
    ]
    print(f"v4 assistant ({MODEL}) - memory loaded - ctrl+c to quit")
    try:
        while True:
            messages.append({"role": "user", "content": input("\nyou: ")})
            print("\nassistant:", run_agent(messages))
    except (EOFError, KeyboardInterrupt):
        print("\nbye!")
