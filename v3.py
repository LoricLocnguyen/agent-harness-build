import os
import sys
import asyncio
import json
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI


# Hiển thị được tiếng Việt và emoji trên Windows console.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# -----------------------------------------------------------------------------
# Gemini OpenAI-compatible client
# -----------------------------------------------------------------------------

api_key = os.environ.get("GEMINI_API_KEY", "").strip()
if not api_key:
    raise RuntimeError(
        "Chưa thiết lập biến môi trường GEMINI_API_KEY. "
        "Hãy đặt API key trước khi chạy chương trình."
    )


client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=api_key,
)

MODEL = "gemini-2.5-flash"


# -----------------------------------------------------------------------------
# Local workspace tools
# -----------------------------------------------------------------------------

WORKSPACE = Path(__file__).resolve().parent / "workspace"
WORKSPACE.mkdir(parents=True, exist_ok=True)


def list_files() -> str:
    """Liệt kê các tệp trong thư mục workspace."""
    files = sorted(path.name for path in WORKSPACE.iterdir() if path.is_file())
    return "\n".join(files) if files else "(empty)"


def read_file(filename: str) -> str:
    """Đọc một tệp trong workspace, không cho phép truy cập ra ngoài thư mục."""
    requested_path = (WORKSPACE / filename).resolve()
    workspace_path = WORKSPACE.resolve()

    if requested_path != workspace_path and workspace_path not in requested_path.parents:
        return "error: invalid filename"

    if not requested_path.is_file():
        return f"error: no {filename}"

    try:
        return requested_path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"error reading {filename}: {exc}"


LOCAL_TOOLS = {
    "list_files": list_files,
    "read_file": read_file,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List the files in the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read one file from the user's workspace folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "The name of the file to read.",
                    }
                },
                "required": ["filename"],
                "additionalProperties": False,
            },
        },
    },
]


# -----------------------------------------------------------------------------
# MCP servers
# -----------------------------------------------------------------------------

# Ánh xạ tên tool sang MCP session sở hữu tool đó.
mcp_sessions: dict[str, ClientSession] = {}


async def connect_mcp(stack: AsyncExitStack) -> None:
    """Khởi động MCP servers và thêm các tool của chúng vào danh sách tool."""
    mcp_servers = {
        "time": StdioServerParameters(command="uvx", args=["mcp-server-time"]),
        "fetch": StdioServerParameters(command="uvx", args=["mcp-server-fetch"]),
    }

    for server_name, params in mcp_servers.items():
        try:
            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(params)
            )
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()

            tools = (await session.list_tools()).tools
            for tool in tools:
                if tool.name in LOCAL_TOOLS:
                    print(
                        f"  [mcp] bỏ qua tool trùng tên '{tool.name}' "
                        "với local tool"
                    )
                    continue

                # Các phiên bản MCP SDK dùng tên khác nhau cho trường này:
                # input_schema (một số bản mới) hoặc inputSchema (bản cũ).
                input_schema = getattr(tool, "input_schema", None)
                if input_schema is None:
                    input_schema = getattr(tool, "inputSchema", None)
                if input_schema is None:
                    print(f"  [mcp] bỏ qua tool '{tool.name}': thiếu input schema")
                    continue

                mcp_sessions[tool.name] = session
                TOOL_SCHEMAS.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description or "MCP tool",
                            "parameters": input_schema,
                        },
                    }
                )

            print(
                f"  [mcp] connected '{server_name}': "
                f"{[tool.name for tool in tools]}"
            )
        except Exception as exc:
            # Một MCP server lỗi không làm hỏng toàn bộ trợ lý.
            print(f"  [mcp] không thể kết nối '{server_name}': {exc}")


async def call_tool(name: str, args: dict[str, Any]) -> str:
    """Chạy local tool hoặc gọi tool từ MCP server tương ứng."""
    if name in LOCAL_TOOLS:
        try:
            result = LOCAL_TOOLS[name](**args)
            return str(result)
        except TypeError as exc:
            return f"error: invalid arguments for {name}: {exc}"
        except Exception as exc:
            return f"error running {name}: {exc}"

    session = mcp_sessions.get(name)
    if session is None:
        return f"error: unknown tool {name}"

    try:
        result = await session.call_tool(name, args)
        text_parts = [
            content.text
            for content in result.content
            if hasattr(content, "text") and content.text is not None
        ]
        return "\n".join(text_parts) or "(tool returned no text)"
    except Exception as exc:
        return f"error calling MCP tool {name}: {exc}"


# -----------------------------------------------------------------------------
# Agent loop
# -----------------------------------------------------------------------------


async def run_agent(user_message: str) -> str:
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a helpful personal assistant.",
        },
        {"role": "user", "content": user_message},
    ]

    while True:
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
        )

        message = response.choices[0].message

        if not message.tool_calls:
            return message.content or ""

        # Gemini/OpenAI-compatible API cần message ở dạng dictionary.
        messages.append(message.model_dump(exclude_none=True))

        for tool_call in message.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                result = f"error: invalid JSON arguments: {exc}"
                args = {}
            else:
                if not isinstance(args, dict):
                    result = "error: tool arguments must be a JSON object"
                else:
                    print(f"  [tool] {tool_call.function.name}({args})")
                    result = await call_tool(tool_call.function.name, args)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )


# -----------------------------------------------------------------------------
# Application entry point
# -----------------------------------------------------------------------------


async def main() -> None:
    async with AsyncExitStack() as stack:
        await connect_mcp(stack)
        print(
            f"\nv3 assistant ({MODEL}) with "
            f"{len(TOOL_SCHEMAS)} tools - ctrl+c to quit"
        )

        try:
            while True:
                question = await asyncio.to_thread(input, "\nyou: ")
                question = question.strip()

                if not question:
                    continue

                try:
                    answer = await run_agent(question)
                    print("\nassistant:", answer)
                except Exception as exc:
                    print(f"\nassistant error: {exc}")
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")


if __name__ == "__main__":
    asyncio.run(main())
