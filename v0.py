import os
import sys
sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI 

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
MODEL = "gemini-3.8-flash"

def chat(user_message : str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role" : "system", "content": "You are a helpful personal assistant."},
            {"role" : "user", "content" : user_message},
        ],
    )
    return response.choices[0].message.content or ""

if __name__ == "__main__":
    print(f"v0 assistant ({MODEL}) - ctrl+c to quit")
    try:
        while True:
            question = input("\nYou:")
            print("\nassistant:",chat(question))
    except (EOFError , KeyboardInterrupt):
        print("\nbye!")