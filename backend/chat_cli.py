import os
import sys
import uuid
import json
import requests

CHAT_API_URL = os.environ.get("CHAT_API_URL", "http://localhost:8000")

def main():
    session_id = str(uuid.uuid4())
    print("Welcome to Gyani RAG CLI!")
    print("Type /exit or /quit to end the chat.")
    print("Type /stream to toggle streaming mode.")
    
    streaming = False
    
    while True:
        try:
            question = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
            
        if not question.strip():
            continue
            
        if question.strip().lower() in ["/exit", "/quit"]:
            break
            
        if question.strip().lower() == "/stream":
            streaming = not streaming
            print(f"Streaming mode: {'ON' if streaming else 'OFF'}")
            continue
            
        payload = {
            "question": question,
            "session_id": session_id
        }
        
        try:
            if streaming:
                response = requests.post(
                    f"{CHAT_API_URL}/chat/stream",
                    json=payload,
                    stream=True
                )
                response.raise_for_status()
                
                print("Assistant: ", end="", flush=True)
                sources = []
                for line in response.iter_lines():
                    if not line:
                        continue
                    line_str = line.decode('utf-8')
                    if line_str.startswith("event: "):
                        # Event type line
                        pass
                    elif line_str.startswith("data: "):
                        data_str = line_str[6:]
                        if not data_str:
                            continue
                        try:
                            data_json = json.loads(data_str)
                            if "token" in data_json:
                                print(data_json["token"], end="", flush=True)
                            elif isinstance(data_json, list):
                                sources = data_json
                        except json.JSONDecodeError:
                            pass
                print()
                if sources:
                    print("\nSources:")
                    for s in sources:
                        print(f" - {s.get('source_path')} (similarity: {s.get('similarity')})")
            else:
                response = requests.post(f"{CHAT_API_URL}/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                print("Assistant:", data.get("answer"))
                sources = data.get("sources", [])
                if sources:
                    print("\nSources:")
                    for s in sources:
                        print(f" - {s.get('source_path')} (similarity: {s.get('similarity')})")
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to {CHAT_API_URL}. Is the server running on localhost:8000?")
        except requests.exceptions.HTTPError as e:
            print(f"Server returned an error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
