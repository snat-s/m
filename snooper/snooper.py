import os
import time
import sqlite3
import base64
import requests
from io import BytesIO
from PIL import Image
from pynput import keyboard

db_path = os.path.expanduser("~/.config/snooper.db")
screenshot_path = "/tmp/screenshot.png"
keystrokes_path = "/tmp/keystrokes.txt"

api = "https://api.hyperbolic.xyz/v1/chat/completions"
api_key = os.environ.get("HYPERBOLIC_API")
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}

def encode_image(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def init_db():
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS activities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            screenshot BLOB,
            keystrokes TEXT,
            category TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(screenshot, keystrokes, category):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO activities (screenshot, keystrokes, category)
        VALUES (?, ?, ?)
    """, (screenshot, keystrokes, category))
    conn.commit()
    conn.close()

def get_screenshot():
    try:
        import subprocess
        import tempfile
        
        # Create a temporary file to store the screenshot
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        # Capture screenshot using gnome-screenshot
        subprocess.run(['gnome-screenshot', '-f', tmp_path], check=True)
        
        # Open and return the image
        return Image.open(tmp_path)
    except Exception as e:
        print(f"Screenshot failed: {e}")
        return None

keystrokes = []
def on_press(key):
    try:
        keystrokes.append(key.char)
    except AttributeError:
        keystrokes.append(str(key))

def analyze_screenshot(img):
    base64_img = encode_image(img)
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": """
                    Categorize the user's activity based on the text. Use one of the following categories:
                    - Reading: Reading articles, books, documentation, emails
                    - Writing: emails, documents, notes
                    - Coding: Writing code.
                    - Watching: Watching anime, films, TV shows, YouTube, tutorials
                    - Browsing: news, shopping sites, forums
                    - Social Media: Scrolling through social media.
                    - Gaming: Playing video games, watching game streams
                    - Learning: Taking online courses, studying, researching
                    - Working: Programming, debugging, attending meetings, managing tasks
                    - Listening: Listening to music, podcasts, audiobooks
                    - Designing: Graphic design, UI/UX design, video editing
                    - Chatting: Messaging, video calling
                    - Idle: Idle/Inactive, screen locked
                    - Miscellaneous: Planning, financial management, health tracking

                    Your output should only be a single word, the category you choose.
                     """},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                ],
            }
        ],
        "model": "Qwen/Qwen2-VL-72B-Instruct",
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
    }
    response = requests.post(api, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Unknown"

def main():
    init_db()

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    while True:
        time.sleep(10)
        screenshot = get_screenshot()
        screenshot.save(screenshot_path)

        with open(screenshot_path, "rb") as f:
            screenshot_blob = f.read()

        category = analyze_screenshot(screenshot)

        keystrokes_str = " ".join(keystrokes)
        keystrokes.clear()

        save_to_db(screenshot_blob, keystrokes_str, category)

        time.sleep(300)

if __name__ == "__main__":
    main()
