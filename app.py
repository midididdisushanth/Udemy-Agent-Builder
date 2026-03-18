from flask import Flask, render_template, request, jsonify
import os
import cohere
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

app = Flask(__name__)

# -------- Cohere Client --------
# Retrieve the key safely from the environment
api_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(api_key)

# -------- System Prompt --------
SYSTEM_PROMPT = """You are StudyBot, a helpful AI assistant designed specifically for students.
You help students with the following:

1. Course Recommendations - Suggest the best courses on Udemy, Coursera, or other platforms based on the student's interest or career goal.
2. Doubt Clarification - Help students understand concepts in programming, math, science, and other subjects clearly and simply.
3. Career Guidance - Guide students on career paths, skills to learn, and how to prepare for jobs in their field.
4. Study Tips - Provide effective study strategies, time management tips, and learning resources.

Always respond in a friendly, encouraging, and clear manner.
Keep responses concise and easy to understand for students.
If recommending courses, always mention the platform name (Udemy/Coursera etc), course topic, and why it is useful.
"""

# -------- Routes --------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json.get('message', '')

    if not user_message:
        return jsonify({'response': 'Please type a message!'})

    try:
        response = co.chat(
            model="command-r-08-2024",
            message=user_message,
            preamble=SYSTEM_PROMPT
        )
        bot_reply = response.text
    except Exception as e:
        bot_reply = f"Sorry, I encountered an error: {str(e)}"

    return jsonify({'response': bot_reply})

# -------- Run --------
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)