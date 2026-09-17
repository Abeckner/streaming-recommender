import os
from flask import Flask, request
from dotenv import load_dotenv
from google import genai
import markdown

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <h1>Streaming Recommender</h1>
    <form action="/recommend" method="post">
        <p>Paste your watch history:</p>
        <textarea name="history" rows="10" cols="50"></textarea>
        <p>Mood (optional):</p>
        <input type="text" name="mood">
        <p>Obscurity (mainstream / mixed / obscure):</p>
        <input type="text" name="obscurity">
        <br><br>
        <button type="submit">Get Recommendations</button>
    </form>
"""

@app.route("/recommend", methods=["post"])
def recommend():
    history = request.form["history"]
    mood = request.form["mood"]
    obscurity = request.form["obscurity"]

    prompt = f"""You are a thoughtful film and TV recommender.
    
A user has given you their watch history:
{history}

Their current mood: {mood}
Their obscurity preference: {obscurity}

First, infer the throughline of their taste - what actually connects 
what they watch, beyond genre. Then recommend 5 titles they haven't 
listed. For each, give the title, year, and 2-3 sentences on why it
fits THEM specifically, tied to what you inferred about their taste.
"""

    response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
    )

    recommendations_html = markdown.markdown(response.text)

    return f"""
        <h1> Your Recommendations</h1>
        <div style="max-width: 600px;">{recommendations_html}</div>
        <br>
        <a href="/"Go back</a>
    """

    # return f"""
    #     <h1>You sent:</h1>
    #     <p><strong>History:</strong> {history}</p>
    #     <p><strong>Mood:</strong> {mood}</p>
    #     <p><strong>Obscurity:</strong> {obscurity}</p>
    #     <a href="/">Go back</a>
    # """

if __name__ == "__main__":
    app.run(debug=True)