import os
from flask import Flask, request
from dotenv import load_dotenv
from google import genai
import markdown
import csv
import io

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

app = Flask(__name__)

@app.route("/")
def home():
    return """
        <h1>Streaming Recommender</h1>
        <form action="/recommend" method="post" enctype="multipart/form-data">
            <p>Option 1 — Upload a watch history file (e.g. Netflix CSV):</p>
            <input type="file" name="history_file">
            <p>Option 2 — Or paste your watch history:</p>
            <textarea name="history" rows="10" cols="50"></textarea>
            <p>Mood (optional):</p>
            <input type="text" name="mood">
            <p>Obscurity (mainstream / mixed / obscure):</p>
            <input type="text" name="obscurity">
            <p>Anything to keep in mind? (e.g. "ignore kids' and teen shows", "only movies", "skip reality TV")</p>
            <textarea name="instructions" rows="3" cols="50"></textarea>
            <br><br>
            <button type="submit">Get Recommendations</button>
        </form>
    """

@app.route("/recommend", methods=["post"])
def recommend():
    uploaded_file = request.files.get("history_file")

    if uploaded_file and uploaded_file.filename:
        raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
        reader = csv.reader(io.StringIO(raw_text))
        titles = set()
        next(reader, None)  # skip the header row
        for row in reader:
            if row:  # skip empty lines
                base_title = row[0].split(":")[0].strip()
                titles.add(base_title)
        history = ", ".join(sorted(titles))
    else:
        history = request.form["history"]
    mood = request.form["mood"]
    obscurity = request.form["obscurity"]
    instructions = request.form.get("instructions", "")
    prompt = f"""You are a thoughtful film and TV recommender.
    
A user has given you their watch history:
{history}

Their current mood: {mood}
Their obscurity preference: {obscurity}
Special instructions from the user (follow these carefully): {instructions}

Note: this watch history may contain shows watched by other people sharing the account. Use the user's special instructions to filter those out, and lean toward the taste that dominates unless told otherwise.

First, infer the throughline of their taste - what actually connects 
what they watch, beyond genre. Then recommend 5 titles they haven't 
listed. For each, give the title, year, and 2-3 sentences on why it
fits THEM specifically, tied to what you inferred about their taste.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
        )
        recommendations_html = markdown.markdown(response.text)
    except Exception as e:
        recommendations_html = f"<p>Something went wrong talking to the recommender. Try again in a moment.</p><p><em>{e}</em></p>"

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