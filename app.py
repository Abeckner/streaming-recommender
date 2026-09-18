import os
from flask import Flask, request, session
from dotenv import load_dotenv
from google import genai
import markdown
import csv
import io
import anthropic

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

app = Flask(__name__)

app.secret_key = os.environ["FLASK_SECRET_KEY"]

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
    refinement = request.form.get("refinement", "").strip() 

    if refinement:
        # This is a follow-up turn - pull the saved context from the session
        history = session.get("history", "")
        mood = session.get("mood", "")
        obscurity = session.get("obscurity", "")
        previous_recs = session.get("previous_recs", "")

        prompt = f"""You are a thoughtful film and TV recommender.

The user's taste (from their watch history):
{history}

Their mood: {mood}

Their obscurity preference: {obscurity}

You have ALREADY recommended these titles - do NOT repeat any of them:
{previous_recs}

The user now says: "{refinement}"

Give 5 NEW titles that respond to theior request and fit their taste.
For each: title, year, and 2-3 sentences on why it fits THEM specifically.
"""
    else:
        # This is a first submission - build history from file or paste
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
        # Save context for future refinement turns
        session["history"] = history
        session["mood"] = mood
        session["obscurity"] = obscurity

    try:
        # --- Gemini fallback (free tier) — uncomment to switch back ---
        # response = client.models.generate_content(
        #     model="gemini-3.6-flash",
        #     contents=prompt,
        # )
        # recs_text = response.text

        # --- Claude Sonnet (current) ---
        message = claude.messages.create(
            model="claude-sonnet-5",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        recs_text = ""
        for block in message.content:
            if block.type == "text":
                recs_text += block.text

        recommendations_html = markdown.markdown(recs_text)

        previous = session.get("previous_recs", "")
        session["previous_recs"] = previous + "\n" + recs_text
    except Exception as e:
        return f"<p>Something went wrong. Try again in a moment.</p><p><em>{e}</em></p><a href='/'>Go back</a>"

    return f"""
        <h1> Your Recommendations</h1>
        <div style="max-width: 600px;">{recommendations_html}</div>
        <hr style="max-width: 600px;">
        <form action="/recommend" method="post">
            <p>Seen these already? Want something different? Tell me how to adjust:</p>
            <textarea name="refinement" rows="3" cols="50"></textarea>
            <br><br>
            <button type="submit">Refine</button>
        </form>
        <br>
        <a href="/">Start over</a>
    """

if __name__ == "__main__":
    app.run(debug=True)