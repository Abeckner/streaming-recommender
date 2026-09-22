# Streaming Recommender
#
# A small Flask web app that recommends films and TV shows based on a user's
# watch history. Users paste their history or upload a Netflix viewing-activity
# CSV export, pick a mood and an obscurity preference, and optionally add
# special instructions. The app sends this to Claude, which infers the
# throughline of their taste and returns 5 tailored recommendations.
#
# Users can then refine the results with follow-up requests. Watch history and
# previous recommendations are stored per session in PostgreSQL so
# refinements build on earlier turns without repeating titles or suggesting
# anything already watched.
import os
from flask import Flask, request, session
from dotenv import load_dotenv
from google import genai
import markdown
import csv
import io
import anthropic
import psycopg
import uuid

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

app = Flask(__name__)

app.secret_key = os.environ["FLASK_SECRET_KEY"]
DATABASE_URL = os.environ["DATABASE_URL"]
def init_db():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    session_id TEXT PRIMARY KEY,
                    history TEXT,
                    previous_recs TEXT
                )
            """)

init_db() 

def get_session_id(): 
    if "session_id" not in session:
        session["session_id"] = str(uuid.uuid4()) 
    return session["session_id"]

def save_conversation(session_id, history, previous_recs):
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversations (session_id, history, previous_recs)
                VALUES (%s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    history = excluded.history,
                    previous_recs = excluded.previous_recs
            """, (session_id, history, previous_recs))

def load_conversation(session_id):
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT history, previous_recs FROM conversations WHERE session_id = %s",
                (session_id,)
            )
            row = cur.fetchone()
    if row:
        return {"history": row[0], "previous_recs": row[1]}
    else:
        return {"history": "", "previous_recs": ""}

KOFI_HTML = """
    <div style="margin-top: 30px; text-align: center;">
        <a href="https://ko-fi.com/aaronbeckner" target="_blank"
           style="display: inline-block; padding: 10px 20px; background: #6f4e37;
                  color: white; text-decoration: none; border-radius: 8px;
                  font-family: system-ui, sans-serif; font-weight: 500;">
            ☕ Enjoying this? Buy me a coffee
        </a>
    </div>
"""

def page(content):
    return f"""
        <html>
        <head><meta name="viewport" content="width=device-width, initial-scale=1"></head>
        <body style="font-family: system-ui, sans-serif; line-height: 1.6; margin: 0;">
            <div style="max-width: 700px; margin: 0 auto; padding: 24px;">
                {content}
            </div>
        </body>
        </html>
    """

@app.route("/")
def home():
    return page(f"""
        <h1>Streaming Recommender</h1>
        <form action="/recommend" method="post" enctype="multipart/form-data">
            <p>Option 1 — Upload a watch history file (e.g. Netflix CSV):</p>
            <p style="font-size: 0.9em;">Need your Netflix history? <a href="https://www.netflix.com/viewingactivity" target="_blank">Get it here</a>, scroll down, click "Download all," then upload the file below.</p>
            <input type="file" name="history_file">
            <p>Option 2 — Or type/paste your watch history:</p>
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
        {KOFI_HTML}
    """)

@app.route("/recommend", methods=["post"])
def recommend():
    session_id = get_session_id()
    saved = load_conversation(session_id)

    refinement = request.form.get("refinement", "").strip()

    if refinement:
        history = saved["history"]
        previous_recs = saved["previous_recs"]
        mood = session.get("mood", "")
        obscurity = session.get("obscurity", "")

        prompt = f"""You are a thoughtful film and TV recommender.

The user's taste (from their watch history):
{history}

Their mood: {mood}

Their obscurity preference: {obscurity}

You have ALREADY recommended these titles - do NOT repeat any of them:
{previous_recs}

The user has ALREADY WATCHED everything in their history above - do NOT recommend anything already on that list.

The user now says: "{refinement}"

Give 5 NEW titles they have not watched and you have not already recommended, that respond to their request and fit their taste.
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
        # Save small values in the session; big history goes to the DB below
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
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        recs_text = ""
        for block in message.content:
            if block.type == "text":
                recs_text += block.text

        recommendations_html = markdown.markdown(recs_text)

        previous = saved["previous_recs"]
        new_previous_recs = previous + "\n" + recs_text
        save_conversation(session_id, history, new_previous_recs)
    except Exception as e:
        return page(f"<p>Something went wrong. Try again in a moment.</p><p><em>{e}</em></p><a href='/'>Go back</a>")

    return page(f"""
        <h1>Your Recommendations</h1>
        <div>{recommendations_html}</div>
        <hr>
        <form action="/recommend" method="post">
            <p>Seen these already? Want something different? Tell me how to adjust:</p>
            <textarea name="refinement" rows="3" cols="50"></textarea>
            <br><br>
            <button type="submit">Refine</button>
        </form>
        <br>
        <a href="/">Start over</a>
        {KOFI_HTML}
    """)

if __name__ == "__main__":
    app.run(debug=True)