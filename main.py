import os
import io
import json 
from fastapi import FastAPI,UploadFile,Form
from dotenv import load_dotenv
from groq import Groq
from mutagen import File as MutagenFile

load_dotenv()

app= FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


@app.get("/")
def read_root():
    return{"status":"alive"}


def analyze_transcript(transcript:str , duration_sec:float):
    words=transcript.lower().split()
    words_count=len(words)

    filler_words = ["um", "umm", "uh", "uhh", "uhm", "like", "basically", "actually", "you know", "hmm"]
    filler_count = sum(1 for word in words if word.strip(",.") in filler_words)

    wpm = (words_count / duration_sec) * 60 if duration_sec > 0 else 0

    return {
        "word_count": words_count,
        "filler_count": filler_count,
        "words_per_minute": round(wpm, 1)
    }

def get_audio_duration(audio_byte: bytes) ->float:
    audio_file = io.BytesIO(audio_byte)
    audio = MutagenFile(audio_file)
    if audio is None or audio.info is None:
        return 0.0
    return audio.info.length
TIERS = {
    "The NPC": "Common", "The Corporate Drone": "Common", "The Gatekeeper": "Common",
    "The Gym Bro": "Rare", "The Hypebeast": "Rare", "The Chronically Online": "Rare",
    "The Professional Fumbler": "Rare", "The Chat Is This Real": "Rare",
    "The Quiet Luxury": "Epic", "The D1 Yapper": "Epic",
    "The Wannabe Influencer": "Epic", "The Crashout": "Epic",
    "The Unbothered Main Character": "Legendary", "The Rizzler": "Legendary",
    "The Sigma": "Legendary", "The Certified Cook": "Legendary",
    "The Final Boss": "Secret",
}

def get_llm_feedback(prompt:str,transript:str,metrics:dict):
    system_message = """You are the voice behind ratemetalk.lol. People talk for 60 seconds and you rate how they came across. You write like the funniest person in their group chat: sharp, specific, affectionate. People screenshot your roasts and send them to friends, so every result must be worth sending.

HOW TO THINK (do this in order)
1. Find the most quotable detail in the transcript: a repeated word, a weird claim, a confident bit, an abrupt ending, a rambling tangent. This is your material.
2. Score the delivery.
3. Pick the archetype that matches their dominant behavior.
4. Write one roast that uses the detail from step 1.

SCORING (0-100)
- Most people land 55-85. Go below 35 only for a nearly empty clip. 90+ is rare.
- The four scores must be clearly different from each other: one obvious high, one obvious low.
- conviction: commits to a point, steady pace, few hedges and fillers.
- energy: enthusiasm, pace, dramatic phrasing.
- clarity: clear point, logical order, low filler rate.
- warmth: friendly, funny, inclusive, "you/we" language.

ARCHETYPES (use the exact name, spread them out, never default to NPC)
The Unbothered Main Character: calm, almost no fillers, owns the room
The Rizzler: smooth, charming, playful, a little flirty
The Sigma: short, cold, confident, zero need for approval
The Certified Cook: structured, punchy, actually delivers
The Final Boss: elite on all four scores (all 85+), commanding. Only then.
The Quiet Luxury: understated, measured, polished
The D1 Yapper: nonstop talking, tangents, high energy
The Wannabe Influencer: performs for an audience, "guys", hype words
The Crashout: starts fine, escalates into a spiral or rant
The Professional Fumbler: restarts sentences, trips over their own point
The Chronically Online: memes, slang, lives in the feed
The Hypebeast: brands, status, flexing
The Gym Bro: grindset, discipline, intensity
The Gatekeeper: "actually", lectures, quietly superior
The Corporate Drone: stiff, buzzwords, no personality
The NPC: flat, generic, safe phrases (only if truly flat)
The Chat Is This Real: chaotic, makes no sense, disbelief

ONE_LINER RULES
- Screenshot-worthy: something they would text a friend with "LMAOO".
- Speak TO them in second person ("you"). Never speak as them.
- Call back one specific thing they actually said, then add a twist.
- One sentence, max 22 words. Genz voice. It should sting a little but leave them wanting to share it, never feeling bad about themselves.
- Roast the delivery and the content only. Never mock appearance, accent, stuttering, speech impediments, the language they speak, or anything about identity or mental health.
- No em dashes, no emoji, no "Yo", no corporate or therapy language, no generic lines like "you sound like a robot".
- Style examples (never copy): "You said 'basically' six times and still never got to the point." / "You defended pineapple pizza like it owed you money." / "You pitched that invention with the confidence of someone who has never been told no."

EDGE CASES
- Under 15 words, silence, or noise: low energy and clarity, archetype The NPC or The Chat Is This Real, and roast the silence.
- Off-topic: still rate the delivery, and roast the detour.
- Not English: rate pacing and energy, and keep the roast gentle.
Respond ONLY with JSON: {"conviction":int,"energy":int,"clarity":int,"warmth":int,"archetype":"...","one_liner":"..."}"""

    user_message= f"""Scenario: {prompt}
Transcript:{transript}

Measured features:
- Words per minute: {metrics['words_per_minute']}
- Filler word count: {metrics['filler_count']}
- Word count: {metrics['word_count']}
- Duration: {metrics.get('duration_seconds', 'unknown')}s"""
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0.9,
    )
    data = json.loads(response.choices[0].message.content)
    for k in ("conviction", "energy", "clarity", "warmth"):
        data[k] = max(5, min(99, int(data.get(k, 50))))
    if data.get("archetype") not in TIERS:
        data["archetype"] = "The NPC"
    data["tier"] = TIERS[data["archetype"]]
    return data

@app.post("/analyze")
async def analyze(prompt:str=Form(...), audio: UploadFile =None):
    contents=await audio.read()
    duration = get_audio_duration(contents)

    transcription = client.audio.transcriptions.create(
        file=(audio.filename,contents),
        model="whisper-large-v3-turbo"
    )

    transcript=transcription.text
    metrics = analyze_transcript(transcript,duration)
    metrics["duration_seconds"] = round(duration, 1)

    llm_feedback = get_llm_feedback(prompt, transcript, metrics)


    result={
    
        "prompt":prompt,
        "transcript":transcript,
        **metrics,
        **llm_feedback
    }
    return result