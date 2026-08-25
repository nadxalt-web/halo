import os
import io
import json 
from fastapi import FastAPI,UploadFile,Form
from dotenv import load_dotenv
from groq import Groq
from mutagen import File as MutagenFile

load_dotenv()

app= FastAPI()
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

def get_llm_feedback(prompt:str,transript:str,metrics:dict):
    system_message = """You are a speech coaching assistant. You will be given a speaking prompt, 
a transcript of someone's spoken response, and some quantitative metrics about that response.

Respond ONLY with valid JSON in exactly this shape, nothing else, no markdown formatting:

{
  "clarity_score": <integer 1-10>,
  "structure_score": <integer 1-10>,
  "conciseness_score": <integer 1-10>,
  "main_strength": "<one sentence>",
  "main_weakness": "<one sentence>",
  "action_item": "<one specific, actionable suggestion>"
}"""
    user_message= f"""prompt they were responding to:{prompt}
Transcript:{transript}

Metrics:
- Word Count :{metrics["word_count"]}
- Filler Words used: {metrics["filler_count"]}
- Words per minute: {metrics["words_per_minute"]}
- Duration :{metrics.get('duration-seconds','unknown')} seconds"""

    response = client.chat.completions.create(
        model = 'openai/gpt-oss-120b',
        messages = [
            {"role":"system","content":system_message},
            {"role":"user","content":user_message}
        ],
        response_format={"type":"json_object"}
    )
    raw_text=response.choices[0].message.content
    feedback = json.loads(raw_text)
    return feedback

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

    llm_feedback = get_llm_feedback(prompt, transcript, metrics)


    result={
    
        "prompt":prompt,
        "transcript":transcript,
        "duration_seconds":round(duration,1),
        **metrics,
        **llm_feedback
    }
    return result