# ========================================
# AI Parking Sign Reader – FastAPI backend för Render
# ========================================

from fastapi import FastAPI, File, UploadFile
import easyocr
from openai import OpenAI
import datetime
import os

# ✅ Byt ut mot din riktiga API-nyckel innan du deployar
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "DIN_API_NYCKEL_HÄR")

# Initiera klienter
client = OpenAI(api_key=OPENAI_API_KEY)
reader = easyocr.Reader(['sv', 'en'])

# Skapa FastAPI-app
app = FastAPI()

@app.get("/")
def root():
    return {"status": "Backend is running!"}

@app.post("/analyze")
async def analyze_parking_sign(image: UploadFile = File(...)):
    # 1. Spara bilden temporärt
    temp_filename = f"temp_{image.filename}"
    with open(temp_filename, "wb") as f:
        f.write(await image.read())

    # 2. OCR
    ocr_results = reader.readtext(temp_filename, detail=0)
    ocr_text = " ".join(ocr_results)

    # 3. Nuvarande tid & dag
    now = datetime.datetime.now()
    current_time = now.strftime("%H:%M")
    current_day = now.strftime("%A")

    # 4. GPT-regeltolkning
    prompt = f"""
    Du är en svensk parkeringsregeltolkare.

    Input (OCR-text från skylten): "{ocr_text}"
    Nuvarande tid: {current_time}
    Nuvarande dag: {current_day}

    Svara enligt följande format:

    1. Börja med ett klart JA/NEJ-svar:
    - "Du FÅR parkera här just nu."
    - "Du får INTE parkera här just nu."

    2. Lägg till en kort förklaring på en rad:
    - Om parkering är tillåten: "Du får parkera fram till kl. XX:XX."
    - Om parkering inte är tillåten: "Du får parkera här först kl. XX:XX."
    """

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Du är en expert på svenska parkeringsregler."},
            {"role": "user", "content": prompt}
        ]
    )

    gpt_answer = completion.choices[0].message.content

    # 5. Rensa upp temporär fil
    os.remove(temp_filename)

    # 6. Returnera svar
    allowed = "INTE" not in gpt_answer.upper()
    return {
        "allowed": allowed,
        "ocr_text": ocr_text,
        "gpt_answer": gpt_answer
    }
