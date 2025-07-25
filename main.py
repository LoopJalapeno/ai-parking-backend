import gc
import easyocr
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import io
from PIL import Image
import base64
from openai import OpenAI

app = FastAPI()

reader = None  # Lazy-load

# GPT-klienten
client = OpenAI()

def init_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=False)

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    global reader
    try:
        # Lazy-load OCR
        init_reader()

        # Läs bilden
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        result = reader.readtext(image_bytes, detail=0)

        # GPT-regeltolkning
        gpt_prompt = (
            "Du är en expert på svenska parkeringsregler. "
            "Här är texten från en parkeringsskylt: "
            f"{result}. "
            "Svara på svenska om man får parkera just nu och fram till när. Var tydlig."
        )

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Du är en hjälpsam assistent."},
                {"role": "user", "content": gpt_prompt}
            ]
        )

        answer = completion.choices[0].message.content

        return JSONResponse(content={"ocr_text": result, "answer": answer})

    except Exception as e:
        return JSONResponse(content={"error": f"Fel vid analys: {str(e)}"})

    finally:
        # Rensa minne efter varje körning
        reader = None
        gc.collect()

@app.get("/")
async def root():
    return {"status": "ok", "message": "Parkerings-API är igång!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
