import gc
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import io
from PIL import Image
from openai import OpenAI

app = FastAPI()

reader = None
client = OpenAI()

def get_reader():
    global reader
    if reader is None:
        import easyocr  # 🔹 Importeras först vid behov
        reader = easyocr.Reader(['en'], gpu=False, detector=False, verbose=False)
    return reader

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        r = get_reader()
        image_bytes = await file.read()
        _ = Image.open(io.BytesIO(image_bytes))  # Validera att det är en bild

        # 🔹 Försök använda endast igenkänning utan detektering
        try:
            result = r.recognize(image_bytes, detail=0)  # vissa versioner stöder detta
        except AttributeError:
            result = r.readtext(image_bytes, detail=0)  # fallback

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
        gc.collect()

@app.get("/")
async def root():
    return {"status": "ok", "message": "Parkerings-API är igång!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
