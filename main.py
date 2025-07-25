import gc
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import io
from PIL import Image
from openai import OpenAI

# 🔹 PaddleOCR importeras endast vid behov
ocr = None

app = FastAPI()
client = OpenAI()

def get_ocr():
    global ocr
    if ocr is None:
        from paddleocr import PaddleOCR
        # 🔹 Minimal modell för låg minnesanvändning
        ocr = PaddleOCR(use_angle_cls=False, lang='en')  
    return ocr

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        r = get_ocr()

        # 🔹 Läs in bilden
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image.save("temp_image.jpg")  # PaddleOCR kräver fil

        # 🔹 OCR-anrop
        ocr_result = r.ocr("temp_image.jpg", cls=False)
        # Konvertera resultatet till en enkel lista med text
        detected_texts = [line[1][0] for line in ocr_result[0]] if ocr_result else []

        # 🔹 GPT-tolkning av parkeringsreglerna
        gpt_prompt = (
            "Du är en expert på svenska parkeringsregler. "
            f"Här är texten från en parkeringsskylt: {detected_texts}. "
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

        return JSONResponse(content={"ocr_text": detected_texts, "answer": answer})

    except Exception as e:
        return JSONResponse(content={"error": f"Fel vid analys: {str(e)}"})

    finally:
        gc.collect()

@app.get("/")
async def root():
    return {"status": "ok", "message": "Parkerings-API är igång (PaddleOCR-version)!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
