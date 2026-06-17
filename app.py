from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pipeline import RAGPipeline

from PIL import Image
import pytesseract
import cv2
import numpy as np
import io

app = FastAPI(
    title="Pharmacy OCR Chatbot API",
    description="API for Pharmacy Chatbot, OCR Prescription Reading and Prescription Analysis",
    version="1.0.0",
    openapi_tags=[
        {"name": "Chat Bot", "description": "Ask the pharmacy chatbot about medicines"},
        {"name": "OCR", "description": "Extract medicines from prescription images"},
        {"name": "Prescription Analysis", "description": "OCR + Chatbot integration"},
    ],
)

rag = RAGPipeline()

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return thresh


def extract_medicine_names(text):
    lines = text.split("\n")
    medicines = []

    ignore_words = [
        "dr", "doctor", "clinic", "patient",
        "date", "age", "name", "prescription", "rx"
    ]

    for line in lines:
        clean = line.strip()

        if len(clean) < 3:
            continue

        if any(word in clean.lower() for word in ignore_words):
            continue

        medicines.append(clean)

    return medicines


@app.get("/", tags=["Home"], summary="API Home")
def home():
    return {
        "message": "Pharmacy OCR Chatbot API is running",
        "ui": "/ui",
        "docs": "/docs",
        "endpoints": {
            "chatbot": "/ask?query=panadol",
            "ocr": "/ocr/prescription",
            "analysis": "/prescription/analyze",
        },
    }


@app.get("/ask", tags=["Chat Bot"], summary="Ask Chatbot")
def ask(query: str):
    return {"response": rag.run(query)}


@app.post(
    "/ocr/prescription",
    tags=["OCR"],
    summary="Extract Medicines From Prescription"
)
async def prescription_ocr(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        processed_img = preprocess_image(image_bytes)

        text = pytesseract.image_to_string(
            processed_img,
            lang="eng",
            config="--psm 6"
        )

        medicines = extract_medicine_names(text)

        return {
            "filename": file.filename,
            "raw_text": text,
            "medicines": medicines,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/prescription/analyze",
    tags=["Prescription Analysis"],
    summary="OCR + Chatbot Analysis"
)
async def prescription_analyze(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        processed_img = preprocess_image(image_bytes)

        text = pytesseract.image_to_string(
            processed_img,
            lang="eng",
            config="--psm 6"
        )

        medicines = extract_medicine_names(text)

        results = []

        for med in medicines:
            bot_response = rag.run(med)
            results.append({
                "medicine": med,
                "chatbot_response": bot_response,
            })

        return {
            "filename": file.filename,
            "raw_text": text,
            "medicines_count": len(medicines),
            "results": results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ui", response_class=HTMLResponse, tags=["UI"], summary="Web UI")
def ui_page():
    return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Pharmacy OCR Chatbot</title>
</head>
<body>
    <h1>💊 Pharmacy OCR Chatbot</h1>

    <h2>Chat Bot</h2>
    <input id="query" placeholder="اكتب اسم الدواء مثل panadol">
    <button onclick="askChatbot()">Ask</button>
    <pre id="chatResult"></pre>

    <hr>

    <h2>OCR Prescription</h2>
    <input type="file" id="ocrFile">
    <button onclick="ocrPrescription()">Read Prescription</button>
    <pre id="ocrResult"></pre>

    <hr>

    <h2>OCR + Chatbot Analysis</h2>
    <input type="file" id="analyzeFile">
    <button onclick="analyzePrescription()">Analyze Prescription</button>
    <pre id="analyzeResult"></pre>

<script>
async function askChatbot() {
    const query = document.getElementById("query").value;
    const res = await fetch(`/ask?query=${encodeURIComponent(query)}`);
    const data = await res.json();
    document.getElementById("chatResult").textContent = JSON.stringify(data, null, 2);
}

async function ocrPrescription() {
    const file = document.getElementById("ocrFile").files[0];
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/ocr/prescription", {
        method: "POST",
        body: formData
    });

    const data = await res.json();
    document.getElementById("ocrResult").textContent = JSON.stringify(data, null, 2);
}

async function analyzePrescription() {
    const file = document.getElementById("analyzeFile").files[0];
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("/prescription/analyze", {
        method: "POST",
        body: formData
    });

    const data = await res.json();
    document.getElementById("analyzeResult").textContent = JSON.stringify(data, null, 2);
}
</script>
</body>
</html>
"""