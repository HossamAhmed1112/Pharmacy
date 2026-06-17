from fastapi import FastAPI, UploadFile, File
from fastapi import HTTPException
from PIL import Image
import pytesseract
import cv2
import numpy as np
import re
import io
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI(title="Prescription OCR API")

# غيّر المسار حسب مكان تثبيت Tesseract عندك
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # تحسين الصورة
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # تحويل أبيض وأسود
    thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    return thresh


def extract_medicines(text):
    lines = text.split("\n")

    medicines = []

    for line in lines:
        clean = line.strip()

        if len(clean) < 3:
            continue

        # تجاهل كلمات ملهاش علاقة غالبًا
        ignore_words = ["dr", "doctor", "clinic", "patient", "date", "age"]
        if any(word in clean.lower() for word in ignore_words):
            continue

        medicines.append({
            "raw_line": clean,
            "name": clean,
            "dose": extract_dose(clean),
            "frequency": extract_frequency(clean)
        })

    return medicines


def extract_dose(line):
    pattern = r"\b(\d+\s?(mg|ml|g|mcg|IU|tab|tablet|capsule|cap))\b"
    match = re.search(pattern, line, re.IGNORECASE)
    return match.group(0) if match else None


def extract_frequency(line):
    patterns = [
        r"once daily",
        r"twice daily",
        r"3 times daily",
        r"every \d+ hours",
        r"\d+x\d+",
        r"\d+ times"
    ]

    for p in patterns:
        match = re.search(p, line, re.IGNORECASE)
        if match:
            return match.group(0)

    return None


from fastapi import HTTPException

@app.post("/ocr/prescription")
async def prescription_ocr(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        processed_img = preprocess_image(image_bytes)

        text = pytesseract.image_to_string(
            processed_img,
            lang="ara+eng",
            config="--psm 6"
        )

        medicines = extract_medicines(text)

        return {
            "filename": file.filename,
            "raw_text": text,
            "medicines": medicines
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))