import re
import easyocr
from difflib import SequenceMatcher

from db import fetch_drugs


ocr_reader = easyocr.Reader(["en"], gpu=False)


def safe_text(value):
    if value is None:
        return ""
    value = str(value).strip()
    if value.lower() in ["nan", "none", "null"]:
        return ""
    return value


def clean_text(text):
    text = str(text)

    # remove Arabic
    text = re.sub(r"[\u0600-\u06FF]+", " ", text)

    # remove Rx / R/
    text = re.sub(r"^(r\/|r\s*/|rx)\s*", "", text, flags=re.IGNORECASE)

    # remove bullets
    text = text.replace("•", "").replace("*", "")

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text

def clean_medicine_name(text):
    return clean_text(text)


def is_noise(text):
    t = text.lower()

    noise_words = [
        "doctor", "dr", "consultant", "cardiologist",
        "clinic", "hospital", "patient", "date",
        "mansour", "farouk", "interventions", "pacemakers",
        "implant", "implanter", "tavi", "phone", "mobile",
        "tel", "fax", "address", "street", "road",
        "signature", "msc", "feb", "fsci"
    ]

    if any(word in t for word in noise_words):
        return True

    if re.fullmatch(r"[\d\s\/\-\:\.]+", t):
        return True

    if len(t) < 3:
        return True

    return False


def medicine_score(text):
    t = text.lower()
    score = 0

    if re.search(r"\d+\s*mg", t):
        score += 3

    if re.search(r"\d+\s*ml", t):
        score += 2

    if re.search(r"\d+\s*/\s*\d+", t):
        score += 3

    if any(x in t for x in [" mr", " xr", " sr"]):
        score += 2

    if re.search(r"[a-zA-Z]{4,}", t):
        score += 2

    if 1 <= len(t.split()) <= 6:
        score += 1

    return score


def normalize_for_match(text):
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b\d+\s*(mg|ml|mcg|g)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def load_database_drugs():
    drugs = fetch_drugs()

    prepared = []

    for drug in drugs:
        name = safe_text(drug.get("name"))
        name_ar = safe_text(drug.get("name_ar"))
        active = safe_text(drug.get("active_ingredient"))
        concentration = safe_text(drug.get("concentration"))

        if not name:
            continue

        prepared.append({
            "drug": drug,
            "name": name,
            "name_ar": name_ar,
            "active_ingredient": active,
            "concentration": concentration,
            "match_text": normalize_for_match(
                f"{name} {active} {concentration}"
            )
        })

    return prepared


DATABASE_DRUGS = load_database_drugs()


def correct_medicine_from_database(ocr_text):
    query = normalize_for_match(ocr_text)

    if not query:
        return None

    best = None
    best_score = 0

    for item in DATABASE_DRUGS:
        db_text = item["match_text"]
        db_name = normalize_for_match(item["name"])

        score = similarity(query, db_text)

        if query in db_name:
            score += 0.35

        first_word = query.split()[0] if query.split() else ""
        if first_word and first_word in db_name:
            score += 0.25

        if score > best_score:
            best_score = score
            best = item

    if best and best_score >= 0.55:
        drug = best["drug"]

        return {
            "ocr_text": ocr_text,
            "corrected_name": safe_text(drug.get("name")),
            "name_ar": safe_text(drug.get("name_ar")),
            "active_ingredient": safe_text(drug.get("active_ingredient")),
            "concentration": safe_text(drug.get("concentration")),
            "score": round(best_score, 3)
        }

    return None


def extract_medicines_from_image(image_path):
    results = ocr_reader.readtext(
        image_path,
        detail=1,
        paragraph=False
    )

    raw_lines = []
    candidates = []

    for item in results:
        text = item[1]
        clean = clean_text(text)

        if not clean:
            continue

        if is_noise(clean):
            continue

        score = medicine_score(clean)

        if score >= 3:
            raw_lines.append(clean)
            candidates.append(clean)

    corrected_results = []
    seen = set()

    for candidate in candidates:
        corrected = correct_medicine_from_database(candidate)

        if corrected:
            corrected_name = corrected["corrected_name"]
            key = corrected_name.lower()

            if key not in seen:
                corrected_results.append(corrected)
                seen.add(key)

    raw_text_lines = []

    for item in corrected_results:
        active = item["active_ingredient"] or "غير متوفر"
        concentration = item["concentration"] or "غير متوفر"

        raw_text_lines.append(
            f"OCR: {item['ocr_text']}  ->  "
            f"Corrected: {item['corrected_name']}  |  "
            f"Active Ingredient: {active}  |  "
            f"Concentration: {concentration}"
        )

    medicines = [
        item["corrected_name"]
        for item in corrected_results
    ]

    return "\n".join(raw_text_lines), medicines