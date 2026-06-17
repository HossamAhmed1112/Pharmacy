import pandas as pd

def clean_value(value, default="غير متوفر"):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    value = str(value).strip()
    if value == "" or value.lower() == "nan":
        return default
    return value


FIELD_KEYWORDS = {
    "price": ["سعر", "بكام", "بكم", "تمن", "ثمن", "price", "cost"],
    "discounted_price": ["بعد الخصم", "خصم", "عرض", "discount"],
    "packaging": ["عبوة", "packaging"],
    "uses": ["استخدام", "بيستخدم", "يعالج", "علاج", "دواعي", "uses", "use"],
    "side_effects": ["اثار", "آثار", "اعراض", "أعراض", "جانبية", "side"],
    "dosage": ["جرعة", "جرعات", "dosage", "dose"],
    "concentration": ["تركيز", "concentration", "strength"],
}


def detect_requested_field(query):
    q = query.lower()
    for field, keywords in FIELD_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in q:
                return field
    return None


def generate_response(query: str, drug: dict) -> str:
    field = detect_requested_field(query)

    name = clean_value(drug.get("name"))
    name_ar = clean_value(drug.get("name_ar"))

    labels = {
        "price": "السعر",
        "discounted_price": "السعر بعد الخصم",
        "packaging": "العبوة",
        "uses": "الاستخدام",
        "side_effects": "الآثار الجانبية",
        "dosage": "الجرعة",
        "concentration": "التركيز",
    }

    if field:
        value = clean_value(drug.get(field))
        if field in ["price", "discounted_price"] and value != "غير متوفر":
            value = f"{value} جنيه"

        return f"💊 {name_ar} - {name}\n🔹 {labels[field]}: {value}"

    return format_product_info(drug)


def format_product_info(drug):
    return (
        f"📦 اسم المنتج: {clean_value(drug.get('name'))}\n"
        f"🇪🇬 الاسم بالعربي: {clean_value(drug.get('name_ar'))}\n"
        f"🔹 العبوة: {clean_value(drug.get('packaging'))}\n"
        f"💰 السعر: {clean_value(drug.get('price'))} جنيه\n"
        f"🏷️ السعر بعد الخصم: {clean_value(drug.get('discounted_price'))} جنيه\n"
        f"✨ نسبة الخصم: {clean_value(drug.get('discount_percentage'), 'لا يوجد')}\n"
        f"💊 الاستخدام: {clean_value(drug.get('uses'))}\n"
        f"⚠️ الآثار الجانبية: {clean_value(drug.get('side_effects'))}\n"
        f"🕒 الجرعة: {clean_value(drug.get('dosage'))}\n"
        f"🧪 التركيز: {clean_value(drug.get('concentration'))}"
    )


def general_chat(query):
    return "الدواء غير متوفر أو الاسم غير صحيح."