import streamlit as st
import math
import tempfile
import os

from pipeline import RAGPipeline
from ocr_service import extract_medicines_from_image, clean_medicine_name


st.set_page_config(
    page_title="Pharmacy OCR Chatbot",
    page_icon="💊",
    layout="wide"
)


@st.cache_resource
def load_rag():
    return RAGPipeline()


rag = load_rag()


def safe_value(value):
    if value is None:
        return "غير متوفر"

    if isinstance(value, float) and math.isnan(value):
        return "غير متوفر"

    if str(value).lower() == "nan":
        return "غير متوفر"

    return value


def display_drug_card(drug):
    st.markdown(f"""
### 💊 {safe_value(drug.get("name_ar"))}

**الاسم الإنجليزي:** {safe_value(drug.get("name"))}

**الاستخدام:** {safe_value(drug.get("uses"))}

**السعر:** {safe_value(drug.get("price"))} جنيه

**العبوة:** {safe_value(drug.get("packaging"))}

**الجرعة:** {safe_value(drug.get("dosage"))}

**الآثار الجانبية:** {safe_value(drug.get("side_effects"))}

**التركيز:** {safe_value(drug.get("concentration"))}
""")
    st.divider()


def display_response(response):
    if isinstance(response, list):
        if len(response) == 0:
            st.warning("لا توجد نتيجة مناسبة.")
            return

        st.success(f"تم العثور على {len(response)} نتيجة")

        for i, drug in enumerate(response, 1):
            if not isinstance(drug, dict):
                st.write(drug)
                continue

            title = drug.get("name") or drug.get("name_ar") or f"Medicine {i}"

            with st.expander(f"💊 {i}. {title}"):
                display_drug_card(drug)

    elif isinstance(response, dict):
        if "__intent_response__" in response:
            st.markdown(response["__intent_response__"])
        else:
            display_drug_card(response)

    else:
        st.write(response)


def format_single_drug_card(drug):
    return f"""
<div class="drug-card">
    <div class="drug-title">
        💊 {safe_value(drug.get("name_ar"))}
    </div>

    <div class="drug-row">
        <span>📦 اسم المنتج:</span>
        <p>{safe_value(drug.get("name"))}</p>
    </div>

    <div class="drug-row">
        <span>🇪🇬 الاسم بالعربي:</span>
        <p>{safe_value(drug.get("name_ar"))}</p>
    </div>

    <div class="drug-row">
        <span>🔹 العبوة:</span>
        <p>{safe_value(drug.get("packaging"))}</p>
    </div>

    <div class="drug-row">
        <span>💰 السعر:</span>
        <p>{safe_value(drug.get("price"))} جنيه</p>
    </div>

    <div class="drug-row">
        <span>🏷️ السعر بعد الخصم:</span>
        <p>{safe_value(drug.get("discounted_price"))} جنيه</p>
    </div>

    <div class="drug-row">
        <span>✨ نسبة الخصم:</span>
        <p>{safe_value(drug.get("discount_percentage"))}</p>
    </div>

    <div class="drug-row">
        <span>💊 الاستخدام:</span>
        <p>{safe_value(drug.get("uses"))}</p>
    </div>

    <div class="drug-row">
        <span>⚠️ الآثار الجانبية:</span>
        <p>{safe_value(drug.get("side_effects"))}</p>
    </div>

    <div class="drug-row">
        <span>🕒 الجرعة:</span>
        <p>{safe_value(drug.get("dosage"))}</p>
    </div>

    <div class="drug-row">
        <span>🧪 التركيز:</span>
        <p>{safe_value(drug.get("concentration"))}</p>
    </div>
</div>
"""


def format_response_as_text(response):
    if isinstance(response, list):
        if len(response) == 0:
            return "❌ لم يتم العثور على أي نتائج."

        html = f"""
<b>🔎 تم العثور على {len(response)} نتيجة</b>
<br><br>
"""

        for i, drug in enumerate(response, 1):
            if not isinstance(drug, dict):
                continue

            html += f"""
<details>
<summary>💊 {i}. {safe_value(drug.get("name"))}</summary>

<br>

<b>🇪🇬 الاسم بالعربي:</b><br>
{safe_value(drug.get("name_ar"))}

<br><br>

<b>💰 السعر:</b><br>
{safe_value(drug.get("price"))} جنيه

<br><br>

<b>📦 العبوة:</b><br>
{safe_value(drug.get("packaging"))}

<br><br>

<b>💊 الاستخدام:</b><br>
{safe_value(drug.get("uses"))}

<br><br>

<b>⚠️ الآثار الجانبية:</b><br>
{safe_value(drug.get("side_effects"))}

<br><br>

<b>🕒 الجرعة:</b><br>
{safe_value(drug.get("dosage"))}

<br><br>

<b>🧪 التركيز:</b><br>
{safe_value(drug.get("concentration"))}

<br><br>

<b>🏷️ السعر بعد الخصم:</b><br>
{safe_value(drug.get("discounted_price"))} جنيه

<br><br>

<b>✨ نسبة الخصم:</b><br>
{safe_value(drug.get("discount_percentage"))}

<br><br>
</details>
<br>
"""

        return html

    elif isinstance(response, dict):
        if "__intent_response__" in response:
            return format_plain_text_response(str(response["__intent_response__"]))

        return format_single_drug_card(response)

    else:
        return format_plain_text_response(str(response))

def run_ocr(uploaded_file):
    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    raw_text, medicines = extract_medicines_from_image(tmp_path)

    os.remove(tmp_path)

    return raw_text, medicines


st.markdown("""
<style>
body {
    background-color: #f0f2f5;
}

.main-title {
    text-align: center;
    color: #075e54;
}

.chat-wrapper {
    background-color: #efeae2;
    padding: 18px;
    border-radius: 18px;
    min-height: 520px;
    max-height: 520px;
    overflow-y: auto;
    border: 1px solid #ddd;
}

.user-msg {
    background-color: #dcf8c6;
    color: #111;
    padding: 10px 14px;
    border-radius: 14px 14px 0 14px;
    margin: 10px 0 10px auto;
    max-width: 70%;
    text-align: right;
    direction: rtl;
    box-shadow: 0 1px 2px rgba(0,0,0,0.12);
    font-size: 15px;
    line-height: 1.6;
}

.bot-msg {
    background-color: #ffffff;
    color: #111;
    padding: 12px;
    border-radius: 14px 14px 14px 0;
    margin: 10px auto 10px 0;
    max-width: 78%;
    text-align: right;
    direction: rtl;
    box-shadow: 0 1px 2px rgba(0,0,0,0.12);
    font-size: 15px;
    line-height: 1.7;
}

.chat-header {
    background-color: #075e54;
    color: white;
    padding: 14px 18px;
    border-radius: 16px 16px 0 0;
    font-size: 20px;
    font-weight: bold;
}

.chat-subtitle {
    font-size: 13px;
    color: #d9fdd3;
    margin-top: 3px;
}

.drug-card {
    background: #ffffff;
    padding: 16px;
    border-radius: 15px;
    border: 1px solid #e0e0e0;
}

.drug-title {
    font-size: 19px;
    font-weight: bold;
    color: #075e54;
    margin-bottom: 14px;
    border-bottom: 1px solid #ddd;
    padding-bottom: 10px;
}

.drug-row {
    margin-bottom: 13px;
    padding-bottom: 10px;
    border-bottom: 1px solid #eeeeee;
}

.drug-row span {
    display: block;
    font-weight: bold;
    color: #075e54;
    margin-bottom: 4px;
}

.drug-row p {
    margin: 0;
    color: #222;
    line-height: 1.7;
}

.drug-details {
    background: #f8f9fa;
    border: 1px solid #e1e1e1;
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 12px;
}

.drug-details summary {
    cursor: pointer;
    font-weight: bold;
    color: #075e54;
    font-size: 15px;
    padding: 8px;
}

.drug-details[open] {
    background: #ffffff;
}
</style>
""", unsafe_allow_html=True)


st.markdown(
    "<h1 class='main-title'>💊 Pharmacy OCR Chatbot</h1>",
    unsafe_allow_html=True
)

st.write(
    "استخدم الشات بوت أو ارفع روشتة كمبيوتر والبرنامج يقرأ الأدوية ويعرض معلوماتها."
)


tab1, tab2, tab3 = st.tabs([
    "🤖 Chat Bot",
    "📄 OCR Prescription",
    "🧾 OCR + Chatbot Analysis"
])


with tab1:
    st.markdown("""
<div class="chat-header">
    💊 Pharmacy Assistant
    <div class="chat-subtitle">online</div>
</div>
""", unsafe_allow_html=True)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

    if len(st.session_state.chat_messages) == 0:
        st.markdown(
            """
            <div class="bot-msg">
                أهلاً 👋<br>
                اسألني عن أي دواء، سعره، استخدامه، جرعته أو آثاره الجانبية.
            </div>
            """,
            unsafe_allow_html=True
        )

    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-msg">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="bot-msg">
                    {msg["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown('</div>', unsafe_allow_html=True)

    query = st.chat_input("اكتب رسالتك هنا...")

    if query:
        st.session_state.chat_messages.append({
            "role": "user",
            "content": query
        })

        with st.spinner("جاري الرد..."):
            response = rag.run(query)

        bot_text = format_response_as_text(response)

        st.session_state.chat_messages.append({
            "role": "bot",
            "content": bot_text
        })

        st.rerun()


with tab2:
    st.header("📄 OCR Prescription")

    uploaded_file = st.file_uploader(
        "ارفع صورة الروشتة",
        type=["png", "jpg", "jpeg"],
        key="ocr_file"
    )

    if uploaded_file is not None:
        st.image(
            uploaded_file,
            caption="Uploaded Prescription",
            use_container_width=True
        )

        if st.button("Read Prescription"):
            with st.spinner("جاري قراءة الروشتة..."):
                raw_text, medicines = run_ocr(uploaded_file)

            st.subheader("النص المستخرج")
            st.text_area("", raw_text, height=250)

            st.subheader("الأدوية المكتشفة")
            if medicines:
                for med in medicines:
                    st.write(f"💊 {med}")
            else:
                st.warning("لم يتم العثور على أدوية واضحة في الصورة.")


with tab3:
    st.header("🧾 OCR + Chatbot Analysis")

    analyze_file = st.file_uploader(
        "ارفع صورة الروشتة للتحليل الكامل",
        type=["png", "jpg", "jpeg"],
        key="analyze_file"
    )

    if analyze_file is not None:
        st.image(
            analyze_file,
            caption="Uploaded Prescription",
            use_container_width=True
        )

        if st.button("Analyze Prescription"):
            with st.spinner("جاري قراءة الروشتة وتحليل الأدوية..."):
                raw_text, medicines = run_ocr(analyze_file)

                results = []

                for med in medicines:
                    clean_med = clean_medicine_name(med)
                    bot_response = rag.run(clean_med)

                    results.append({
                        "medicine": clean_med,
                        "chatbot_response": bot_response
                    })

            st.subheader("النص المستخرج")
            st.text_area("", raw_text, height=250)

            st.subheader("عدد الأدوية المكتشفة")
            st.write(len(medicines))

            st.subheader("نتائج التحليل")

            if results:
                for item in results:
                    st.markdown(
                        f"## 🔎 الدواء المقروء من الروشتة: {item['medicine']}"
                    )
                    display_response(item["chatbot_response"])
            else:
                st.warning("لم يتم العثور على أدوية لتحليلها.")