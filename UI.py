import streamlit as st
from pipeline import RAGPipeline
from llm import generate_response
import pandas as pd
import numpy as np
import json
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Pharmacy Assistant",
    page_icon="💊"
)


def create_product_summary(drug):
    return {
        "name": drug.get("name"),
        "name_ar": drug.get("name_ar"),
        "packaging": drug.get("packaging"),
        "price": drug.get("price"),
        "discounted_price": drug.get("discounted_price"),
        "discount_percentage": drug.get("discount_percentage"),
        "uses": drug.get("uses"),
        "side_effects": drug.get("side_effects"),
        "dosage": drug.get("dosage"),
        "concentration": drug.get("concentration"),
    }


def create_compare_table(products):
    rows = []
    for drug in products:
        rows.append({
            "اسم": drug.get("name"),
            "العبوة": drug.get("packaging"),
            "السعر": drug.get("price"),
            "السعر بعد الخصم": drug.get("discounted_price"),
            "نسبة الخصم": drug.get("discount_percentage"),
            "التصنيف": drug.get("category") or drug.get("type") or "",
            "التركيب": drug.get("active_ingredient") or "",
        })
    return pd.DataFrame(rows)


def render_result(content, history_mode=False):
    if isinstance(content, str):
        st.markdown(content)

    elif isinstance(content, list):
        if not content:
            st.markdown("عذراً، لم أتمكن من العثور على المنتج المطلوب.")
            return

        st.info(f"تم العثور على {len(content)} نتائج.")

        df = pd.DataFrame([
            {
                "اسم": d.get("name"),
                "العبوة": d.get("packaging"),
                "السعر": d.get("price"),
                "السعر بعد الخصم": d.get("discounted_price"),
                "نسبة الخصم": d.get("discount_percentage"),
            }
            for d in content
        ])

        st.dataframe(df, use_container_width=True)

        if history_mode:
            return

        for i, d in enumerate(content):
            with st.container():
                cols = st.columns([4, 1, 1])

                with cols[0]:
                    st.write(f"**{d.get('name')}** — {d.get('packaging', '')}")

                with cols[1]:
                    if st.button("🛒 سلة", key=f"add_{st.session_state.last_query}_{i}"):
                        st.session_state.cart.append(d)
                        st.success("أُضيف للسلة ✅")

                with cols[2]:
                    if st.button("⚖️ قارن", key=f"cmp_{st.session_state.last_query}_{i}"):
                        st.session_state.compare.append(d)
                        st.info("أُضيف للمقارنة")

                with st.expander(f"📋 تفاصيل: {d.get('name')}"):
                    detail_cols = st.columns(2)

                    with detail_cols[0]:
                        st.markdown(f"**💊 الاسم:** {d.get('name') or '—'}")
                        st.markdown(f"**🧪 العنصر النشط:** {d.get('active_ingredient') or '—'}")
                        st.markdown(f"**📦 العبوة:** {d.get('packaging') or '—'}")
                        st.markdown(f"**🏷️ التصنيف:** {d.get('category') or d.get('type') or '—'}")

                    with detail_cols[1]:
                        price = d.get("price")
                        disc_price = d.get("discounted_price")
                        disc_pct = d.get("discount_percentage")

                        st.markdown(f"**💰 السعر:** {price or '—'}")

                        if disc_price and disc_price != price:
                            st.markdown(f"**🔖 السعر بعد الخصم:** {disc_price}")

                        if disc_pct and not (isinstance(disc_pct, float) and np.isnan(disc_pct)):
                            st.markdown(f"**📉 نسبة الخصم:** {disc_pct}%")

                    resp_text = generate_response(st.session_state.last_query or d.get("name", ""), d)

                    st.markdown("---")
                    st.markdown(resp_text)

                    copy_text = (
                        f"اسم: {d.get('name')}\n"
                        f"العبوة: {d.get('packaging')}\n"
                        f"السعر: {d.get('price')}\n"
                        f"السعر بعد الخصم: {d.get('discounted_price')}\n"
                        f"نسبة الخصم: {d.get('discount_percentage')}"
                    )

                    components.html(
                        f"""
                        <button id='copy-btn-{i}' style='padding:4px 12px;cursor:pointer;'>نسخ التفاصيل 📋</button>
                        <span id='copy-msg-{i}' style='margin-right:8px;color:green;font-size:13px;'></span>
                        <script>
                        document.getElementById('copy-btn-{i}').addEventListener('click', () => {{
                            navigator.clipboard.writeText({json.dumps(copy_text)}).then(() => {{
                                document.getElementById('copy-msg-{i}').innerText = 'تم النسخ ✅';
                                setTimeout(() => {{
                                    document.getElementById('copy-msg-{i}').innerText = '';
                                }}, 2500);
                            }});
                        }});
                        </script>
                        """,
                        height=45,
                    )

    elif isinstance(content, dict):
        df = pd.DataFrame([{
            "اسم": content.get("name"),
            "العبوة": content.get("packaging"),
            "السعر": content.get("price"),
            "السعر بعد الخصم": content.get("discounted_price"),
            "نسبة الخصم": content.get("discount_percentage"),
        }])

        st.dataframe(df, use_container_width=True)

        if history_mode:
            return

        resp = generate_response(st.session_state.last_query or content.get("name", ""), content)
        st.markdown(resp)

    else:
        st.markdown("عذراً، حدث خطأ غير متوقع.")


@st.cache_resource
def load_rag():
    return RAGPipeline()


rag = load_rag()

st.title("💊 Pharmacy Assistant")

for key, default in [
    ("messages", []),
    ("cart", []),
    ("compare", []),
    ("last_result", None),
    ("last_query", ""),
    ("show_compare", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def do_search(query: str):
    query = query.strip()
    if not query:
        return

    result = rag.run(query)

    st.session_state.last_query = query
    st.session_state.last_result = result

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": result
    })


with st.sidebar:
    if st.button("🗑️ محادثة جديدة"):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.session_state.last_query = ""
        st.rerun()

    st.markdown("---")
    st.header("السلة")

    if st.session_state.cart:
        cart_df = pd.DataFrame([create_product_summary(d) for d in st.session_state.cart])
        total_price = sum(float(d.get("price") or 0) for d in st.session_state.cart)

        st.write(f"عدد المنتجات: {len(st.session_state.cart)}")
        st.write(f"الإجمالي التقريبي: {total_price:.2f}")
        st.dataframe(cart_df, use_container_width=True)

        st.download_button(
            "تحميل CSV للسلة",
            cart_df.to_csv(index=False).encode("utf-8-sig"),
            "cart.csv",
            "text/csv",
        )

        if st.button("إزالة آخر منتج من السلة", key="remove_last_cart"):
            st.session_state.cart.pop()
            st.rerun()

        if st.button("مسح السلة", key="clear_cart"):
            st.session_state.cart = []
            st.rerun()
    else:
        st.write("السلة فارغة")

    st.markdown("---")
    st.header("قائمة المقارنة")

    if st.session_state.compare:
        compare_df = pd.DataFrame([create_product_summary(d) for d in st.session_state.compare])

        st.write(f"عدد المنتجات للمقارنة: {len(st.session_state.compare)}")
        st.dataframe(compare_df, use_container_width=True)

        st.download_button(
            "تحميل CSV للمقارنة",
            compare_df.to_csv(index=False).encode("utf-8-sig"),
            "compare.csv",
            "text/csv",
        )

        if st.button("إزالة آخر منتج من المقارنة", key="remove_last_compare"):
            st.session_state.compare.pop()
            st.rerun()

        if st.button("مسح المقارنة", key="clear_compare"):
            st.session_state.compare = []
            st.rerun()

        if st.button("عرض مقارنة مفصلة", key="show_compare_detail"):
            st.session_state.show_compare = True

        if st.button("إخفاء المقارنة", key="hide_compare_detail"):
            st.session_state.show_compare = False

        if st.session_state.show_compare:
            st.dataframe(create_compare_table(st.session_state.compare), use_container_width=True)
    else:
        st.write("قائمة المقارنة فارغة")

    st.markdown("---")

    if st.button("عرض الاستعلامات الفاشلة", key="show_failed"):
        try:
            with open("failed_queries.log", "r", encoding="utf-8") as f:
                lines = f.readlines()
            st.text_area("استعلامات فاشلة", "".join(lines[-20:]), height=300)
        except FileNotFoundError:
            st.write("لا توجد استعلامات فاشلة حتى الآن")


if st.session_state.get("pending_search"):
    pending = st.session_state.pop("pending_search")
    do_search(pending)
    st.rerun()


search_input = st.text_input(
    "ابحث باسم المنتج أو العنصر النشط أو الفئة:",
    placeholder="مثال: بانادول، فولتارين، سعر بانادول، اثار جانبية فولتارين",
)

search_examples = ["بانادول", "فولتارين", "أوجمنتين", "علاج السعال", "مسكن", "فيتامين"]

if not search_input:
    st.write("### بحث سريع")
    example_cols = st.columns(len(search_examples))

    for idx, example in enumerate(search_examples):
        if example_cols[idx].button(example, key=f"example_btn_{idx}"):
            st.session_state["pending_search"] = example
            st.rerun()

else:
    prev_input = st.session_state.get("_last_suggest_input", "")

    if search_input != prev_input:
        st.session_state["_suggestions_cache"] = rag.retriever.search_top_k(search_input, k=8)
        st.session_state["_last_suggest_input"] = search_input

    suggestions_ac = st.session_state.get("_suggestions_cache", [])

    exact_match = None

    for d, sc in suggestions_ac:
        candidate = (d.get("name") or "").strip().lower()
        if candidate == search_input.strip().lower():
            exact_match = d
            break

    if exact_match is not None:
        st.caption(f"✅ تم التعرف على المنتج: **{exact_match.get('name')}**")

        if st.button("بحث", key="search_button_exact"):
            do_search(exact_match.get("name") or search_input)
            st.rerun()

    else:
        if suggestions_ac:
            ac_names = [
                f"{d.get('name')} — {d.get('name_ar')} — {d.get('packaging')}"
                for d, sc in suggestions_ac
            ]

            selected_suggestion = st.selectbox(
                "اختر من الاقتراحات:",
                options=[""] + ac_names,
            )

            if selected_suggestion:
                search_input = selected_suggestion.split(" — ")[0]

        st.caption("اقتراحات البحث تظهر مباشرةً حسب الاسم، العبوة، العنصر النشط أو الفئة.")

        suggestions = suggestions_ac[:6]

        if suggestions:
            st.write("### اقتراحات سريعة")
            cols = st.columns(min(len(suggestions), 3))

            for idx, (d, sc) in enumerate(suggestions):
                col = cols[idx % len(cols)]

                label = (
                    f"{d.get('name')} — {d.get('packaging', 'غير محدد')} — "
                    f"{d.get('active_ingredient') or d.get('category') or 'بدون فئة'} ({sc:.0%})"
                )

                if col.button(label, key=f"suggestion_btn_{idx}"):
                    st.session_state["pending_search"] = d.get("name") or search_input
                    st.rerun()

            st.write("أو اضغط زر البحث للحصول على نتائج بالاستعلام الحالي.")

            if st.button("بحث", key="search_button"):
                do_search(search_input)
                st.rerun()

        else:
            st.info("لا توجد اقتراحات دقيقة لهذا الاستعلام. اضغط بحث للبحث عبر جميع المنتجات.")

            if st.button("بحث", key="search_button_fallback"):
                do_search(search_input)
                st.rerun()


for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            is_last_assistant = idx == len(st.session_state.messages) - 1
            render_result(msg["content"], history_mode=not is_last_assistant)
        else:
            st.markdown(str(msg["content"]))


if prompt := st.chat_input("اكتب سؤالك أو اسم دواء جديد..."):
    do_search(prompt)
    st.rerun()


if len(st.session_state.compare) >= 2:
    st.markdown("---")
    st.subheader("مقارنة سريعة للمنتجات المختارة")
    st.dataframe(create_compare_table(st.session_state.compare), use_container_width=True)

elif st.session_state.compare:
    st.markdown("---")
    st.write("أضف منتجين على الأقل إلى المقارنة لعرض الجدول التفصيلي.")