import os
import re
import difflib
import datetime
import numpy as np

from embeddings import embed_texts, embed_query, build_annoy_index, load_annoy_index


INTENT_PATTERNS = {
    "price": [
        "price", "prices", "cost", "costs", "how much",
        "سعر", "السعر", "اسعار", "الاسعار", "أسعار", "الأسعار",
        "بكام", "بكم", "تمن", "تمنه", "تمنها", "ثمن",
        "كام", "كم سعر", "سعره", "سعرها",
    ],

    "side_effects": [
        "side_effects", "side_effect", "side effects", "side effect",
        "adverse", "adverse effects",
        "اثار جانبيه", "اثار جانبية", "الاثار الجانبيه", "الاثار الجانبية",
        "آثار جانبية", "الآثار الجانبية", "أثار جانبية", "الأثار الجانبية",
        "اعراض جانبيه", "اعراض جانبية", "الأعراض الجانبية", "الاعراض الجانبية",
        "تاثيرات جانبيه", "تاثيرات جانبية", "التاثيرات الجانبيه",
        "التاثيرات الجانبية", "تأثيرات جانبية", "التأثيرات الجانبية",
        "مضاعفات",
    ],

    "uses": [
        "uses", "use", "used for", "indication", "indications", "benefits",
        "استخدام", "استخدامات", "الاستخدام", "الاستخدامات",
        "إستخدام", "إستخدامات", "استعمال", "استعمالات",
        "الاستعمال", "الاستعمالات", "بيتستخدم", "بيستخدم",
        "بيستخدم في", "يستخدم في", "بيستخدم لايه", "يستخدم لايه",
        "بيتعمل ايه", "بيعمل ايه", "بيعالج ايه", "يعالج ايه",
        "فايده", "فائدة", "فوائد", "علاج", "علاجات",
    ],

    "dosage": [
        "dosage", "dose", "doses", "daily dose", "how many",
        "جرعه", "جرعة", "الجرعه", "الجرعة", "جرعات", "الجرعات",
        "عدد الجرعات", "عدد الجرعات اليوميه", "عدد الجرعات اليومية",
        "كام جرعه", "كام جرعة", "كم جرعه", "كم جرعة",
        "كام حبه", "كام حبة", "كم حبه", "كم حبة",
        "كام قرص", "كم قرص", "كل قد ايه", "كل قد إيه",
        "اخد", "آخد", "يتاخد", "الكميه", "الكمية", "كمية",
    ],
}


def normalize_text(text):
    text = str(text).strip().lower()
    text = re.sub(r"[؟?.,،؛:!()\[\]{}\"']", " ", text)
    text = re.sub(r"[\-_]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def detect_intent(query: str):
    q = normalize_text(query)

    for intent, keywords in INTENT_PATTERNS.items():
        for kw in keywords:
            if normalize_text(kw) in q:
                return intent

    return None


def extract_drug_name(query: str) -> str:
    q = query.strip()

    all_keywords = [
        kw
        for keywords in INTENT_PATTERNS.values()
        for kw in keywords
    ]

    for kw in sorted(all_keywords, key=len, reverse=True):
        q = re.sub(re.escape(kw), "", q, flags=re.IGNORECASE)

    q = re.sub(r"[؟?.,،؛:!()\[\]{}\"']", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    return q


def format_intent_response(drug: dict, intent: str) -> str:
    name = drug.get("name") or drug.get("name_ar") or "الدواء"

    if intent == "price":
        price = drug.get("price")
        disc = drug.get("discounted_price")
        pct = drug.get("discount_percentage")

        if not price:
            return f"❌ السعر غير متاح لـ **{name}**"

        msg = f"💰 سعر **{name}**: {price} جنيه"

        if disc and str(disc) != str(price):
            msg += f"\n🔖 السعر بعد الخصم: {disc} جنيه"

        if pct and not (isinstance(pct, float) and np.isnan(pct)):
            msg += f"\n📉 نسبة الخصم: {pct}%"

        return msg

    if intent == "side_effects":
        se = drug.get("side_effects")

        if not se or str(se).strip().lower() in ("", "nan", "none"):
            return f"❌ لا تتوفر معلومات عن الآثار الجانبية لـ **{name}**"

        return f"⚠️ الآثار الجانبية لـ **{name}**:\n{se}"

    if intent == "uses":
        uses = drug.get("uses")

        if not uses or str(uses).strip().lower() in ("", "nan", "none"):
            return f"❌ لا تتوفر معلومات عن استخدامات **{name}**"

        return f"💊 استخدامات **{name}**:\n{uses}"

    if intent == "dosage":
        dos = drug.get("dosage")

        if not dos or str(dos).strip().lower() in ("", "nan", "none"):
            return f"❌ لا تتوفر معلومات عن الجرعة لـ **{name}**"

        return f"📋 جرعة **{name}**:\n{dos}"

    return ""


def is_empty(value):
    if value is None:
        return True

    try:
        if np.isnan(value):
            return True
    except Exception:
        pass

    value = str(value).strip()
    return value == "" or value.lower() == "nan"


def build_search_text(drug):
    fields = []

    for key in [
        "name",
        "name_ar",
        "packaging",
        "uses",
        "side_effects",
        "dosage",
        "concentration",
        "category",
        "active_ingredient",
        "alternate_names",
        "synonyms",
    ]:
        value = drug.get(key)

        if is_empty(value):
            continue

        fields.append(str(value))

    return " ".join(fields).strip()


def log_failed_query(query):
    try:
        with open("failed_queries.log", "a", encoding="utf-8") as f:
            f.write(f"{datetime.datetime.utcnow().isoformat()}\t{query}\n")
    except Exception:
        pass


class Retriever:
    def __init__(self, drugs):
        self.drugs = drugs

        self.search_texts = [
            build_search_text(drug)
            for drug in drugs
        ]

        self.normalized_texts = [
            normalize_text(text)
            for text in self.search_texts
        ]

        self.drug_names_normalized = [
            normalize_text(drug.get("name") or "")
            for drug in drugs
        ]

        self.drug_names_ar_normalized = [
            normalize_text(drug.get("name_ar") or "")
            for drug in drugs
        ]

        if os.path.exists("embeddings.npy"):
            print("Loading saved embeddings...")
            self.embeddings = np.load("embeddings.npy")

            if len(self.embeddings) != len(self.search_texts):
                print("Embeddings size mismatch. Rebuilding...")
                self.embeddings = embed_texts(self.search_texts)
                np.save("embeddings.npy", self.embeddings)
        else:
            print("Creating embeddings...")
            self.embeddings = embed_texts(self.search_texts)
            np.save("embeddings.npy", self.embeddings)

        self.annoy_index = None

        try:
            if not os.path.exists("embeddings.ann"):
                print("Building Annoy index...")
                build_annoy_index(self.embeddings)

            self.annoy_index = load_annoy_index(self.embeddings.shape[1])

            if self.annoy_index is not None:
                print("Annoy index loaded")

        except Exception as exc:
            print(f"Annoy failed: {exc}")
            self.annoy_index = None

        self._embed_cache = {}

    def _get_query_embedding(self, query: str):
        if query not in self._embed_cache:
            if len(self._embed_cache) > 200:
                self._embed_cache.clear()

            self._embed_cache[query] = embed_query(query)

        return self._embed_cache[query]

    def search_by_name_contains(self, query, limit=50):
        q_norm = normalize_text(query)

        if not q_norm:
            return []

        results = []
        seen = set()

        for idx, drug in enumerate(self.drugs):
            name_norm = self.drug_names_normalized[idx]
            name_ar_norm = self.drug_names_ar_normalized[idx]

            if q_norm in name_norm or q_norm in name_ar_norm:
                key = normalize_text(drug.get("name") or drug.get("name_ar") or str(idx))

                if key not in seen:
                    results.append(drug)
                    seen.add(key)

                if len(results) >= limit:
                    break

        return results

    def search_top_k(self, query, k=5):
        q_norm = normalize_text(query)

        if not q_norm:
            return []

        results = []
        seen = set()

        # 1) exact name match
        for idx, name_norm in enumerate(self.drug_names_normalized):
            if q_norm == name_norm:
                return [(self.drugs[idx], 1.0)]

        # 2) direct substring match in name only
        for idx, name_norm in enumerate(self.drug_names_normalized):
            if q_norm in name_norm:
                results.append((self.drugs[idx], 0.95))
                seen.add(idx)

                if len(results) >= k:
                    return results

        # 3) direct substring match in Arabic name
        for idx, name_ar_norm in enumerate(self.drug_names_ar_normalized):
            if idx in seen:
                continue

            if q_norm in name_ar_norm:
                results.append((self.drugs[idx], 0.95))
                seen.add(idx)

                if len(results) >= k:
                    return results

        # 4) direct substring match in full search text
        for idx, text_norm in enumerate(self.normalized_texts):
            if idx in seen:
                continue

            if q_norm in text_norm:
                results.append((self.drugs[idx], 0.85))
                seen.add(idx)

                if len(results) >= k:
                    return results

        # 5) fuzzy match against drug names only
        if len(results) < k:
            fuzzy_scores = []

            for idx, name_norm in enumerate(self.drug_names_normalized):
                if idx in seen:
                    continue

                score = difflib.SequenceMatcher(
                    None,
                    q_norm,
                    name_norm
                ).ratio()

                if score >= 0.55:
                    fuzzy_scores.append((idx, score))

            fuzzy_scores.sort(key=lambda x: x[1], reverse=True)

            for idx, score in fuzzy_scores:
                results.append((self.drugs[idx], score))
                seen.add(idx)

                if len(results) >= k:
                    return results

        # 6) semantic search fallback
        if len(results) < k:
            try:
                q_emb = self._get_query_embedding(query)

                if self.annoy_index is not None:
                    idxs, dists = self.annoy_index.get_nns_by_vector(
                        q_emb.tolist(),
                        k * 2,
                        include_distances=True
                    )

                    for idx, dist in zip(idxs, dists):
                        idx = int(idx)

                        if idx in seen:
                            continue

                        score = max(0.0, 1.0 - float(dist))

                        if score < 0.55:
                            continue

                        results.append((self.drugs[idx], score))
                        seen.add(idx)

                        if len(results) >= k:
                            break

                else:
                    sims = np.dot(self.embeddings, q_emb)

                    for idx in np.argsort(-sims)[:k * 2]:
                        idx = int(idx)

                        if idx in seen:
                            continue

                        score = float(sims[idx])

                        if score < 0.55:
                            continue

                        results.append((self.drugs[idx], score))
                        seen.add(idx)

                        if len(results) >= k:
                            break

            except Exception as e:
                print("Semantic search error:", e)

        return results[:k]

    def search(self, query):
        intent = detect_intent(query)

        search_query = extract_drug_name(query) if intent else query

        if not search_query.strip():
            search_query = query

        # لو المستخدم كتب اسم دواء فقط بدون intent
        # رجّع كل الأنواع التي تحتوي على الاسم
        if not intent:
            name_matches = self.search_by_name_contains(search_query, limit=50)

            if name_matches:
                return name_matches if len(name_matches) > 1 else name_matches[0]

        top = self.search_top_k(search_query, k=5)

        if not top:
            log_failed_query(query)
            return None

        best_drug, best_score = top[0]

        print("QUERY =", search_query)
        for drug, score in top[:5]:
            print(
                drug.get("name"),
                "->",
                round(score, 3)
            )

        if best_score < 0.55:
            log_failed_query(query)
            return "WRONG_NAME"

        if intent:
            return {
                "__intent_response__": format_intent_response(
                    best_drug,
                    intent
                )
            }

        matched_drugs = []
        query_norm = normalize_text(search_query)

        for drug, score in top:
            name = normalize_text(drug.get("name", ""))
            name_ar = normalize_text(drug.get("name_ar", ""))

            if query_norm in name or query_norm in name_ar or score >= 0.90:
                matched_drugs.append(drug)

        unique = []
        seen = set()

        for drug in matched_drugs:
            key = normalize_text(drug.get("name") or drug.get("name_ar") or "")

            if key not in seen:
                unique.append(drug)
                seen.add(key)

        if len(unique) > 1:
            return unique

        return best_drug

    def contains_drug(self, query):
        intent = detect_intent(query)
        search_query = extract_drug_name(query) if intent else query

        if not search_query.strip():
            search_query = query

        top = self.search_top_k(search_query, k=1)

        if not top:
            return False

        _, score = top[0]

        return score >= 0.55