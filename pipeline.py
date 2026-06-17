from db import fetch_drugs
from retriever import Retriever
from llm import generate_response, general_chat


class RAGPipeline:
    def __init__(self):
        self.drugs = fetch_drugs()
        self.retriever = Retriever(self.drugs)

    def run(self, query: str):
        result = self.retriever.search(query)

        if result is None:
            return general_chat(query)

        if result == "WRONG_NAME":
            return "اسم الدواء غير صحيح أو غير موجود في قاعدة البيانات. جرّب تكتب الاسم كامل أو اختاره من الاقتراحات."

        # intent response (سعر / اثار جانبية / جرعة / استخدامات)
        if isinstance(result, dict) and "__intent_response__" in result:
            return result["__intent_response__"]

        if isinstance(result, list):
            return result

        if isinstance(result, dict):
            return generate_response(query, result)

        return general_chat(query)















# from db import fetch_drugs
# from retriever import Retriever
# from llm import generate_response, general_chat


# class RAGPipeline:
#     def __init__(self):
#         self.drugs = fetch_drugs()
#         self.retriever = Retriever(self.drugs)

#     def run(self, query: str):
#         result = self.retriever.search(query)

#         if result is None:
#             return general_chat(query)

#         if result == "WRONG_NAME":
#             return "اسم الدواء غير صحيح أو غير موجود في قاعدة البيانات. جرّب تكتب الاسم كامل أو اختاره من الاقتراحات."

#         if isinstance(result, list):
#             return result

#         if isinstance(result, dict):
#             return generate_response(query, result)

#         return general_chat(query)