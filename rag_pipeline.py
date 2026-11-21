import os
# توجيه الكاش لبارتيشن D عشان يستخدم الموديل اللي حملناه
os.environ["HF_HOME"] = r"D:\rag_project\hf_cache"

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# 1. إعداد مفتاح Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# 2. تحميل البيانات وقاعدة بيانات البحث وموديل الـ Embeddings
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

index = faiss.read_index("faiss_index.bin")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve(query: str, top_k: int = 4) -> list[dict]:
    query_vec = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = index.search(query_vec, k=top_k)
    
    results = []
    for idx, score in zip(indices[0], scores[0]):
        chunk = chunks[idx].copy()
        chunk["score"] = float(score)
        results.append(chunk)
    return results

def generate_answer(query: str, top_k: int = 4) -> dict:
    # استرجاع أقرب الفقرات
    retrieved_chunks = retrieve(query, top_k=top_k)
    
    # تجهيز السياق مع اسم الملف ورقم الصفحة لكل فقرة
    context_parts = []
    for i, c in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[Passage {i} | Source: {c['source']} | Page: {c['page']}]\n{c['text']}"
        )
    context_str = "\n\n".join(context_parts)
    
    # توجيه صارم للنموذج للالتزام بالمصادر والتوثيق أو الامتناع عن الإجابة
    system_prompt = """You are a strict research assistant. Answer the user's question using ONLY the provided context passages.
Rules:
1. Every factual claim in your answer MUST include an inline citation in the exact format: [Source: filename, Page: X].
2. Do NOT use any outside knowledge or assume facts not explicitly stated in the passages.
3. If the provided passages do not contain enough information to answer the question, you MUST respond ONLY with:
"Insufficient evidence in the provided documents to answer this question."
"""

    user_prompt = f"Context Passages:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"

    available_models = [
        m.id for m in client.models.list().data
        if not any(w in m.id.lower() for w in ["whisper", "guard", "tts", "orpheus"])
    ]
    chosen_model = next((m for m in available_models if "70b" in m or "120b" in m or "llama-4" in m), available_models[0])

    response = client.chat.completions.create(
        model=chosen_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0  # صفر لمنع التأليف والهلوسة تماماً
    )
    
    answer = response.choices[0].message.content.strip()
    return {
        "question": query,
        "answer": answer,
        "retrieved": retrieved_chunks
    }

# --- تجربة حقيقية على النظام ---
if __name__ == "__main__":
    # تجربة 1: سؤال إجابته موجودة في الأوراق البحثية
    q1 = "What is the main difference between RAG-Sequence and RAG-Token models?"
    res1 = generate_answer(q1)
    print(f"\nQ1: {res1['question']}")
    print(f"Answer:\n{res1['answer']}")
    print("-" * 60)

    # تجربة 2: سؤال ملوش إجابة في الأوراق (لاختبار الامتناع عن التأليف)
    q2 = "What is the population of Alexandria, Egypt in 2026?"
    res2 = generate_answer(q2)
    print(f"\nQ2: {res2['question']}")
    print(f"Answer:\n{res2['answer']}")