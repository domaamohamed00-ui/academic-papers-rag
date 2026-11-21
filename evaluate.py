import os
os.environ["HF_HOME"] = r"D:\rag_project\hf_cache"

import json
import time
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

from ingest import extract_pages_from_pdf, chunk_documents
from rag_pipeline import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# اختيار الموديل المتاح تلقائياً
available_models = [
    m.id for m in client.models.list().data
    if not any(w in m.id.lower() for w in ["whisper", "guard", "tts", "orpheus"])
]
CHOSEN_MODEL = next((m for m in available_models if "70b" in m or "120b" in m or "llama-4" in m), available_models[0])
print(f"Using Groq Model: {CHOSEN_MODEL}")

def call_llm_safe(messages, temperature=0.0):
    """استدعاء الـ LLM مع انتظار بسيط لتجنب ضغط الطلبات"""
    for attempt in range(3):
        try:
            time.sleep(2.2)
            resp = client.chat.completions.create(
                model=CHOSEN_MODEL,
                messages=messages,
                temperature=temperature
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"تنبيه مؤقت ({e})، جاري المحاولة مرة أخرى...")
            time.sleep(15)
    return "ERROR"

def build_temp_index(all_pages, chunk_size, overlap=50):
    chunks = chunk_documents(all_pages, chunk_size=chunk_size, overlap=overlap)
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, show_progress_bar=False, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return chunks, index

def judge_answer(question, reference_answer, model_answer, context_str):
    """تقييم دقة الإجابة والهلوسة (Unsupported Claims)"""
    judge_prompt = f"""Evaluate the generated answer against the reference answer and provided context.
Return ONLY a valid JSON object with two boolean keys:
- "correct": true if the generated answer accurately captures the key facts in the reference answer, false otherwise.
- "unsupported_claim": true if the generated answer states facts NOT supported by the context, false otherwise.

Question: {question}
Reference Answer: {reference_answer}
Context: {context_str}
Generated Answer: {model_answer}

JSON:"""
    raw = call_llm_safe([{"role": "user", "content": judge_prompt}])
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        return bool(data.get("correct", False)), bool(data.get("unsupported_claim", False))
    except Exception:
        # تقييم احتياطي لو الرد مكنش JSON صريح
        is_correct = any(w.lower() in model_answer.lower() for w in reference_answer.split()[:4])
        return is_correct, False

def evaluate_split(questions, chunks, index, config_name, top_k=4):
    records = []
    system_prompt = """You are a strict research assistant. Answer the user's question using ONLY the provided context passages.
Rules:
1. Every factual claim in your answer MUST include an inline citation in the exact format:.
2. Do NOT use any outside knowledge or assume facts not explicitly stated in the passages.
3. If the provided passages do not contain enough information to answer the question, you MUST respond ONLY with:
"Insufficient evidence in the provided documents to answer this question."
"""

    for q in questions:
        query_vec = embed_model.encode([q["question"]], normalize_embeddings=True).astype("float32")
        scores, indices = index.search(query_vec, k=top_k)
        retrieved = [chunks[idx] for idx in indices[0]]
        retrieved_sources = [c["source"] for c in retrieved]

        context_str = "\n\n".join(
            [f"[Passage {i+1} | Source: {c['source']} | Page: {c['page']}]\n{c['text']}" for i, c in enumerate(retrieved)]
        )
        user_prompt = f"Context Passages:\n{context_str}\n\nQuestion: {q['question']}\n\nAnswer:"
        answer = call_llm_safe([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        abstained = "insufficient evidence" in answer.lower()

        if q["answerable"]:
            recall_hit = 1 if q["expected_source"] in retrieved_sources else 0
            citation_correct = 1 if (q["expected_source"] in answer and not abstained) else 0
            ans_correct, unsupported = judge_answer(q["question"], q["reference_answer"], answer, context_str)
            abstention_ok = None
        else:
            recall_hit = None
            citation_correct = None
            ans_correct = 1 if abstained else 0
            unsupported = 0 if abstained else 1
            abstention_ok = 1 if abstained else 0

        records.append({
            "config": config_name,
            "id": q["id"],
            "split": q["split"],
            "answerable": q["answerable"],
            "question": q["question"],
            "expected_source": q["expected_source"],
            "retrieved_sources": ", ".join(set(retrieved_sources)),
            "recall_at_4": recall_hit,
            "answer_correct": int(ans_correct),
            "citation_correct": citation_correct,
            "unsupported_claim": int(unsupported),
            "abstention_correct": abstention_ok,
            "model_answer": answer
        })
    return pd.DataFrame(records)

def summarize_metrics(df, label):
    ans_df = df[df["answerable"] == True]
    unans_df = df[df["answerable"] == False]
    return {
        "Configuration": label,
        "Recall@4": round(ans_df["recall_at_4"].mean(), 3),
        "Answer_Correctness": round(df["answer_correct"].mean(), 3),
        "Citation_Correctness": round(ans_df["citation_correct"].mean(), 3),
        "Abstention_Accuracy": round(unans_df["abstention_correct"].mean(), 3),
        "Unsupported_Claim_Rate": round(df["unsupported_claim"].mean(), 3)
    }

if __name__ == "__main__":
    with open("eval_questions.json", "r", encoding="utf-8") as f:
        all_questions = json.load(f)

    val_q = [q for q in all_questions if q["split"] == "val"]
    test_q = [q for q in all_questions if q["split"] == "test"]

    print("1. جاري قراءة الأوراق البحثية للمقارنة...")
    all_pages = []
    for file in os.listdir("papers"):
        if file.lower().endswith(".pdf"):
            all_pages.extend(extract_pages_from_pdf(os.path.join("papers", file)))

    # تجربة إعدادين مختلفين على بيانات التحقق (Validation)
    configs = [250, 500]
    val_summaries = []
    val_dfs = {}

    for c_size in configs:
        print(f"\n2. تقييم إعداد chunk_size={c_size} على أسئلة Validation...")
        chunks_tmp, index_tmp = build_temp_index(all_pages, chunk_size=c_size, overlap=50)
        df_val = evaluate_split(val_q, chunks_tmp, index_tmp, f"chunk_{c_size}")
        val_dfs[c_size] = (df_val, chunks_tmp, index_tmp)
        val_summaries.append(summarize_metrics(df_val, f"chunk_size={c_size}"))

    val_table = pd.DataFrame(val_summaries)
    val_table.to_csv("validation_comparison.csv", index=False)
    print("\n=== جدول مقارنة Validation ===")
    print(val_table.to_string(index=False))

    # اختيار أفضل إعداد بناءً على مجموع Recall@4 ودقة الإجابة
    best_idx = (val_table["Recall@4"] + val_table["Answer_Correctness"]).idxmax()
    best_chunk_size = configs[best_idx]
    print(f"\n>>> أفضل إعداد تم اختياره: chunk_size={best_chunk_size}")

    # التقييم النهائي على أسئلة Held-out Test
    print(f"\n3. جاري التقييم النهائي على أسئلة Held-out Test باستخدام chunk_size={best_chunk_size}...")
    _, best_chunks, best_index = val_dfs[best_chunk_size]
    df_test = evaluate_split(test_q, best_chunks, best_index, f"best_chunk_{best_chunk_size}")

    test_table = pd.DataFrame([summarize_metrics(df_test, f"Held-out Test (chunk={best_chunk_size})")])
    test_table.to_csv("test_results_summary.csv", index=False)
    print("\n=== جدول نتائج الاختبار النهائي (Held-out Test) ===")
    print(test_table.to_string(index=False))

    # حفظ كل التفاصيل وحالات الفشل (Error Analysis)
    full_df = pd.concat([val_dfs[250][0], val_dfs[500][0], df_test], ignore_index=True)
    full_df.to_csv("detailed_predictions.csv", index=False)

    failures = df_test[(df_test["answer_correct"] == 0) | (df_test["recall_at_4"] == 0) | (df_test["unsupported_claim"] == 1)]
    failures.to_csv("error_analysis.csv", index=False)
    print(f"\nتم حفظ الجداول بنجاح! عدد حالات الفشل في الاختبار النهائي: {len(failures)} (محفوظة في error_analysis.csv)")