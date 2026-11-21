import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 1. تحميل القطع النصية اللي جهزناها
with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]
print(f"جاري تحميل موديل الـ Embeddings وتحويل {len(texts)} قطعة لأرقام...")

# 2. تحميل موديل الـ Embedding (هينزل أول مرة بس وحجمه صغير ~90 ميجا)
model = SentenceTransformer("all-MiniLM-L6-v2")

# تحويل النصوص لـ Vectors مع عمل Normalization عشان نستخدم Cosine Similarity
embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
embeddings = np.array(embeddings).astype("float32")

# 3. بناء وحفظ قاعدة بيانات FAISS
dimension = embeddings.shape[1]  # حجم الـ Vector (384 رقم لكل قطعة)
index = faiss.IndexFlatIP(dimension)  # Inner Product مع Normalization = Cosine Similarity
index.add(embeddings)

faiss.write_index(index, "faiss_index.bin")
print("\nتم حفظ قاعدة البيانات بنجاح في ملف faiss_index.bin!")

# --- تجربة سريعة للتأكد إن البحث شغال ---
test_query = "What is the difference between RAG-Sequence and RAG-Token?"
query_vec = model.encode([test_query], normalize_embeddings=True).astype("float32")

# جلب أقرب 3 قطع للسؤال
scores, indices = index.search(query_vec, k=3)

print(f"\nنتيجة البحث التجريبي عن: '{test_query}'")
for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
    item = chunks[idx]
    print(f"\n[{rank}] درجة التشابه: {score:.3f} | المصدر: {item['source']} (صفحة {item['page']})")
    print(f"النص: {item['text'][:200]}...")