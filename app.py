import os
import pandas as pd
import streamlit as st
from rag_pipeline import generate_answer

st.set_page_config(page_title="Academic Papers RAG System", page_icon="📚", layout="wide")

st.title("📚 نظام الإجابة الموثقة من الأوراق البحثية (RAG System)")
st.markdown("اسأل أي سؤال في الأوراق البحثية المفهرسة، وسيقوم النظام باسترجاع الفقرات الداعمة وصياغة الإجابة مع التوثيق برقم الصفحة.")

# شريط جانبي يعرض الأوراق المفهرسة وإعدادات البحث
with st.sidebar:
    st.header("⚙️ إعدادات النظام")
    top_k = st.slider("عدد الفقرات المسترجعة (Top-K):", min_value=2, max_value=6, value=4)
    
    st.divider()
    st.subheader("📄 الأوراق البحثية المتاحة (7)")
    if os.path.exists("papers"):
        for pdf in sorted(os.listdir("papers")):
            if pdf.lower().endswith(".pdf"):
                st.caption(f"• {pdf}")

# تقسيم الواجهة إلى تبويبين: التجربة الحية + نتائج التقييم
tab1, tab2 = st.tabs(["💬 اسأل الأبحاث (Live Demo)", "📊 جداول التقييم (Evaluation Results)"])

with tab1:
    sample_questions = [
        "اختر سؤالاً جاهزاً للتجربة أو اكتب سؤالك بالأسفل...",
        "What is the main difference between RAG-Sequence and RAG-Token models?",
        "How does LoRA reduce the number of trainable parameters during fine-tuning?",
        "What U-shaped performance curve is observed in the Lost in the Middle paper?",
        "What is the population of Alexandria, Egypt in 2026?"
    ]
    selected_sample = st.selectbox("أمثلة سريعة:", sample_questions)
    
    default_q = "" if selected_sample.startswith("اختر") else selected_sample
    query = st.text_input("اكتب سؤالك هنا:", value=default_q, placeholder="مثال: How does Self-RAG critique its own output?")

    if st.button("بحث وتوليد الإجابة", type="primary") and query.strip():
        with st.spinner("جاري البحث في الأوراق وصياغة الإجابة الموثقة..."):
            result = generate_answer(query, top_k=top_k)
            answer = result["answer"]
            retrieved = result["retrieved"]

        st.subheader("💡 الإجابة:")
        if "insufficient evidence" in answer.lower():
            st.warning(f"⚠️ {answer}")
        else:
            st.success(answer)

        st.subheader("🔍 الفقرات المسترجعة (Evidence Passages):")
        for i, chunk in enumerate(retrieved, start=1):
            with st.expander(f"الفقرة {i} | المصدر: {chunk['source']} — صفحة {chunk['page']} (درجة التشابه: {chunk['score']:.3f})"):
                st.write(chunk["text"])

with tab2:
    st.subheader("1. مقارنة إعدادات التقطيع (Validation Comparison)")
    if os.path.exists("validation_comparison.csv"):
        st.dataframe(pd.read_csv("validation_comparison.csv"), use_container_width=True)
    else:
        st.info("شغّل ملف evaluate.py لعرض جدول المقارنة.")

    st.subheader("2. نتائج الاختبار النهائي (Held-out Test)")
    if os.path.exists("test_results_summary.csv"):
        st.dataframe(pd.read_csv("test_results_summary.csv"), use_container_width=True)

    st.subheader("3. تحليل حالات الفشل (Error Analysis)")
    if os.path.exists("error_analysis.csv"):
        err_df = pd.read_csv("error_analysis.csv")
        if len(err_df) > 0:
            st.dataframe(err_df, use_container_width=True)
        else:
            st.success("لا توجد حالات فشل في الاختبار النهائي!")