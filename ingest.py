import os
import re
import json
import fitz  # مكتبة PyMuPDF

def clean_text(text: str) -> str:
    # دمج الكلمات المقسومة بشرطة في نهاية السطر
    text = re.sub(r'-\n+', '', text)
    # تحويل السطور الجديدة لمسافة واحدة
    text = re.sub(r'\n+', ' ', text)
    # إزالة المسافات الزيادة
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_pages_from_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    pages_data = []

    for page_idx, page in enumerate(doc):
        page_height = page.rect.height
        blocks = page.get_text("blocks")
        
        valid_texts = []
        for b in blocks:
            x0, y0, x1, y1, text, block_no, block_type = b
            # استخراج النصوص فقط وتجاهل الهوامش العلوية والسفلية المتكررة
            if block_type == 0 and y0 > 45 and y1 < (page_height - 45):
                cleaned = clean_text(text)
                if cleaned:
                    valid_texts.append(cleaned)
        
        full_page_text = " ".join(valid_texts)
        if full_page_text:
            pages_data.append({
                "source": filename,
                "page": page_idx + 1,
                "text": full_page_text
            })
            
    return pages_data

def chunk_documents(pages_data: list[dict], chunk_size: int = 300, overlap: int = 50) -> list[dict]:
    chunks = []
    chunk_counter = 0

    for page_item in pages_data:
        words = page_item["text"].split()
        if not words:
            continue
            
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if len(chunk_words) < 25 and i > 0:
                continue
                
            chunks.append({
                "chunk_id": f"{page_item['source']}_p{page_item['page']}_c{chunk_counter}",
                "source": page_item["source"],
                "page": page_item["page"],
                "text": chunk_text
            })
            chunk_counter += 1
            
    return chunks

if __name__ == "__main__":
    all_pages = []
    for file in os.listdir("papers"):
        if file.lower().endswith(".pdf"):
            print(f"جاري قراءة: {file}...")
            all_pages.extend(extract_pages_from_pdf(os.path.join("papers", file)))
            
    chunks = chunk_documents(all_pages, chunk_size=300, overlap=50)
    
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
        
    print(f"\nتم بنجاح! استخرجنا {len(all_pages)} صفحة وقسمناهم لـ {len(chunks)} قطعة في ملف chunks.json")