# rag_app/rag_engine.py
import os
import pickle
import requests
import numpy as np
import faiss
from django.conf import settings

EMBEDDING_URL = "http://localhost:8081/embedding"
LLM_URL = "http://localhost:8080/completion"  # или /v1/chat/completions
EMBEDDING_DIM = 1024  # для bge-m3

INDEX_PATH = os.path.join(settings.BASE_DIR, "faiss_index.bin")
CHUNKS_PATH = os.path.join(settings.BASE_DIR, "chunks.pkl")

def embed_text(text: str) -> np.ndarray:
    response = requests.post(EMBEDDING_URL, json={"content": text})
    response.raise_for_status()
    return np.array(response.json()['embedding'], dtype=np.float32)

def embed_batch(texts):
    # Можно отправлять по одному, но для скорости – используйте батчинг,
    # если ваш сервер поддерживает. Пока оставим цикл.
    return np.vstack([embed_text(t) for t in texts])

def load_index():
    """Загружает FAISS индекс и список чанков."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return None, None
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, 'rb') as f:
        chunks = pickle.load(f)
    return index, chunks

def save_index(index, chunks):
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNGS_PATH, 'wb') as f:
        pickle.dump(chunks, f)

def build_index_from_documents(doc_texts):
    """Создаёт индекс из списка текстов (чанков)."""
    vectors = embed_batch(doc_texts)
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(vectors)
    save_index(index, doc_texts)
    return index, doc_texts

def retrieve(query: str, top_k=3):
    """Возвращает список релевантных чанков."""
    index, chunks = load_index()
    if index is None:
        return []
    q_vec = embed_text(query).reshape(1, -1)
    distances, indices = index.search(q_vec, top_k)
    results = [chunks[i] for i in indices[0] if i < len(chunks)]
    return results

def generate_answer(prompt: str) -> str:
    payload = {
        "prompt": prompt,
        "temperature": 0.3,
        "max_tokens": 512,
        "stop": ["\n\n"]
    }
    response = requests.post(LLM_URL, json=payload)
    response.raise_for_status()
    return response.json()['content']

def rag_query(question: str) -> str:
    context_chunks = retrieve(question, top_k=3)
    context = "\n\n".join(context_chunks)
    system_prompt = "Ты — полезный ассистент студента. Отвечай на вопросы, используя только предоставленный контекст. Если ответа нет в контексте, скажи об этом честно."
    user_prompt = f"Контекст:\n{context}\n\nВопрос: {question}\nОтвет:"
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    return generate_answer(full_prompt)

def add_chunks_to_index(new_chunks: list):
    """Добавляет новые чанки в существующий FAISS индекс и сохраняет обновлённый индекс."""
    index, old_chunks = load_index()
    if index is None:
        # Если индекса ещё нет — создаём новый
        build_index_from_documents(new_chunks)
        return

    # Векторизуем новые чанки
    new_vectors = embed_batch(new_chunks)
    # Добавляем в существующий индекс
    index.add(new_vectors)
    # Обновляем список всех чанков
    all_chunks = old_chunks + new_chunks
    # Сохраняем
    save_index(index, all_chunks)