# app.py
import os
import gradio as gr
from openai import OpenAI

# ============ ЛОКАЛЬНЫЕ МОДЕЛИ ============
# llama.cpp отдаёт OpenAI-совместимый API. Ключ не нужен, но клиент требует непустой.
LLM_BASE = "http://127.0.0.1:8080/v1"
EMB_BASE = "http://127.0.0.1:8081/v1"
API_KEY  = "sk-no-key-required"

llm = OpenAI(base_url=LLM_BASE, api_key=API_KEY)
emb = OpenAI(base_url=EMB_BASE, api_key=API_KEY)

# Узнать: curl http://127.0.0.1:8080/v1/models
LLM_MODEL = "qwen3"      
EMB_MODEL = "bge-m3"

SYSTEM_PROMPT = (
    "Ты полезный ассистент студента. Отвечай по-русски, кратко и по делу."
    "Если ниже дан контекст из базы знаний — опирайся только на него."
    # "Отвечай 'не знаю' если в базе знаний нет ответа"
)

# ============ ЭМБЕДДИНГИ (уже рабочие) ============
def embed(texts: list[str]) -> list[list[float]]:
    r = emb.embeddings.create(model=EMB_MODEL, input=texts)
    return [d.embedding for d in r.data]

# ============ ЗАГЛУШКА RAG ============
def retrieve(query: str, k: int = 4) -> list[dict]:
    """
    Позже: SELECT top-k из pgvector по embed([query])[0].
    Пока пусто — отвечаем без RAG.
    """
    return []

def build_context(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    lines = ["Контекст из базы знаний:"]
    for i, c in enumerate(chunks, 1):
        lines.append(f"[{i}] ({c.get('source', '?')}) {c['text']}")
    return "\n".join(lines)

# ============ ОТВЕТ (стриминг, вся история разговора сохраняется) ============
def respond(message: str, history: list):
    chunks = retrieve(message, k=4)
    context = build_context(chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})

    user_content = f"{context}\n\nВопрос: {message}" if context else message
    messages.append({"role": "user", "content": user_content})

    stream = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
        temperature=0.2,
    )
    partial = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        partial += delta
        yield partial

# ============ ИНГЕСТ (заглушка, но эмбеддинги уже считает) ============
def upload_files(files, collection: str):
    if not files:
        return "Файлы не выбраны."
    names = [os.path.basename(f.name) for f in files]
    # демонстрация: считаем эмбеддинг первого куска, чтобы проверить связку
    try:
        v = embed(["тест"])[0]
        dim = len(v)
        emb_info = f"эмбеддинг-сервер отвечает, размерность = {dim}"
    except Exception as e:
        emb_info = f"ОШИБКА эмбеддинг-сервера: {e}"
    return (
        f"Получено {len(files)} файл(ов) в коллекцию «{collection}»: "
        + ", ".join(names)
        + f"\n\n{emb_info}\n\nИндексация (chunk → embedding → pgvector) — следующий шаг."
    )

# ============ UI ============
with gr.Blocks(title="RAG-агент (локальный)") as demo:
    gr.Markdown("# RAG-агент \"Ассистент студента МГТУ им. Носова\"")

    with gr.Tab("Чат"):
        gr.ChatInterface(respond)

    with gr.Tab("Загрузка документов"):
        files = gr.File(file_count="multiple", label="PDF / DOCX / TXT / MD")
        collection = gr.Textbox(value="default", label="Коллекция")
        btn = gr.Button("Загрузить", variant="primary")
        status = gr.Markdown()
        btn.click(upload_files, inputs=[files, collection], outputs=status)

if __name__ == "__main__":
    demo.launch()