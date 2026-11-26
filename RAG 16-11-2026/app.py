import hashlib
import json
import math
import os
import re
import time
from datetime import datetime
from pathlib import Path

import chromadb
from fastembed import TextEmbedding
from flask import Flask, jsonify, render_template, request

from Assistent import RAGAssistant


# ---------------------------------------------------------------------
# Настройки приложения
# ---------------------------------------------------------------------

app = Flask(__name__)

DB_PATH = "./chroma_db"
COLLECTION_NAME = "knowledge_base"
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"

# Порог косинусного сходства (60%). Чанки с меньшим сходством отбрасываются
SOURCE_RELEVANCE_THRESHOLD = 0.60

BENCHMARK_QUESTIONS_FILE = Path("./data/Вопросы для Benchmark.txt")
BENCHMARK_REPORTS_DIR = Path("./benchmark_reports")

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

_embed_model = None


# ---------------------------------------------------------------------
# Инициализация RAG-ассистента и ChromaDB
# ---------------------------------------------------------------------

print("Инициализация RAG-ассистента...")

try:
    assistant = RAGAssistant(llm_model="qwen2.5:3b")
    print("RAG-ассистент загружен.")
except Exception as error:
    print(f"Ошибка загрузки ассистента: {error}")
    assistant = None


chroma_client = chromadb.PersistentClient(path=DB_PATH)

chunks_collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def get_embed_model():
    """Загружает модель эмбеддингов при первом обращении."""
    global _embed_model

    if _embed_model is None:
        print(f"Загрузка модели эмбеддингов: {EMBED_MODEL_NAME}")
        _embed_model = TextEmbedding(model_name=EMBED_MODEL_NAME)

    return _embed_model


def compute_cosine_metrics(vec1, vec2):
    """Вычисляет косинусное сходство, расстояние и точность в %."""
    dot = sum(float(a) * float(b) for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(float(a) * float(a) for a in vec1))
    norm2 = math.sqrt(sum(float(b) * float(b) for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return {"similarity": 0.0, "distance": 1.0, "accuracy": 0.0}

    sim = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    dist = max(0.0, 1.0 - sim)
    acc = max(0.0, sim) * 100.0

    return {
        "similarity": round(sim, 4),
        "distance": round(dist, 4),
        "accuracy": round(acc, 2),
    }


def filter_relevant_sources(question: str, sources: list, model, threshold: float = 0.60) -> list:
    """Фильтрует чанки, оставляя только те, которые действительно содержат нужную информацию."""
    if not sources:
        return []

    try:
        q_vec = list(model.embed([f"query: {question}"]))[0]
    except Exception:
        return sources

    filtered = []
    for src in sources:
        src_text = src.get("text", "") if isinstance(src, dict) else str(src)
        src_text_clean = src_text.strip()

        if not src_text_clean:
            continue

        try:
            s_vec = list(model.embed([f"passage: {src_text_clean}"]))[0]
            metrics = compute_cosine_metrics(q_vec, s_vec)

            # Если косинусное сходство чанка с вопросом >= порога, считаем его релевантным
            if metrics["similarity"] >= threshold:
                filtered.append(src)
        except Exception:
            filtered.append(src)

    return filtered


def is_no_info_answer(answer_str: str) -> bool:
    """Проверяет, ответила ли модель, что информации нет в базе."""
    low = answer_str.lower()
    keywords = [
        "нет информации", "не найдено", "не содержится", "не упоминается",
        "отсутствует", "не располагаю", "к сожалению", "не удалось найти",
        "база знаний не содержит"
    ]
    return any(kw in low for kw in keywords)


# ---------------------------------------------------------------------
# Страницы и API
# ---------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chunks")
def chunks_page():
    return render_template("chunks.html")


@app.route("/benchmark")
def benchmark_page():
    return render_template("benchmark.html")


@app.route("/api/ask", methods=["POST"])
def ask():
    if assistant is None:
        return jsonify({"error": "Ассистент не инициализирован."}), 500

    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()

    if not query:
        return jsonify({"error": "Пустой запрос."}), 400

    try:
        answer, sources = assistant.answer_question(query)
        return jsonify({"answer": answer, "sources": sources})
    except Exception:
        app.logger.exception("Ошибка обработки вопроса")
        return jsonify({"error": "Ошибка обработки вопроса."}), 500


@app.route("/api/clear", methods=["POST"])
def clear_history():
    if assistant is not None:
        assistant.clear_memory()

    return jsonify({"status": "ok", "message": "История диалога очищена."})


@app.route("/api/chunks/sources", methods=["GET"])
def get_sources():
    data = chunks_collection.get(include=["metadatas"])
    sources = sorted({
        metadata.get("source")
        for metadata in data.get("metadatas", [])
        if metadata and metadata.get("source")
    })
    return jsonify({"sources": sources})


@app.route("/api/chunks", methods=["GET"])
def get_chunks():
    search = request.args.get("search", "").strip().lower()
    strategy = request.args.get("strategy", "").strip()
    source = request.args.get("source", "").strip()

    try:
        page = max(int(request.args.get("page", 1)), 1)
        per_page = min(max(int(request.args.get("per_page", 20)), 1), 100)
    except ValueError:
        return jsonify({"error": "Некорректные параметры страницы."}), 400

    data = chunks_collection.get(include=["documents", "metadatas"])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    items = []

    for index, chunk_id in enumerate(data.get("ids", [])):
        text = documents[index] if index < len(documents) else ""
        metadata = metadatas[index] if index < len(metadatas) else {}
        metadata = metadata or {}

        item_strategy = metadata.get("strategy", "не указано")
        item_source = metadata.get("source", "неизвестный файл")

        if strategy and item_strategy != strategy:
            continue

        if source and item_source != source:
            continue

        if search and search not in f"{text} {item_source}".lower():
            continue

        items.append({
            "id": chunk_id,
            "text": text,
            "source": item_source,
            "strategy": item_strategy,
            "chunk_index": metadata.get("chunk_index", ""),
        })

    items.sort(key=lambda item: (
        item["source"],
        int(item["chunk_index"]) if str(item["chunk_index"]).isdigit() else 0,
    ))

    total = len(items)
    pages = (total + per_page - 1) // per_page
    start = (page - 1) * per_page

    return jsonify({
        "items": items[start:start + per_page],
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": pages,
    })


@app.route("/api/chunks/semantic-search", methods=["GET"])
def semantic_search_chunks():
    query = request.args.get("query", "").strip()
    strategy = request.args.get("strategy", "").strip()

    if not query:
        return jsonify({"error": "Введите поисковый запрос."}), 400

    try:
        top_k = min(max(int(request.args.get("top_k", 20)), 1), 50)
    except ValueError:
        return jsonify({"error": "Некорректное значение top_k."}), 400

    try:
        model = get_embed_model()
        query_vector = list(model.embed([f"query: {query}"]))[0].tolist()

        count = chunks_collection.count()
        if count == 0:
            return jsonify({"items": [], "total": 0})

        params = {
            "query_embeddings": [query_vector],
            "n_results": min(top_k, count),
            "include": ["documents", "metadatas", "distances"],
        }

        if strategy:
            params["where"] = {"strategy": strategy}

        result = chunks_collection.query(**params)
        documents = result.get("documents", [[]])[0] or []
        metadatas = result.get("metadatas", [[]])[0] or []
        distances = result.get("distances", [[]])[0] or []

        items = []
        for index, text in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            items.append({
                "id": "",
                "text": text,
                "source": metadata.get("source", "неизвестный файл"),
                "strategy": metadata.get("strategy", "не указано"),
                "chunk_index": metadata.get("chunk_index", ""),
                "distance": round(float(distances[index]), 5) if index < len(distances) else None,
            })

        return jsonify({"items": items, "total": len(items)})

    except Exception:
        app.logger.exception("Ошибка семантического поиска")
        return jsonify({"error": "Ошибка семантического поиска."}), 500


# ---------------------------------------------------------------------
# API загрузки файлов и индексации
# ---------------------------------------------------------------------

def chunk_by_fixed_window(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_by_paragraphs(text: str) -> list[str]:
    text_norm = text.replace("\r\n", "\n").replace("\r", "\n")
    return [p.strip() for p in re.split(r"\n\s*\n", text_norm) if p.strip()]


def delete_file_chunks(filename: str, strategy: str) -> None:
    data = chunks_collection.get(include=["metadatas"])
    ids_to_delete = []

    for chunk_id, metadata in zip(data.get("ids", []), data.get("metadatas", [])):
        if metadata and metadata.get("source") == filename and metadata.get("strategy") == strategy:
            ids_to_delete.append(chunk_id)

    if ids_to_delete:
        chunks_collection.delete(ids=ids_to_delete)


@app.route("/api/chunks/ingest", methods=["POST"])
def ingest_files():
    strategy = request.form.get("strategy", "").strip()
    files = request.files.getlist("files")

    if strategy not in {"fixed_window", "paragraphs"}:
        return jsonify({"error": "Не выбран допустимый способ разбиения."}), 400

    if not files or all(not file.filename for file in files):
        return jsonify({"error": "Выберите один или несколько TXT-файлов."}), 400

    try:
        model = get_embed_model()
    except Exception as error:
        return jsonify({"error": f"Не удалось загрузить модель эмбеддингов: {error}"}), 500

    processed_files = []
    errors = []
    total_chunks = 0

    for uploaded_file in files:
        filename = os.path.basename(uploaded_file.filename or "").strip()
        if not filename or not filename.lower().endswith(".txt"):
            continue

        try:
            raw_text = uploaded_file.read()
            text = raw_text.decode("utf-8-sig")

            if strategy == "fixed_window":
                chunks = chunk_by_fixed_window(text)
            else:
                chunks = chunk_by_paragraphs(text)

            if not chunks:
                continue

            delete_file_chunks(filename, strategy)

            prepared_chunks = [f"passage: {chunk}" for chunk in chunks]
            vectors = [v.tolist() for v in model.embed(prepared_chunks)]
            file_hash = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:12]

            ids = [f"{file_hash}_{strategy}_{i}" for i in range(len(chunks))]
            metadatas = [{"source": filename, "strategy": strategy, "chunk_index": i} for i in range(len(chunks))]

            chunks_collection.add(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas)
            processed_files.append({"filename": filename, "chunks": len(chunks)})
            total_chunks += len(chunks)

        except Exception as error:
            errors.append(f"{filename}: {error}")

    return jsonify({
        "message": "Обработка завершена.",
        "files": processed_files,
        "errors": errors,
        "total_chunks": total_chunks,
        "database_total": chunks_collection.count(),
    })


@app.route("/api/chunks/clear", methods=["POST"])
def clear_database():
    ids = chunks_collection.get(include=[]).get("ids", [])
    for start in range(0, len(ids), 500):
        chunks_collection.delete(ids=ids[start:start + 500])

    return jsonify({"message": "База чанков очищена.", "deleted": len(ids), "remaining": 0})


# ---------------------------------------------------------------------
# Бенчмарк с фильтрацией релевантных чанков
# ---------------------------------------------------------------------

@app.route("/api/benchmark/run", methods=["POST"])
def run_benchmark():
    if assistant is None:
        return jsonify({"error": "Ассистент не инициализирован."}), 500

    if not BENCHMARK_QUESTIONS_FILE.exists():
        return jsonify({"error": f"Файл вопросов не найден: {BENCHMARK_QUESTIONS_FILE.as_posix()}"}), 404

    try:
        model = get_embed_model()
    except Exception as error:
        return jsonify({"error": f"Ошибка загрузки модели эмбеддингов: {error}"}), 500

    try:
        raw_lines = [
            line.strip()
            for line in BENCHMARK_QUESTIONS_FILE.read_text(encoding="utf-8-sig").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        if not raw_lines:
            return jsonify({"error": "Файл вопросов пуст."}), 400

        results = []

        for number, line in enumerate(raw_lines, start=1):
            if "|" in line:
                parts = line.split("|", 1)
                question = parts[0].strip()
                reference = parts[1].strip()
            else:
                question = line.strip()
                reference = None

            if hasattr(assistant, "clear_memory"):
                assistant.clear_memory()

            started = time.perf_counter()

            try:
                answer, raw_sources = assistant.answer_question(question)
                elapsed = round(time.perf_counter() - started, 2)
                answer_str = str(answer).strip()

                # ФИЛЬТРАЦИЯ: оставляем только релевантные источники
                relevant_sources = filter_relevant_sources(
                    question, raw_sources, model, threshold=SOURCE_RELEVANCE_THRESHOLD
                )

                # Расчет точности по косинусному расстоянию
                if answer_str:
                    ans_vec = list(model.embed([f"passage: {answer_str}"]))[0]

                    if reference:
                        ref_vec = list(model.embed([f"passage: {reference}"]))[0]
                        metrics = compute_cosine_metrics(ans_vec, ref_vec)
                        eval_target = "эталонный ответ"
                    elif relevant_sources:
                        context_str = " ".join(
                            s.get("text", "") if isinstance(s, dict) else str(s) for s in relevant_sources
                        )
                        ctx_vec = list(model.embed([f"passage: {context_str}"]))[0]
                        metrics = compute_cosine_metrics(ans_vec, ctx_vec)
                        eval_target = "релевантный контекст"
                    else:
                        if is_no_info_answer(answer_str):
                            metrics = {"similarity": 1.0, "distance": 0.0, "accuracy": 100.0}
                            eval_target = "корректное отсутствие информации"
                        else:
                            q_vec = list(model.embed([f"query: {question}"]))[0]
                            metrics = compute_cosine_metrics(ans_vec, q_vec)
                            eval_target = "вопрос (без релевантных чанков)"
                else:
                    metrics = {"similarity": 0.0, "distance": 1.0, "accuracy": 0.0}
                    eval_target = "нет ответа"

                results.append({
                    "number": number,
                    "question": question,
                    "reference": reference,
                    "answer": answer_str,
                    "sources": relevant_sources,
                    "raw_sources_count": len(raw_sources or []),
                    "relevant_sources_count": len(relevant_sources),
                    "seconds": elapsed,
                    "cosine_distance": metrics["distance"],
                    "cosine_similarity": metrics["similarity"],
                    "accuracy": metrics["accuracy"],
                    "eval_target": eval_target,
                    "error": None,
                })

            except Exception as error:
                results.append({
                    "number": number,
                    "question": question,
                    "reference": reference,
                    "answer": "",
                    "sources": [],
                    "raw_sources_count": 0,
                    "relevant_sources_count": 0,
                    "seconds": round(time.perf_counter() - started, 2),
                    "cosine_distance": 1.0,
                    "cosine_similarity": 0.0,
                    "accuracy": 0.0,
                    "eval_target": "ошибка",
                    "error": str(error),
                })

        valid_items = [r for r in results if r["error"] is None]
        avg_accuracy = round(sum(r["accuracy"] for r in valid_items) / len(valid_items), 2) if valid_items else 0.0
        avg_distance = round(sum(r["cosine_distance"] for r in valid_items) / len(valid_items), 4) if valid_items else 1.0
        avg_seconds = round(sum(r["seconds"] for r in results) / len(results), 2) if results else 0.0

        return jsonify({
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model": getattr(assistant, "llm_model", "не указана"),
            "question_count": len(raw_lines),
            "average_accuracy": avg_accuracy,
            "average_distance": avg_distance,
            "average_seconds": avg_seconds,
            "results": results,
        })

    except Exception as error:
        app.logger.exception("Ошибка выполнения бенчмарка")
        return jsonify({"error": f"Ошибка выполнения бенчмарка: {error}"}), 500


@app.route("/api/benchmark/save", methods=["POST"])
def save_benchmark():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        return jsonify({"error": "Некорректные данные отчёта."}), 400

    BENCHMARK_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = BENCHMARK_REPORTS_DIR / f"benchmark_{timestamp}.json"

    try:
        report_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return jsonify({"message": "Отчёт сохранён.", "filename": report_path.name})
    except Exception:
        return jsonify({"error": "Не удалось сохранить отчёт."}), 500


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Размер запроса превышает допустимый лимит."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
