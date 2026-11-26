import json
import urllib.error
import urllib.request
import chromadb
from sentence_transformers import SentenceTransformer


class RAGAssistant:

    def __init__(self, llm_model: str = "qwen2.5:3b", history_limit: int = 0):
        self.llm_model = llm_model
        self.ollama_url = "http://127.0.0.1:11434"

        # История диалога отключена — каждый запрос обрабатывается независимо
        self.history_limit = history_limit
        self.chat_history = []

        # 1. Подключаемся к базе ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        try:
            self.collection = self.chroma_client.get_collection(
                name="knowledge_base"
            )
        except Exception:
            raise RuntimeError(
                "База данных не найдена или пуста! Сначала загрузите файлы через веб-интерфейс или скрипты индексации."
            )

        # 2. Проверяем доступность сервера Ollama и наличие модели
        self._check_ollama_status()

        # 3. Инициализируем локальную модель эмбеддингов SentenceTransformers
        print("Загрузка поисковой модели SentenceTransformers (BAAI/bge-m3)...")
        self.embed_model = SentenceTransformer("BAAI/bge-m3")

    def _check_ollama_status(self):
        """Проверяет, запущен ли сервер Ollama и скачана ли нужная модель."""
        try:
            req = urllib.request.Request(f"{self.ollama_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                installed_models = [
                    m.get("name") for m in data.get("models", [])
                ]

                has_model = any(
                    self.llm_model in m for m in installed_models
                )
                if not has_model:
                    print(
                        f"\n[ВНИМАНИЕ]: Модель '{self.llm_model}' не найдена в Ollama!"
                    )
                    print(
                        f"Скачайте её командой в терминале: ollama pull {self.llm_model}\n"
                    )
        except Exception:
            raise RuntimeError(
                "Сервер Ollama не запущен! Убедитесь, что приложение Ollama запущено."
            )

    def retrieve_relevant_chunks(
        self, query: str, similarity_threshold: float = 0.50, max_candidates: int = 10
    ) -> list[str]:
        """Динамически находит все чанки в базе, удовлетворяющие порогу косинусного сходства.

        Количество результатов не ограничено жестко 3-мя.
        """
        try:
            # Префикс 'query: ' рекомендуется для моделей семейства BGE-M3
            formatted_query = f"query: {query}" if not query.startswith("query: ") else query
            query_vector = self.embed_model.encode(formatted_query).tolist()

            count = self.collection.count()
            if count == 0:
                return []

            n_results = min(max_candidates, count)
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                include=["documents", "distances"]
            )

            if not results or not results.get("documents") or not results["documents"][0]:
                return []

            docs = results["documents"][0]
            distances = results.get("distances", [[]])[0]

            relevant_chunks = []
            for doc, dist in zip(docs, distances):
                # В ChromaDB при метрике cosine: similarity = 1.0 - distance
                similarity = 1.0 - float(dist)
                if similarity >= similarity_threshold:
                    relevant_chunks.append(doc)

            return relevant_chunks
        except Exception as e:
            print(f"[Ошибка при поиске в векторной базе]: {e}")
            return []

    def clear_memory(self):
        """Очищает историю диалога."""
        self.chat_history = []

    def _generate_llm_response(self, messages: list) -> str:
        """Прямой HTTP-запрос к API Ollama для генерации ответа."""
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "stream": False,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.ollama_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 503:
                return "Ошибка 503: Сервер Ollama перегружен или не смог запустить модель."
            return f"Ошибка HTTP от Ollama ({e.code}): {e.reason}"
        except Exception as e:
            return f"Ошибка соединения с Ollama: {e}"

    def answer_question(self, query: str) -> tuple[str, list[str]]:
        """Формирует ответ строго на основе динамически найденного контекста без накопительной истории."""
        # 1. Поиск релевантных чанков по порогу косинусного сходства (динамическое количество)
        context_chunks = self.retrieve_relevant_chunks(query, similarity_threshold=0.50, max_candidates=3)

        # Склеиваем найденные чанки в один блок контекста
        context = "\n---\n".join(context_chunks) if context_chunks else "Документы отсутствуют."

        # 2. Системный промпт (строго по контексту)
        system_prompt = (
            "Ты — полезный локальный ИИ-ассистент.\n"
            "Отвечай на вопрос пользователя строго на основе предоставленного ниже Контекста.\n"
            "Если ответа нет в контексте, напиши: 'В предоставленных документах нет информации по данному вопросу.'\n\n"
            f"КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:\n{context}"
        )

        # 3. Формируем сообщения без передачи старой истории бесед
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        # 4. Генерируем ответ через Ollama
        answer_text = self._generate_llm_response(messages)

        return answer_text, context_chunks


def main():
    try:
        assistant = RAGAssistant(llm_model="qwen2.5:3b")
    except RuntimeError as err:
        print(f"\n[ОШИБКА ЗАПУСКА]: {err}")
        return

    print("=" * 60)
    print("Локальный RAG-ассистент готов к работе!")
    print("Управляющие команды:")
    print("  'exit' или 'выход' — завершить программу")
    print("  'clear' или 'очистить' — сбросить состояние")
    print("=" * 60)

    while True:
        query = input("\nВы: ").strip()

        if not query:
            continue

        if query.lower() in ["exit", "quit", "выход"]:
            print("До свидания!")
            break

        if query.lower() in ["clear", "очистить", "сброс"]:
            assistant.clear_memory()
            print("[Состояние очищено.]")
            continue

        print("[Поиск в базе и генерация ответа...]")
        answer, sources = assistant.answer_question(query)

        print("\n--- НАЙДЕННЫЕ ИСТОЧНИКИ (НА ОСНОВЕ СХОДСТВА) ---")
        if sources:
            for i, chunk in enumerate(sources, 1):
                preview = (
                    chunk[:100].replace("\n", " ") + "..."
                    if len(chunk) > 100
                    else chunk
                )
                print(f"[{i}] {preview}")
        else:
            print("[Релевантные источники в базе не найдены]")

        print("\n--- ОТВЕТ АССИСТЕНТА ---")
        print(answer)
        print("=" * 60)


if __name__ == "__main__":
    main()