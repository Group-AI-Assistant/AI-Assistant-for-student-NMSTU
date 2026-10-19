import json
import urllib.error
import urllib.request
import chromadb
from fastembed import TextEmbedding


class RAGAssistant:

    def __init__(self, llm_model: str = "qwen2.5:3b", history_limit: int = 6):
        self.llm_model = llm_model
        self.ollama_url = "http://127.0.0.1:11434"
        
        # Лимит реплик в памяти (6 реплик = 3 ваших вопроса + 3 ответа ИИ)
        self.history_limit = history_limit
        self.chat_history = []  # Хранилище для истории диалога

        # 1. Подключаемся к базе ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        try:
            self.collection = self.chroma_client.get_collection(
                name="knowledge_base"
            )
        except Exception:
            raise RuntimeError(
                "База данных не найдена или пуста! Сначала запустите файл 1_ignest.py или 2_ignest.py."
            )

        # 2. Проверяем доступность сервера Ollama и наличие модели
        self._check_ollama_status()

        # 3. Инициализируем локальную модель эмбеддингов FastEmbed
        print("Загрузка поисковой модели FastEmbed...")
        self.embed_model = TextEmbedding(model_name="intfloat/multilingual-e5-large")

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
        self, query: str, top_k: int = 3
    ) -> list[str]:
        """Векторизует вопрос локально через FastEmbed и ищет top_k самых похожих чанков в базе."""
        try:
            query_vector = list(self.embed_model.embed([query]))[0].tolist()
            results = self.collection.query(
                query_embeddings=[query_vector], n_results=top_k
            )
            if results and results.get("documents") and results["documents"][0]:
                return results["documents"][0]
            return []
        except Exception as e:
            print(f"[Ошибка при поиске в векторной базе]: {e}")
            return []

    def clear_memory(self):
        """Очищает историю диалога."""
        self.chat_history = []
        print("\n[Память ассистента успешно очищена! Начните новую тему.]")

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
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 503:
                return "Ошибка 503: Сервер Ollama перегружен или не смог запустить модель."
            return f"Ошибка HTTP от Ollama ({e.code}): {e.reason}"
        except Exception as e:
            return f"Ошибка соединения с Ollama: {e}"

    def answer_question(self, query: str) -> tuple[str, list[str]]:
        """Формирует ответ с учетом найденного контекста и истории диалога."""
        # 1. Поиск релевантных чанков по текущему вопросу пользователя
        context_chunks = self.retrieve_relevant_chunks(query, top_k=3)

        # Склеиваем найденные чанки в один блок контекста
        context = "\n---\n".join(context_chunks) if context_chunks else "Документы отсутствуют."

        # 2. Системный промпт (всегда идет первым сообщением для ИИ)
        system_prompt = (
            "Ты — полезный локальный ИИ-ассистент.\n"
            "Отвечай на вопрос пользователя строго на основе предоставленного ниже Контекста.\n"
            "Если ответа нет в контексте, напиши: 'В предоставленных документах нет информации по данному вопросу.'\n"
            "При ведении диалога учитывай предыдущую историю беседы.\n\n"
            f"КОНТЕКСТ ИЗ БАЗЫ ЗНАНИЙ:\n{context}"
        )

        # 3. Собираем структуру сообщений для модели
        messages = [{"role": "system", "content": system_prompt}]
        
        # Добавляем в запрос накопленную историю диалога
        messages.extend(self.chat_history)
        
        # Добавляем текущий вопрос пользователя
        messages.append({"role": "user", "content": query})

        # 4. Генерируем ответ через Ollama
        answer_text = self._generate_llm_response(messages)

        # 5. Сохраняем текущий шаг в историю диалога
        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "assistant", "content": answer_text})

        # Ограничиваем длину истории (чтобы не перегружать контекст модели)
        if len(self.chat_history) > self.history_limit:
            self.chat_history = self.chat_history[-self.history_limit:]

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
    print("  'clear' или 'очистить' — стереть память диалога и начать новую тему")
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
            continue

        print("[Поиск в базе и генерация ответа...]")
        answer, sources = assistant.answer_question(query)

        # Выводим источники, если они были найдены
        #if sources:
        #    print("\n--- ИСПОЛЬЗОВАННЫЙ КОНТЕКСТ (ИСТОЧНИКИ) ---")
        #    for i, chunk in enumerate(sources, 1):
        #        preview = (
        #            chunk[:100].replace("\n", " ") + "..."
        #            if len(chunk) > 100
        #           else chunk
        #        )
        #       print(f"[{i}] {preview}")
        # else:
        #    print("\n[Контекст из базы не найден]")

        print("\n--- ОТВЕТ АССИСТЕНТА ---")
        print(answer)
        print("=" * 60)


if __name__ == "__main__":
    main()