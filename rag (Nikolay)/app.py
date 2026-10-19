from flask import Flask, jsonify, render_template, request
from Assistent import RAGAssistant

app = Flask(__name__)

# Инициализируем нашего RAG-ассистента один раз при старте веб-сервера
print("Инициализация RAG-ассистента для веб-интерфейса...")
try:
    assistant = RAGAssistant(llm_model="qwen2.5:3b")
    print("RAG-ассистент успешно загружен и готов к приему веб-запросов!")
except Exception as e:
    print(f"Ошибка при старте ассистента: {e}")
    assistant = None


# Главная страница (выдает HTML-интерфейс)
@app.route("/")
def index():
    return render_template("index.html")


# Эндпоинт для обработки вопросов из формы на сайте
@app.route("/api/ask", methods=["POST"])
def ask():
    if not assistant:
        return jsonify({"error": "Ассистент не инициализирован"}), 500

    data = request.get_json()
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "Пустой запрос"}), 400

    # Вызываем логику RAG
    answer, sources = assistant.answer_question(user_query)

    # Возвращаем результат в формате JSON
    return jsonify({"answer": answer, "sources": sources})


# Эндпоинт для сброса памяти диалога
@app.route("/api/clear", methods=["POST"])
def clear_history():
    if assistant:
        assistant.clear_memory()
    return jsonify({"status": "ok", "message": "Память диалога очищена"})


if __name__ == "__main__":
    # Запускаем локальный веб-сервер на порту 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
