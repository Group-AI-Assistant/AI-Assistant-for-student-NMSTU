import ollama

try:
    # Пробуем новый синтаксис
    res = ollama.embed(model="nomic-embed-text", input="Привет")
    print("УСПЕХ! Метод embed работает, размер вектора:", len(res["embeddings"][0]))
except Exception as e1:
    print("Ошибка метода embed:", e1)
    try:
        # Пробуем старый синтаксис
        res = ollama.embeddings(model="nomic-embed-text", prompt="Привет")
        print("УСПЕХ! Метод embeddings работает, размер вектора:", len(res["embedding"]))
    except Exception as e2:
        print("Ошибка метода embeddings:", e2)