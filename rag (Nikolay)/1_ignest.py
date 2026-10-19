import os
import chromadb
from fastembed import TextEmbedding

# =====================================================================
# ВАРИАНТ 1: Фиксированное разбиение по длине с перекрытием
# =====================================================================
def chunk_by_fixed_window(
    text: str, chunk_size: int = 400, overlap: int = 80
) -> list[str]:
    """Режет текст на равные отрезки по символам с перекрытием (overlap)."""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def main():
    print("=" * 60)
    print("Инициализация FastEmbed (мультиязычные эмбеддинги E5)...")
    # Используем мультиязычную модель, отлично понимающую русский язык
    embed_model = TextEmbedding(model_name="intfloat/multilingual-e5-large")

    # Инициализация локальной базы ChromaDB (создаст папку ./chroma_db)
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # Создаем или подключаемся к коллекции
    collection = chroma_client.get_or_create_collection(
        name="knowledge_base", metadata={"hnsw:space": "cosine"}
    )

    data_folder = "./data"
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
        print(
            f"Создана папка '{data_folder}'. Положите туда .txt файлы и запустите скрипт снова."
        )
        return

    # Перебираем все .txt файлы в папке data
    files = [f for f in os.listdir(data_folder) if f.endswith(".txt")]
    if not files:
        print(f"В папке '{data_folder}' нет текстовых файлов (.txt).")
        return

    total_added_chunks = 0
    skipped_files = 0

    for file_name in files:
        # ПРОВЕРКА: Была ли база уже наполнена этим файлом ранее
        existing = collection.get(where={"source": file_name}, limit=1)
        if existing and existing.get("ids"):
            print(f"Файл '{file_name}' уже есть в базе знаний. Пропуск...")
            skipped_files += 1
            continue

        file_path = os.path.join(data_folder, file_name)
        print(f"\nОбработка НОВОГО файла: {file_name}...")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Нарезаем файл на чанки фиксированного размера
        chunks = chunk_by_fixed_window(text, chunk_size=400, overlap=80)
        print(f"Сформировано чанков: {len(chunks)}")

        if not chunks:
            continue

        # Модели семейства E5 требуют префикс 'passage: ' для индексируемого текста
        prepared_chunks = [f"passage: {chunk}" for chunk in chunks]

        # Быстрая пакетная векторизация через FastEmbed
        embeddings_generator = embed_model.embed(prepared_chunks)
        embeddings_list = [emb.tolist() for emb in embeddings_generator]

        # Заносим каждый чанк в БД
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings_list)):
            chunk_id = f"{file_name}_fixed_{idx}"

            collection.add(
                ids=[chunk_id],
                embeddings=[vector],
                documents=[chunk],
                metadatas=[
                    {
                        "source": file_name,
                        "strategy": "fixed_window",
                        "chunk_index": idx,
                    }
                ],
            )
            total_added_chunks += 1

    print("\n" + "=" * 60)
    print(f"[УСПЕХ] Индексация завершена!")
    print(f"Пропущено (уже были в базе): {skipped_files} файлов.")
    print(f"Успешно обработано новых чанков занесено: {total_added_chunks}")
    print("=" * 60)


if __name__ == "__main__":
    main()