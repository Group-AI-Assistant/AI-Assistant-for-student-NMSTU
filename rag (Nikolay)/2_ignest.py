import os
import re
import chromadb
from fastembed import TextEmbedding

# =====================================================================
# ВАРИАНТ 2: Смысловое разбиение по абзацам и логическим блокам
# =====================================================================
def chunk_by_paragraphs(
    text: str, max_chunk_size: int = 500, min_chunk_size: int = 100
) -> list[str]:
    """Разбивает текст по абзацам (двойной перевод строки).

    Короткие абзацы объединяет вместе, а длинные делит по предложениям.
    """
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for paragraph in raw_paragraphs:
        if len(current_chunk) + len(paragraph) + 1 <= max_chunk_size:
            current_chunk = (
                f"{current_chunk}\n{paragraph}".strip()
                if current_chunk
                else paragraph
            )
        else:
            if current_chunk:
                chunks.append(current_chunk)

            if len(paragraph) > max_chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                temp_chunk = ""
                for sent in sentences:
                    if len(temp_chunk) + len(sent) + 1 <= max_chunk_size:
                        temp_chunk = (
                            f"{temp_chunk} {sent}".strip()
                            if temp_chunk
                            else sent
                        )
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = sent
                current_chunk = temp_chunk
            else:
                current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

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

        # Нарезаем файл на смысловые абзацы
        chunks = chunk_by_paragraphs(
            text, max_chunk_size=500, min_chunk_size=100
        )
        print(f"Сформировано смысловых чанков: {len(chunks)}")

        if not chunks:
            continue

        # Модели семейства E5 требуют префикс 'passage: ' для индексируемого текста
        prepared_chunks = [f"passage: {chunk}" for chunk in chunks]

        # Пакетная векторизация через FastEmbed
        embeddings_generator = embed_model.embed(prepared_chunks)
        embeddings_list = [emb.tolist() for emb in embeddings_generator]

        # Заносим каждый чанк в БД
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings_list)):
            chunk_id = f"{file_name}_para_{idx}"

            collection.add(
                ids=[chunk_id],
                embeddings=[vector],
                documents=[chunk],
                metadatas=[
                    {
                        "source": file_name,
                        "strategy": "paragraphs",
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