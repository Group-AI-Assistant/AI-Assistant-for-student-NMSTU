import chromadb


def inspect_database():
    DB_PATH = "./chroma_db"
    COLLECTION_NAME = "knowledge_base"

    print("=" * 60)
    print(f"Подключение к базе данных ChromaDB по пути: {DB_PATH}")
    print("=" * 60)

    try:
        # 1. Подключаемся к базе
        chroma_client = chromadb.PersistentClient(path=DB_PATH)

        # 2. Получаем список всех существующих коллекций (таблиц)
        collections = chroma_client.list_collections()
        col_names = [col.name for col in collections]
        print(f"Найденные коллекции в БД: {col_names}")

        if COLLECTION_NAME not in col_names:
            print(
                f"[ОШИБКА]: Коллекция '{COLLECTION_NAME}' не найдена. База пуста!"
            )
            return

        # 3. Подключаемся к нашей коллекции
        collection = chroma_client.get_collection(name=COLLECTION_NAME)

        # 4. Узнаем общее количество чанков
        total_chunks = collection.count()
        print(f"Общее количество чанков в базе: {total_chunks}")

        if total_chunks == 0:
            print("База данных пуста.")
            return

        # 5. Извлекаем ВСЕ чанки (документы, ID и метаданные)
        # По умолчанию .get() возвращает всё, кроме самих числовых векторов (они слишком длинные)
        db_content = collection.get()

        ids = db_content.get("ids", [])
        documents = db_content.get("documents", [])
        metadatas = db_content.get("metadatas", [])

        print("\n--- СПИСОК ЧАНКОВ В БАЗЕ ---")
        for idx in range(len(ids)):
            chunk_id = ids[idx]
            text = documents[idx]
            meta = metadatas[idx] if metadatas else {}

            # Красиво форматируем вывод
            source_file = meta.get("source", "Неизвестно")
            strategy = meta.get("strategy", "Неизвестно")
            chunk_idx = meta.get("chunk_index", "N/A")

            # Ограничиваем вывод текста чанка первыми 150 символами
            preview_text = (
                text[:150].replace("\n", " ") + "..."
                if len(text) > 150
                else text
            )

            print(f"\n[Чанк {idx + 1}]")
            print(f"  ID чанка:   {chunk_id}")
            print(f"  Источник:  {source_file}")
            print(f"  Стратегия: {strategy} (Индекс: {chunk_idx})")
            print(f"  Текст:     \"{preview_text}\"")
            print("-" * 40)

    except Exception as e:
        print(f"Произошла ошибка при чтении базы: {e}")


if __name__ == "__main__":
    inspect_database()