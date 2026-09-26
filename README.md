**Пока я (Ильяс) выбрал так:**
Основная LLM https://huggingface.co/RefalMachine/RuadaptQwen3-4B-Instruct-GGUF.
Эмбеддинг LLM https://huggingface.co/cPilotGod/baai-bge-m3-568m-gguf.
Обе в версиях Q4_K_M.

Чтобы их запустить нужно скачать llama.cpp через командную строку: winget install llama.cpp (Windows).

В качестве веб-интерфейса сейчас используется Gradio, потому что он создан для работы с нейросетями.

Поэтому если захотите потестить, то надо: 
1. Клонировать репозиторий себе на комп.
2. Отдельно скачать две LLM по ссылкам выше.
3. Установить llama.cpp.
4. Ввести в командную строку в папке проекта:
        winget install llama.cpp (Windows).
        python -m venv venv #создать виртуальную среду
        venv\Scripts\activate.bat #активировать среду
        pip install gradio openai #установить пакеты
5. Открыть 3 терминала (командные строки).
6. Ввести: llama-server -m my-model.gguf --host localhost --port 8080 (вместо my-model пишите название llm-файла, так две нейросети запускаете, qwen3 на порту 8080, bge-m3 embeding на порту 8081).

*Что-то в этой инструкции может не работать. Пишите, если не сработает.*

**Во вкладке Contributing описано как добавлять файлы сюда.**
