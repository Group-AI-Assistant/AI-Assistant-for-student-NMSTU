from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .rag_engine import rag_query, build_index_from_documents, add_chunks_to_index
from .models import Document
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os

def index(request):
    return render(request, 'rag_app/index.html')

@csrf_exempt  # для простоты, но лучше использовать токен
def ask(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        question = data.get('question', '')
        if not question:
            return JsonResponse({'error': 'Вопрос не может быть пустым'}, status=400)
        try:
            answer = rag_query(question)
            return JsonResponse({'answer': answer})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    return JsonResponse({'error': 'Method not allowed'}, status=405)

def upload_page(request):
    """Страница загрузки файлов."""
    return render(request, 'rag_app/upload.html')

@csrf_exempt  # для простоты; в продакшене используйте CSRF-токен в форме
def upload_file(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Метод не разрешён'}, status=405)

    uploaded_file = request.FILES.get('file')
    if not uploaded_file:
        return JsonResponse({'error': 'Файл не выбран'}, status=400)

    # Проверка расширения
    if not uploaded_file.name.endswith('.txt'):
        return JsonResponse({'error': 'Поддерживаются только .txt файлы'}, status=400)

    try:
        # Читаем содержимое
        content = uploaded_file.read().decode('utf-8')
    except UnicodeDecodeError:
        return JsonResponse({'error': 'Ошибка кодировки. Файл должен быть в UTF-8.'}, status=400)

    # Разбиваем на чанки (простое разбиение по двойным переносам строк)
    chunks = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
    if not chunks:
        return JsonResponse({'error': 'Файл не содержит текста или слишком короток'}, status=400)

    # Добавляем в индекс
    try:
        add_chunks_to_index(chunks)
    except Exception as e:
        return JsonResponse({'error': f'Ошибка индексации: {str(e)}'}, status=500)

    # (Опционально) сохраняем сам файл в media
    # default_storage.save('documents/' + uploaded_file.name, ContentFile(uploaded_file.read()))

    return JsonResponse({
        'message': f'Файл успешно загружен и проиндексирован. Добавлено чанков: {len(chunks)}'
    })