from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .rag_engine import rag_query, build_index_from_documents
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

# Если нужно загружать документы через веб-интерфейс
def upload_document(request):
    if request.method == 'POST' and request.FILES.get('file'):
        file = request.FILES['file']
        # Сохраняем файл
        doc = Document.objects.create(title=file.name, file=file)
        # Читаем текст (пример для .txt, для PDF/DOCX нужны дополнительные библиотеки)
        content = file.read().decode('utf-8')
        # Разбивка на чанки (простая)
        chunks = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
        # Перестраиваем индекс (добавляем новые чанки к существующим)
        # ВАЖНО: для больших объёмов лучше перестраивать весь индекс, но для простоты сделаем так:
        index, old_chunks = load_index()
        if index is None:
            build_index_from_documents(chunks)
        else:
            new_vectors = embed_batch(chunks)
            index.add(new_vectors)
            all_chunks = old_chunks + chunks
            save_index(index, all_chunks)
        doc.processed = True
        doc.save()
        return JsonResponse({'status': 'ok', 'chunks_added': len(chunks)})
    return JsonResponse({'error': 'Invalid request'}, status=400)