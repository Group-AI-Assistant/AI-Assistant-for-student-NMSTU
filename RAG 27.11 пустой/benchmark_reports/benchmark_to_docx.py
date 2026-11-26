import json
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

# Загружаем JSON
with open('benchmark.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

doc = Document()

# Базовый стиль
style = doc.styles['Normal']
style.font.name = 'Times New Roman'
style.font.size = Pt(12)

for item in data['results']:
    num = item['number']
    question = item['question']
    seconds = item['seconds']
    answer = item['answer']
    sources = item['sources']
    
    # Заголовок вопроса (жирный, 14 pt)
    p = doc.add_paragraph()
    run = p.add_run(f"{num}. {question}")
    run.bold = True
    run.font.size = Pt(14)
    
    # Время ответа (курсив)
    p = doc.add_paragraph()
    run = p.add_run(f"Время ответа: {seconds} с")
    run.italic = True
    
    # Ответ
    doc.add_paragraph(answer)
    
    # Источники
    if sources:
        doc.add_paragraph("Источники:")
        for src in sources:
            doc.add_paragraph(src, style='List Bullet')
    
    # Пустая строка между вопросами
    doc.add_paragraph()

# Сохраняем результат
doc.save('output.docx')