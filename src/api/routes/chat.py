import json
from fastapi import APIRouter, Depends, Query, UploadFile, File, Form, HTTPException
from src.core.models.chat import ChatRequest, ChatResponse
from src.api.dependencies import get_current_pipeline, get_naive_pipeline, get_advanced_pipeline, get_modular_pipeline

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(
    pipeline_type: str = Query(
        None, description="'naive', 'advanced' или 'modular'"),
    data: str = Form(...),          # Получаем наш JSON как строку из формы
    file: UploadFile = File(None)   # Получаем файл (по умолчанию None)
):
    # 1. Превращаем строку обратно в объект ChatRequest
    try:
        request_dict = json.loads(data)
        request = ChatRequest(**request_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения JSON: {e}")

    # 2. Обрабатываем прикрепленный файл
    file_context = ""
    if file:
        try:
            content = await file.read()
            # Пока читаем как обычный текст. Для реальных PDF/DOCX тут понадобятся библиотеки типа PyMuPDF
            text_content = content.decode('utf-8')
            file_context = f"\n\n[СОДЕРЖИМОЕ ПРИКРЕПЛЕННОГО ФАЙЛА '{file.filename}']:\n{text_content}"
            print(f"[ФАЙЛ ПОЛУЧЕН] Имя: {file.filename}")
        except Exception as e:
            file_context = f"\n\n[ОШИБКА] Не удалось прочитать текст из файла {file.filename}."
            print(f"Ошибка чтения файла: {e}")

    # 3. Приклеиваем текст файла к сообщению пользователя
    final_message = request.message + file_context

    # 4. Выбираем нужный пайплайн
    if pipeline_type == "naive":
        pipeline = get_naive_pipeline()
    elif pipeline_type == "advanced":
        pipeline = get_advanced_pipeline()
    elif pipeline_type == "modular":
        pipeline = get_modular_pipeline()
    else:
        pipeline = get_current_pipeline()

    # 5. Вызываем пайплайн с обновленным сообщением
    response = pipeline.query(
        user_message=final_message,  # <-- Передаем текст вместе с файлом!
        history=request.history,
        use_vector_db=request.use_vector_db,
        use_web_search=request.use_web_search
    )
    return response
