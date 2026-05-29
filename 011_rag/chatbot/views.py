import json

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .models import ChatHistory

_rag = None
_llm = None
_retriever = None


def _get_rag_components():
    global _rag, _llm, _retriever
    if _rag is None:
        from rag_bge_m3_class_table import RagBgeM3
        _rag = RagBgeM3()
        _llm = _rag.get_llm()
        _retriever = _rag.build_rag_components()
    return _rag, _llm, _retriever


@csrf_exempt
def chat(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST만 허용됩니다.'}, status=405)

    try:
        data = json.loads(request.body)
        question = data.get('message', '').strip()
    except Exception:
        question = request.POST.get('message', '').strip()

    if not question:
        return JsonResponse({'answer': '질문을 입력해주세요.'})

    try:
        rag, llm, retriever = _get_rag_components()
        answer = rag.runnable_lambda(retriever, llm, question)
    except Exception as e:
        answer = f'오류가 발생했습니다: {e}'

    ChatHistory.objects.create(question=question, answer=answer)

    return JsonResponse({'answer': answer})


def index(request):
    return render(request, 'chatbot/chatbot_jundea.html')
