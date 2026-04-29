from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.dependencies import get_retrieval_service, get_llm_service
from app.knowledge_engine.prompting import build_rfp_prompt
from app.services.chat_memory import chat_memory

router = APIRouter(prefix="/chat", tags=["Chat"])

class ChatRequest(BaseModel):
    session_id: str
    message: str


@router.post("/stream")
def chat_stream(req: ChatRequest):

    retrieval_service = get_retrieval_service()
    llm_service = get_llm_service()

    # 1. Retrieve context
    results = retrieval_service.search(req.message, top_k=5)
    context = [r["content"] for r in results]

    # 2. Build prompt
    prompt = build_rfp_prompt(req.message, context)

    # 3. Add memory
    history = chat_memory.get(req.session_id)
    for msg in history[-5:]:
        prompt += f"\n{msg['role']}: {msg['content']}"

    def generator():
        full_response = ""

        for chunk in llm_service.stream(prompt):
            full_response += chunk
            yield chunk

        # Save memory
        chat_memory.add(req.session_id, "user", req.message)
        chat_memory.add(req.session_id, "assistant", full_response)

    return StreamingResponse(generator(), media_type="text/plain")