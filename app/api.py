from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import ask_question


app = FastAPI(
    title="AI/ML Research Navigator API",
    description="RAG API for computer vision research papers",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/")
def root():
    return {
        "message": "AI/ML Research Navigator API"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):

    answer, documents = ask_question(request.question)

    sources = []
    seen = set()

    for document in documents:

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        page = document.metadata.get(
            "page",
            "Unknown"
        )

        source_key = (source, page)

        if source_key not in seen:
            seen.add(source_key)

            sources.append({
                "source": source,
                "page": page,
            })

    return {
        "answer": answer,
        "sources": sources,
    }