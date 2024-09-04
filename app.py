from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

app = FastAPI()

# Models for request/response payloads

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[List[str]] = None

class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]
    usage: dict

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[dict]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[dict]
    usage: dict

# Mock data for endpoints
import time

def mock_choice(text):
    return {
        "text": text,
        "index": 0,
        "logprobs": None,
        "finish_reason": "length"
    }

def mock_chat_choice(content):
    return {
        "message": {"role": "assistant", "content": content},
        "index": 0,
        "finish_reason": "stop"
    }

# Endpoints

@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {"id": "text-davinci-003", "object": "model", "owned_by": "openai"},
            {"id": "gpt-4", "object": "model", "owned_by": "openai"}
        ]
    }

@app.get("/v1/models/{model_id}")
async def retrieve_model(model_id: str):
    if model_id not in ["text-davinci-003", "gpt-4"]:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"id": model_id, "object": "model", "owned_by": "openai"}

@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    return CompletionResponse(
        id="cmpl-xyz",
        object="text_completion",
        created=int(time.time()),
        model=request.model,
        choices=[mock_choice(f"Generated text for prompt: {request.prompt}")],
        usage={"prompt_tokens": 5, "completion_tokens": request.max_tokens, "total_tokens": request.max_tokens + 5}
    )

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    last_message = request.messages[-1]["content"]
    return ChatCompletionResponse(
        id="chatcmpl-xyz",
        object="chat.completion",
        created=int(time.time()),
        model=request.model,
        choices=[mock_chat_choice(f"Response to: {last_message}")],
        usage={"prompt_tokens": 5, "completion_tokens": 20, "total_tokens": 25}
    )

@app.post("/v1/edits")
async def create_edit():
    return {
        "id": "edit-xyz",
        "object": "edit",
        "choices": [
            {"text": "Corrected text.", "index": 0}
        ]
    }

@app.get("/v1/fine-tunes")
async def list_finetunes():
    return {"data": []}

@app.get("/v1/files/{file_id}")
async def retrieve_file(file_id: str):
    return {"id": file_id, "object": "file"}

@app.post("/v1/files")
async def upload_file():
    return {"id": "file-xyz", "object": "file"}

@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str):
    return {"id": file_id, "object": "file", "deleted": True}


# Add uvicorn to main call
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
