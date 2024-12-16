from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from llmlingua import PromptCompressor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Initialize LLMLingua Prompt Compressor
llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True,  # Whether to use llmlingua-2
    device_map="cpu"
)

# CORSの設定を追加（全てのオリジンを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # すべてのオリジンを許可
    allow_credentials=True,
    allow_methods=["*"],  # すべてのHTTPメソッドを許可
    allow_headers=["*"],  # すべてのヘッダーを許可
)

# Define a request model
class PromptRequest(BaseModel):
    prompt: str
    compression_rate: float = 0.33  # Default compression rate

# prompt圧縮
@app.post("/compress")
async def compress_prompt(request: PromptRequest):
    try:
        # Perform compression
        compressed_prompt = llm_lingua.compress_prompt(
            request.prompt, 
            rate=request.compression_rate, 
            force_tokens=['\n', '?']
        )
        return {"compressed_prompt": compressed_prompt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error compressing prompt: {str(e)}")

@app.get("/")
async def root():
    return {"message": "LLMLingua API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")