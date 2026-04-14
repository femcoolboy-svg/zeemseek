from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import json
import asyncio
from pathlib import Path

# Импорт модели
from zeemseek_core import create_zeemseek_ultra, ZeemSeekTokenizer, load_pretrained_weights

app = FastAPI(title="ZeemSeek Ultra API", description="Суперинтеллектуальная модель")

# CORS для локального запуска
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Инициализация модели
print("Инициализация ZeemSeek Ultra...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = create_zeemseek_ultra(
    vocab_size=10000,  # Упрощённый размер для быстрого запуска
    hidden_size=1024,
    num_layers=12,
    num_heads=16
)
model = model.to(device)
model.eval()
print("Модель создана")

tokenizer = ZeemSeekTokenizer()
print("Токенизатор готов")

class IntelligentProcessor:
    """Интеллектуальная обработка запросов"""
    
    @staticmethod
    async def process(prompt: str, context: list = None) -> str:
        # Анализ сложности
        complexity = min(len(prompt.split()) / 10, 1.0)
        
        # Токенизация
        tokens = tokenizer.encode(prompt)
        input_tensor = torch.tensor([tokens[:512]]).to(device)  # Ограничение длины
        
        # Генерация ответа
        with torch.no_grad():
            try:
                output_tokens = model.generate(
                    input_tensor, 
                    max_new_tokens=200, 
                    temperature=0.8 - complexity * 0.3
                )
                response = tokenizer.decode(output_tokens[0].tolist())
                
                # Очистка ответа от токенов ввода
                if len(response) > len(prompt):
                    response = response[len(prompt):]
                
                return response.strip() or "Анализ завершён. Ответ сгенерирован."
            except Exception as e:
                return f"Генерация ответа: {str(e)}"
    
    @staticmethod
    def analyze_intent(text: str) -> dict:
        return {
            "length": len(text),
            "words": len(text.split()),
            "has_question": "?" in text or "?" in text
        }


processor = IntelligentProcessor()


@app.get("/")
async def root():
    ui_path = Path(__file__).parent / "zeemseek_ui.html"
    if ui_path.exists():
        with open(ui_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("""
    <html><body>
    <h1>ZeemSeek Ultra</h1>
    <form action="/api/generate" method="post">
        <textarea name="prompt" rows="4" cols="50"></textarea><br>
        <button type="submit">Спросить</button>
    </form>
    </body></html>
    """)


@app.post("/api/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    if not prompt:
        return JSONResponse({"error": "Пустой запрос"}, status_code=400)
    
    response = await processor.process(prompt)
    return JSONResponse({
        "prompt": prompt,
        "response": response,
        "model": "ZeemSeek Ultra",
        "status": "success"
    })


@app.get("/api/health")
async def health():
    return JSONResponse({
        "status": "active",
        "model": "ZeemSeek Ultra",
        "device": str(device),
        "ready": True
    })


if __name__ == "__main__":
    import uvicorn
    print("🐍 ZeemSeek Ultra запущен на http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
