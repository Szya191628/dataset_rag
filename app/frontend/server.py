"""
小爱商城 - 智能客服前端服务
启动后访问 http://127.0.0.1:8080 打开前端页面
"""
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="小爱商城前端服务", description="智能客服前端入口")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).parent


@app.get("/")
async def index():
    """返回前端主页面"""
    return FileResponse(FRONTEND_DIR / "index.html", media_type="text/html")


@app.get("/health")
async def health():
    return {"ok": True, "service": "frontend"}


if __name__ == "__main__":
    print("=" * 50)
    print("  小爱商城 - 智能客服前端服务")
    print("  访问地址: http://127.0.0.1:8080")
    print("=" * 50)
    uvicorn.run(app, host="127.0.0.1", port=8080)
