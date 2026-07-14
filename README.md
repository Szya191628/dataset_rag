# Dataset RAG — 智能知识库问答系统

基于 **LangGraph + Milvus + Neo4j** 构建的 RAG（Retrieval-Augmented Generation）知识库系统，支持文档导入、多路并行检索、知识图谱查询和智能问答。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      前端 (FastAPI)                          │
│          http://localhost:8080  智能客服聊天界面               │
└──────────┬──────────────────────────────────┬───────────────┘
           │                                  │
     ┌─────▼─────┐                    ┌───────▼───────┐
     │ 查询服务    │                    │  导入服务      │
     │ :8001      │                    │  :8000        │
     └─────┬─────┘                    └───────┬───────┘
           │                                  │
     ┌─────▼─────────────────────┐    ┌──────▼──────────────┐
     │   查询 LangGraph 工作流    │    │  导入 LangGraph 工作流│
     │                           │    │                      │
     │  意图确认 → 4路并行检索    │    │  PDF解析 → 图片处理   │
     │  ├─ 向量检索 (Milvus)     │    │  → 文档分块 → 品名识别 │
     │  ├─ HyDE 向量检索         │    │  → BGE向量化          │
     │  ├─ 知识图谱 (Neo4j)      │    │  → Milvus入库         │
     │  └─ Web搜索 (MCP)         │    │                      │
     │  → RRF融合 → Rerank       │    └──────────────────────┘
     │  → 答案生成               │
     └───────────────────────────┘
```

## 核心特性

### 知识导入流程

| 步骤 | 说明 |
|------|------|
| PDF 解析 | 使用 MinerU (magic-pdf) 将 PDF 转换为 Markdown |
| 图片处理 | 提取文档中的图片，上传至 MinIO 对象存储 |
| 文档分块 | 按语义切分长文档为适合向量化的小片段 |
| 品名识别 | LLM 识别文档中的产品名称，建立产品-文档关联 |
| BGE 向量化 | 使用 BGE-M3 模型生成稠密/稀疏向量 |
| Milvus 入库 | 将向量数据持久化存储至 Milvus 向量数据库 |

### 多路并行检索

查询时同时启动 **4 路并行检索**，通过 RRF（Reciprocal Rank Fusion）融合排序：

| 检索路径 | 说明 |
|---------|------|
| **向量检索** | BGE-M3 稠密向量 + 稀疏向量混合检索 |
| **HyDE 向量检索** | 先让 LLM 生成假设性答案，再用该答案做向量检索 |
| **知识图谱检索** | 基于 Neo4j 图数据库的实体关系查询 |
| **Web 搜索** | 通过 MCP (Model Context Protocol) 调用外部搜索 |

检索结果经 **RRF 融合** → **BGE-Reranker 重排序** → **LLM 生成最终答案**。

## 技术栈

| 组件 | 技术 |
|------|------|
| 工作流引擎 | LangGraph |
| 向量数据库 | Milvus |
| 图数据库 | Neo4j |
| 对象存储 | MinIO |
| 对话历史 | MongoDB |
| 文档解析 | MinerU (magic-pdf) |
| 嵌入模型 | BGE-M3 (FlagEmbedding) |
| 重排模型 | BGE-Reranker-v2-M3 |
| LLM | 通义千问 / OpenAI 兼容接口 |
| Web 搜索 | 阿里云百炼 MCP |
| 后端框架 | FastAPI + Uvicorn |
| 包管理 | uv |

## 项目结构

```
dataset_rag/
├── app/
│   ├── clients/              # 外部服务客户端
│   │   ├── milvus_utils.py       # Milvus 向量数据库操作
│   │   ├── neo4j_utils.py        # Neo4j 图数据库操作
│   │   ├── mongo_history_utils.py# MongoDB 对话历史
│   │   └── minio_utils.py        # MinIO 对象存储
│   ├── conf/                 # 配置模块
│   │   ├── lm_config.py          # LLM 配置
│   │   ├── embedding_config.py   # 嵌入模型配置
│   │   ├── reranker_config.py    # 重排模型配置
│   │   ├── milvus_config.py      # Milvus 连接配置
│   │   ├── mineru_config.py      # MinerU 解析配置
│   │   └── minio_config.py       # MinIO 连接配置
│   ├── core/                 # 核心工具
│   │   ├── logger.py             # 统一日志 (Loguru)
│   │   └── load_prompt.py        # Prompt 加载器
│   ├── frontend/             # 前端服务
│   │   ├── index.html            # 智能客服聊天页面
│   │   └── server.py             # 前端 FastAPI 服务
│   ├── import_process/       # 知识导入流程
│   │   ├── agent/
│   │   │   ├── main_graph.py     # 导入 LangGraph 工作流定义
│   │   │   ├── state.py          # 导入状态定义
│   │   │   └── nodes/            # 各处理节点
│   │   │       ├── node_entry.py             # 入口/参数校验
│   │   │       ├── node_pdf_to_md.py         # PDF → Markdown
│   │   │       ├── node_md_img.py            # 图片处理
│   │   │       ├── node_document_split.py    # 文档分块
│   │   │       ├── node_item_name_recognition.py # 品名识别
│   │   │       ├── node_bge_embedding.py     # BGE 向量化
│   │   │       └── node_import_milvus.py     # Milvus 入库
│   │   └── api/
│   │       └── file_import_service.py  # 文件上传 API (:8000)
│   ├── query_process/        # 查询问答流程
│   │   ├── agent/
│   │   │   ├── main_graph.py     # 查询 LangGraph 工作流定义
│   │   │   ├── state.py          # 查询状态定义
│   │   │   └── nodes/            # 各处理节点
│   │   │       ├── node_item_name_confirm.py # 意图/品名确认
│   │   │       ├── node_search_embedding.py  # 向量检索
│   │   │       ├── node_search_embedding_hyde.py # HyDE 检索
│   │   │       ├── node_query_kg.py          # 知识图谱检索
│   │   │       ├── node_web_search_mcp.py    # Web 搜索
│   │   │       ├── node_rrf.py               # RRF 融合
│   │   │       ├── node_rerank.py            # Rerank 重排
│   │   │       └── node_answer_output.py     # 答案生成
│   │   └── api/
│   │       └── query_service.py  # 查询 API (:8001)
│   ├── lm/                   # LLM 调用封装
│   ├── tool/                 # 工具脚本（模型下载等）
│   └── utils/                # 通用工具函数
├── prompts/                  # Prompt 模板文件
├── output/                   # 文件处理输出目录
├── milvus-data/              # Milvus 本地数据
├── pyproject.toml            # 项目依赖
├── uv.toml                   # uv 包管理配置
└── .env                      # 环境变量（不入库）
```

## 快速开始

### 1. 环境要求

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) 包管理器

### 2. 安装依赖

```bash
# 安装 uv（如尚未安装）
pip install uv

# 同步依赖
uv sync
```

### 3. 配置环境变量

复制 `.env.example`（如有）或手动创建 `.env` 文件，配置以下关键变量：

```env
# LLM 配置
LLM_DEFAULT_MODEL=your-model-name
MIMO_API_KEY=your-api-key
MIMO_BASE_URL=your-base-url

# 嵌入模型
BGE_DEVICE=cuda:0          # 或 cpu

# Milvus 向量数据库
MILVUS_URL=http://localhost:19530
CHUNKS_COLLECTION=your_chunks_collection
EMBEDDING_DIM=1024

# MongoDB
MONGO_URL=mongodb://localhost:27017
MONGO_DB_NAME=your_db_name

# MinIO 对象存储
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=your-access-key
MINIO_SECRET_KEY=your-secret-key
MINIO_BUCKET_NAME=your-bucket

# Reranker
BGE_RERANKER_LARGE=your-reranker-model-path
BGE_RERANKER_DEVICE=cuda:0

# MCP Web 搜索
DASHSCOPE_API_KEY=your-dashscope-key
MCP_DASHSCOPE_BASE_URL=your-mcp-url

# MinerU 文档解析
MINERU_API_TOKEN=your-mineru-token
MINERU_BASE_URL=your-mineru-url
```

### 4. 启动服务

需要分别启动 3 个服务：

```bash
# 终端 1：启动知识导入服务 (端口 8000)
uv run python -m app.import_process.api.file_import_service

# 终端 2：启动查询问答服务 (端口 8001)
uv run python -m app.query_process.api.query_service

# 终端 3：启动前端页面 (端口 8080)
uv run python -m app.frontend.server
```

### 5. 访问系统

| 地址 | 说明 |
|------|------|
| http://localhost:8080 | 智能客服聊天界面 |
| http://localhost:8000/import.html | 文件导入页面 |
| http://localhost:8000/docs | 导入服务 API 文档 (Swagger) |
| http://localhost:8001/docs | 查询服务 API 文档 (Swagger) |

## API 接口

### 文件导入服务 (:8000)

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/upload` | 上传文件，支持多文件批量上传 |
| `GET` | `/status/{task_id}` | 查询任务处理进度 |
| `GET` | `/import.html` | 文件导入前端页面 |
| `GET` | `/health` | 健康检查 |

### 查询问答服务 (:8001)

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/query` | 发送查询，支持同步/流式 (SSE) 返回 |
| `GET` | `/stream/{session_id}` | SSE 流式获取实时结果 |
| `GET` | `/history/{session_id}` | 获取会话历史记录 |
| `DELETE` | `/history/{session_id}` | 清空会话历史 |
| `GET` | `/health` | 健康检查 |

### 查询请求示例

```json
POST /query
{
  "query": "华为显示器怎么连接电脑？",
  "session_id": "optional-session-id",
  "is_stream": true
}
```

## Prompt 模板

`prompts/` 目录下存放各环节的 Prompt 模板：

| 文件 | 用途 |
|------|------|
| `answer_out.prompt` | 最终答案生成 |
| `hyde_prompt.prompt` | HyDE 假设性答案生成 |
| `rewritten_query_and_itemnames.prompt` | 查询改写与品名提取 |
| `item_name_recognition.prompt` | 文档品名识别 |
| `product_recognition_system.prompt` | 产品识别系统提示词 |
| `image_summary.prompt` | 图片内容摘要 |

## License

本项目仅供学习和研究使用。
