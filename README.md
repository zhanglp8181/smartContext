# 智能上下文压缩系统 (smartContext)

## 项目概述

smartContext 是一个智能上下文压缩系统，旨在帮助用户在与大语言模型交互时高效处理长文档内容。通过文件解析和内容压缩技术，系统能够在不超出模型上下文限制的前提下，将关键信息传递给大语言模型以获取准确回答。

### 核心解决的问题

- 大语言模型输入长度受限（Token限制）
- 用户上传的文件内容过长，难以直接用于问答
- 不同格式文件的统一解析与处理

## 功能特性

- **多格式文件支持**：支持 txt, csv, md, docx, pptx, xlsx 等多种文件格式
- **多种压缩策略**：
  - 提取式摘要 (extractive)
  - 生成式摘要 (abstractive)
  - 基于嵌入的压缩 (embedding)
  - 关键词提取 (keyword)
- **智能Token管理**：自动计算并控制总 Token 数，确保不超过模型限制
- **RESTful API接口**：基于 FastAPI 构建，提供标准的 RESTful 接口
- **自动生成API文档**：集成 Swagger UI，方便接口调试和使用

## 技术架构

- **后端框架**：FastAPI 0.116.2
- **服务器**：Uvicorn 0.34.3
- **LLM框架**：LangChain (langchain==0.3.27)
- **文件处理**：
  - docx: python-docx==1.2.0
  - pptx: python-pptx==1.0.2
  - Excel/CSV: pandas==2.3.2
- **文本处理**：tiktoken==0.11.0, beautifulsoup4==4.13.5
- **配置管理**：pydantic-settings==2.10.1, python-dotenv==1.0.0

## 环境要求

- Python 3.10+
- pandoc (用于文档格式转换)
- DeepSeek API 密钥或兼容的 API 密钥

## 安装依赖

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 阿里云镜像加速（可选）
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## 配置

1. 复制环境变量文件：
```bash
cp .env.example .env
```

2. 在 .env 文件中配置您的 API 密钥和其他参数：
```env
# DeepSeek API 配置
DEEPSEEK_API_KEY=your_deepseek_api_key_here
API_BASE=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat

# 硅基流动 API 配置（用于嵌入模型）
SILICONFLOW_API_KEY=your_siliconflow_api_key_here

# 上下文配置
MAX_CONTENT_TOKENS=4000
LLM_MAX_TOKENS=1000
TEMPERATURE=0.7
```

## 运行应用

```bash
python main.py
```

应用将在 http://localhost:8894 上运行

## API 文档

启动应用后，访问以下地址查看 Swagger UI 文档：

- API 文档地址：http://localhost:8894/docs
- ReDoc 文档地址：http://localhost:8894/redoc

## 使用方法

1. 启动服务后，访问 http://localhost:8894/docs
2. 使用 `/upload/` 接口上传文件并提出问题
3. 系统会自动解析文件内容，根据选择的压缩策略进行内容压缩
4. 压缩后的内容会与您的问题一起发送给大语言模型获取答案

## 目录结构

```
smartContext/
├── core/                 # 核心业务逻辑
│   ├── compression.py    # 内容压缩实现
│   ├── file_parser.py    # 文件解析实现
│   ├── llm_integration.py# LLM 集成
│   └── models.py         # 数据模型定义
├── routers/              # API 路由
│   └── upload_router.py  # 上传相关路由
├── utils/                # 工具模块
│   ├── config.py         # 配置管理
│   └── token_counter.py  # Token 计算
├── output_dir/           # 输出文件目录
└── main.py               # 应用入口文件
```

## 注意事项

- 需要安装 pandoc 以支持文档格式转换功能
- 确保配置正确的 API 密钥以使用大语言模型服务
- 上传的文件会被临时保存在 `output_dir` 目录中