# Smart Context Compression System (smartContext)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Build-Passing-brightgreen.svg" alt="Build Status">
  <img src="https://img.shields.io/badge/Release-v1.0.0-blue.svg" alt="Release">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg" alt="Supported Platforms">
  <img src="https://img.shields.io/badge/LLM-DeepSeek%20%7C%20Compatible-orange.svg" alt="Supported LLMs">
  <img src="https://img.shields.io/badge/Files-txt%20%7C%20csv%20%7C%20md%20%7C%20docx%20%7C%20pptx%20%7C%20xlsx-yellow.svg" alt="Supported File Formats">
</p>

<p align="center">
  <a href="#english-version">
    <img src="https://img.shields.io/badge/English-Click%20Here-blue" alt="EN">
  </a>
  <a href="#中文版本">
    <img src="https://img.shields.io/badge/中文-点击这里-red" alt="CN">
  </a>
</p>

<div align="center">
  <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/fastapi.svg" width="40" height="40" alt="FastAPI">
  <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/python.svg" width="40" height="40" alt="Python">
  <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/openai.svg" width="40" height="40" alt="OpenAI">
  <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/pandas.svg" width="40" height="40" alt="Pandas">
</div>

<div id="english-version"></div>

## English Version

### 📋 Project Overview

smartContext is an intelligent context compression system designed to help users efficiently process long document content when interacting with large language models. Through file parsing and content compression technologies, the system can deliver key information to large language models without exceeding model context limitations to obtain accurate answers.

### 🎯 Core Problems Solved

- Large language model input length limitations (Token limits)
- User-uploaded file content is too long for direct question answering
- Unified parsing and processing of different file formats

### ✨ Features

- **Multi-format File Support**: Supports txt, csv, md, docx, pptx, xlsx file formats
- **Multiple Compression Strategies**:
  - Extractive summarization
  - Abstractive summarization
  - Embedding-based compression
  - Keyword extraction
- **Smart Token Management**: Automatically calculates and controls total token count to ensure it does not exceed model limits
- **RESTful API Interface**: Built on FastAPI, providing standard RESTful interfaces
- **Auto-generated API Documentation**: Integrated with Swagger UI for easy API debugging and usage

### 🏗️ Technical Architecture

- **Backend Framework**: FastAPI 0.116.2
- **Server**: Uvicorn 0.34.3
- **LLM Framework**: LangChain (langchain==0.3.27)
- **File Processing**:
  - docx: python-docx==1.2.0
  - pptx: python-pptx==1.0.2
  - Excel/CSV: pandas==2.3.2
- **Text Processing**: tiktoken==0.11.0, beautifulsoup4==4.13.5
- **Configuration Management**: pydantic-settings==2.10.1, python-dotenv==1.0.0

### 🛠️ Requirements

- Python 3.10+
- pandoc (for document format conversion)
- DeepSeek API key or compatible API key

### 🚀 Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Aliyun mirror acceleration (optional)
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### ⚙️ Configuration

1. Copy environment file:
```bash
cp .env.example .env
```

2. Configure your API key and other parameters in the .env file:
```env
# DeepSeek API Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key_here
API_BASE=https://api.deepseek.com/v1
MODEL_NAME=deepseek-chat

# SiliconFlow API Configuration (for embedding models)
SILICONFLOW_API_KEY=your_siliconflow_api_key_here

# Context Configuration
MAX_CONTENT_TOKENS=4000
LLM_MAX_TOKENS=1000
TEMPERATURE=0.7
```

### ▶️ Running the Application

```bash
python main.py
```

The application will run on http://localhost:8894

### 📚 API Documentation

After starting the application, visit the following addresses to view the Swagger UI documentation:

- API Documentation: http://localhost:8894/docs
- ReDoc Documentation: http://localhost:8894/redoc

### 📖 Usage

1. After starting the service, visit http://localhost:8894/docs
2. Use the `/upload/` endpoint to upload files and ask questions
3. The system will automatically parse the file content and compress it according to the selected compression strategy
4. The compressed content will be sent to the large language model along with your question to get an answer

### 📁 Directory Structure

```
smartContext/
├── core/                 # Core business logic
│   ├── compression.py    # Content compression implementation
│   ├── file_parser.py    # File parsing implementation
│   ├── llm_integration.py# LLM integration
│   └── models.py         # Data model definitions
├── routers/              # API routes
│   └── upload_router.py  # Upload-related routes
├── utils/                # Utility modules
│   ├── config.py         # Configuration management
│   └── token_counter.py  # Token calculation
├── output_dir/           # Output file directory
└── main.py               # Application entry point
```

### 📝 Notes

- pandoc needs to be installed to support document format conversion
- Ensure the correct API key is configured to use the large language model service
- Uploaded files will be temporarily saved in the `output_dir` directory

---

<div id="中文版本"></div>

## 中文版本

### 📋 项目概述

smartContext 是一个智能上下文压缩系统，旨在帮助用户在与大语言模型交互时高效处理长文档内容。通过文件解析和内容压缩技术，系统能够在不超出模型上下文限制的前提下，将关键信息传递给大语言模型以获取准确回答。

### 🎯 核心解决的问题

- 大语言模型输入长度受限（Token限制）
- 用户上传的文件内容过长，难以直接用于问答
- 不同格式文件的统一解析与处理

### ✨ 功能特性

- **多格式文件支持**：支持 txt, csv, md, docx, pptx, xlsx 等多种文件格式
- **多种压缩策略**：
  - 提取式摘要
  - 生成式摘要
  - 基于嵌入的压缩
  - 关键词提取
- **智能Token管理**：自动计算并控制总 Token 数，确保不超过模型限制
- **RESTful API接口**：基于 FastAPI 构建，提供标准的 RESTful 接口
- **自动生成API文档**：集成 Swagger UI，方便接口调试和使用

### 🏗️ 技术架构

- **后端框架**：FastAPI 0.116.2
- **服务器**：Uvicorn 0.34.3
- **LLM框架**：LangChain (langchain==0.3.27)
- **文件处理**：
  - docx: python-docx==1.2.0
  - pptx: python-pptx==1.0.2
  - Excel/CSV: pandas==2.3.2
- **文本处理**：tiktoken==0.11.0, beautifulsoup4==4.13.5
- **配置管理**：pydantic-settings==2.10.1, python-dotenv==1.0.0

### 🛠️ 环境要求

- Python 3.10+
- pandoc (用于文档格式转换)
- DeepSeek API 密钥或兼容的 API 密钥

### 🚀 安装依赖

```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 阿里云镜像加速（可选）
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### ⚙️ 配置

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

### ▶️ 运行应用

```bash
python main.py
```

应用将在 http://localhost:8894 上运行

### 📚 API 文档

启动应用后，访问以下地址查看 Swagger UI 文档：

- API 文档地址：http://localhost:8894/docs
- ReDoc 文档地址：http://localhost:8894/redoc

### 📖 使用方法

1. 启动服务后，访问 http://localhost:8894/docs
2. 使用 `/upload/` 接口上传文件并提出问题
3. 系统会自动解析文件内容，根据选择的压缩策略进行内容压缩
4. 压缩后的内容会与您的问题一起发送给大语言模型获取答案

### 📁 目录结构

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

### 📝 注意事项

- 需要安装 pandoc 以支持文档格式转换功能
- 确保配置正确的 API 密钥以使用大语言模型服务
- 上传的文件会被临时保存在 `output_dir` 目录中