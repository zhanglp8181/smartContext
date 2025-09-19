# Smart Context Compression System (smartContext)

<p align="center">
  <a href="README_CN.md">🇨🇳 中文版 README</a> •
  <a href="README.md">🇺🇸 English README</a>
</p>

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

<div align="center">
  <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/fastapi.svg" width="40" height="40" alt="FastAPI">
  <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/python.svg" width="40" height="40" alt="Python">
  <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/openai.svg" width="40" height="40" alt="OpenAI">
  <img src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/pandas.svg" width="40" height="40" alt="Pandas">
</div>

## 📋 Project Overview

smartContext is an intelligent context compression system designed to help users efficiently process long document content when interacting with large language models. Through file parsing and content compression technologies, the system can deliver key information to large language models without exceeding model context limitations to obtain accurate answers.

## 🎯 Core Problems Solved

- Large language model input length limitations (Token limits)
- User-uploaded file content is too long for direct question answering
- Unified parsing and processing of different file formats

## ✨ Features

- **Multi-format File Support**: Supports txt, csv, md, docx, pptx, xlsx file formats
- **Multiple Compression Strategies**:
  - Extractive summarization
  - Abstractive summarization
  - Embedding-based compression
  - Keyword extraction
- **Smart Token Management**: Automatically calculates and controls total token count to ensure it does not exceed model limits
- **RESTful API Interface**: Built on FastAPI, providing standard RESTful interfaces
- **Auto-generated API Documentation**: Integrated with Swagger UI for easy API debugging and usage

## 🏗️ Technical Architecture

- **Backend Framework**: FastAPI 0.116.2
- **Server**: Uvicorn 0.34.3
- **LLM Framework**: LangChain (langchain==0.3.27)
- **File Processing**:
  - docx: python-docx==1.2.0
  - pptx: python-pptx==1.0.2
  - Excel/CSV: pandas==2.3.2
- **Text Processing**: tiktoken==0.11.0, beautifulsoup4==4.13.5
- **Configuration Management**: pydantic-settings==2.10.1, python-dotenv==1.0.0

## 🛠️ Requirements

- Python 3.10+
- pandoc (for document format conversion)
- DeepSeek API key or compatible API key

## 🚀 Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Aliyun mirror acceleration (optional)
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## ⚙️ Configuration

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

## ▶️ Running the Application

```bash
python main.py
```

The application will run on http://localhost:8894

## 📚 API Documentation

After starting the application, visit the following addresses to view the Swagger UI documentation:

- API Documentation: http://localhost:8894/docs
- ReDoc Documentation: http://localhost:8894/redoc

## 📖 Usage

1. After starting the service, visit http://localhost:8894/docs
2. Use the `/upload/` endpoint to upload files and ask questions
3. The system will automatically parse the file content and compress it according to the selected compression strategy
4. The compressed content will be sent to the large language model along with your question to get an answer

## 📁 Directory Structure

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

## 📝 Notes

- pandoc needs to be installed to support document format conversion
- Ensure the correct API key is configured to use the large language model service
- Uploaded files will be temporarily saved in the `output_dir` directory

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.