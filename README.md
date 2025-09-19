# 智能上下文压缩系统

## 项目概述

该系统允许用户上传文件并提出问题，系统会解析文件内容，压缩后与问题一起发送给大语言模型获取答案。

## 功能特性

- 支持多种文件格式：txt, csv, md, docx, pptx, xlsx
- 多种内容压缩策略：提取式摘要、生成式摘要、基于嵌入的压缩、关键词提取
- 智能Token管理，确保不超过模型上下文限制

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置

1. 复制环境变量文件：
```bash
cp .env.example .env
```

2. 在.env文件中添加您的OpenAI API密钥：
```
OPENAI_API_KEY=your_actual_api_key_here
```

## 运行应用

```bash
python main.py
```

应用将在 http://localhost:8000 上运行

## API文档

启动应用后，访问 http://localhost:8000/docs 查看Swagger UI文档
