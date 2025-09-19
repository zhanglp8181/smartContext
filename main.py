import logging
import io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from core.file_parser import FileParser
from core.compression import ContentCompressor
from core.llm_integration import LLMIntegration
from core.models import UploadRequest, FileType, CompressionStrategy
import base64
from utils.token_counter import count_tokens
from utils.config import get_settings

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = FastAPI(title="智能上下文压缩系统")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化组件
file_parser = FileParser()
content_compressor = ContentCompressor()
llm_integration = LLMIntegration()
settings = get_settings()

@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    question: str = Form(...),
    compression_strategy: CompressionStrategy = Form(CompressionStrategy.EXTRACTIVE)
):
    try:
        logging.info(f"Processing upload request. File: {file.filename}, Strategy: {compression_strategy}")
        # 读取文件内容
        content = await file.read()
        logging.debug(f"File content length: {len(content)}")

        # 确定文件类型
        file_extension = file.filename.split('.')[-1].lower()
        file_type = None

        if file_extension in ['txt', 'md']:
            file_type = FileType.TEXT if file_extension == 'txt' else FileType.MD
        elif file_extension == 'csv':
            file_type = FileType.CSV
        elif file_extension == 'docx':
            file_type = FileType.DOCX
        elif file_extension == 'pptx':
            file_type = FileType.PPTX
        elif file_extension == 'xlsx':
            file_type = FileType.XLSX
        else:
            raise HTTPException(status_code=400, detail="不支持的文件类型")

        # 解析文件
        logging.info(f"Parsing file as type: {file_type}")
        # 对于二进制文件类型，传递原始字节内容
        if file_type in [FileType.DOCX, FileType.PPTX, FileType.XLSX]:
            parsed_content = file_parser.parse_file(
                content,  # 直接传递二进制内容
                file.filename,
                file_type
            )
        else:
            # 对于文本文件类型，先解码
            parsed_content = file_parser.parse_file(
                content.decode('utf-8'),
                file.filename,
                file_type
            )
        logging.debug(f"Parsed content length: {len(parsed_content)}")

        # 压缩内容
        logging.info("Starting content compression")
        processed_content = content_compressor.compress_content(
            parsed_content,
            question,
            compression_strategy,
            settings.max_context_tokens - count_tokens(question) - 100  # 预留100token给提示词
        )
        logging.debug(f"Compressed content length: {len(processed_content.compressed_text)}")

        # 获取LLM答案
        logging.info("Getting LLM answer")
        response = llm_integration.get_answer(
            processed_content.compressed_text,
            question
        )
        logging.debug(f"LLM response: {response.answer[:200]}...")

        return {
            "answer": response.answer,
            "compression_ratio": processed_content.compression_ratio,
            "original_tokens": count_tokens(parsed_content),
            "compressed_tokens": processed_content.token_count,
            "processing_time": response.processing_time
        }

    except Exception as e:
        logging.error(f"处理文件时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理文件时出错: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8894)