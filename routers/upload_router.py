from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from core.file_parser import FileParser
from core.compression import ContentCompressor
from core.llm_integration import LLMIntegration
from core.models import FileType, CompressionStrategy
from utils.token_counter import count_tokens
from utils.config import get_settings

router = APIRouter(prefix="/api/v1", tags=["upload"])

# 初始化组件
file_parser = FileParser()
content_compressor = ContentCompressor()
llm_integration = LLMIntegration()
settings = get_settings()

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    question: str = Form(...),
    compression_strategy: CompressionStrategy = Form(CompressionStrategy.EXTRACTIVE)
):
    try:
        # 读取文件内容
        content = await file.read()

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
        parsed_content = file_parser.parse_file(
            content.decode('utf-8'),
            file.filename,
            file_type
        )

        # 压缩内容
        processed_content = content_compressor.compress_content(
            parsed_content,
            question,
            compression_strategy,
            settings.max_context_tokens - count_tokens(question) - 100  # 预留100token给提示词
        )

        # 获取LLM答案
        response = llm_integration.get_answer(
            processed_content.compressed_text,
            question
        )

        return {
            "answer": response.answer,
            "compression_ratio": processed_content.compression_ratio,
            "original_tokens": count_tokens(parsed_content),
            "compressed_tokens": processed_content.token_count,
            "processing_time": response.processing_time
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理文件时出错: {str(e)}")
