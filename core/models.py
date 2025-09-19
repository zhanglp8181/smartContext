from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, List, Dict, Any

class FileType(str, Enum):
    TEXT = "text"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    CSV = "csv"
    MD = "md"

class CompressionStrategy(str, Enum):
    EXTRACTIVE = "extractive"  # 提取式摘要
    ABSTRACTIVE = "abstractive" # 生成式摘要
    EMBEDDING = "embedding"    # 基于嵌入的压缩
    KEYWORD = "keyword"        # 关键词提取

class UploadRequest(BaseModel):
    file_content: str = Field(..., description="Base64编码的文件内容")
    file_name: str = Field(..., description="文件名")
    question: str = Field(..., description="用户问题")
    compression_strategy: CompressionStrategy = Field(
        default=CompressionStrategy.EXTRACTIVE,
        description="压缩策略"
    )

class ProcessedContent(BaseModel):
    original_text: str
    compressed_text: str
    token_count: int
    compression_ratio: float

class LLMResponse(BaseModel):
    answer: str
    source_document: str
    processing_time: float
