from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # DeepSeek API 配置
    api_key: str = Field(..., validation_alias="DEEPSEEK_API_KEY")
    api_base: str = Field("https://api.deepseek.com/v1", validation_alias="API_BASE")
    model_name: str = Field("deepseek-chat", validation_alias="MODEL_NAME")
    
    # 硅基流动 API 配置（用于嵌入模型）
    siliconflow_api_key: str = Field(..., validation_alias="SILICONFLOW_API_KEY")
    siliconflow_api_base: str = "https://api.siliconflow.cn/v1"
    siliconflow_embedding_model: str = "BAAI/bge-large-zh-v1.5"

    # 上下文窗口最大 token 数
    max_context_tokens: int = Field(4000, validation_alias="MAX_CONTENT_TOKENS")

    # 模型最大输出 token 数
    llm_max_tokens: int = Field(1000, validation_alias="LLM_MAX_TOKENS")

    # 温度参数
    temperature: float = Field(0.7, validation_alias="TEMPERATURE")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8"
    }

def get_settings():
    return Settings()