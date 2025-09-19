import logging
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from core.models import LLMResponse
from utils.config import get_settings
import time

# 获取日志记录器
logger = logging.getLogger(__name__)

class LLMIntegration:
    def __init__(self):
        self.settings = get_settings()
        self.llm = ChatOpenAI(
            temperature=self.settings.temperature,
            openai_api_key=self.settings.api_key,
            openai_api_base=self.settings.api_base,
            model_name=self.settings.model_name,
            max_tokens=self.settings.llm_max_tokens
        )

        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="基于以下上下文信息，请回答用户的问题。如果上下文信息不足以回答问题，请如实告知。\n\n上下文:\n{context}\n\n问题: {question}\n\n回答:"
        )
        logger.info("LLMIntegration initialized successfully")
        # 修改后
        # self.prompt_template = PromptTemplate(
        #     input_variables=["context", "question"],
        #     template="请根据以下文档内容回答问题。如果文档中没有相关信息，请说明无法根据提供的文档回答问题。\n\n文档内容:\n{context}\n\n问题:\n{question}\n\n请用中文回答:"
        # )
    def get_answer(self, context: str, question: str) -> LLMResponse:
        """使用LLM获取答案"""
        logger.info("Starting LLM processing")
        logger.debug(f"Context length: {len(context)}")
        logger.debug(f"Question: {question}")
        
        start_time = time.time()

        # 使用新的方式调用LLM
        formatted_prompt = self.prompt_template.format(context=context, question=question)
        logger.debug(f"Formatted prompt type: {type(formatted_prompt)}")
        logger.debug(f"Formatted prompt length: {len(formatted_prompt)}")
        logger.debug(f"Formatted prompt (first 200 chars): {formatted_prompt[:200]}...")
        
        # 额外检查确保格式化后的提示词是字符串
        if not isinstance(formatted_prompt, str):
            logger.error(f"Formatted prompt is not a string! Type: {type(formatted_prompt)}, Value: {formatted_prompt}")
            raise ValueError(f"Formatted prompt is not a string! Type: {type(formatted_prompt)}")
        
        try:
            logger.debug("Calling LLM with formatted prompt")
            # 使用Chat模型调用方式
            messages = [HumanMessage(content=formatted_prompt)]
            result = self.llm.invoke(messages)
            logger.debug(f"LLM result type: {type(result)}")
            logger.debug(f"LLM result: {str(result)[:200]}...")
            # 提取答案文本
            answer = result.content
            logger.debug(f"Extracted answer: {answer[:200]}...")

            processing_time = time.time() - start_time
            logger.info(f"LLM processing completed in {processing_time:.2f} seconds")

            return LLMResponse(
                answer=answer,
                source_document=context[:500] + "..." if len(context) > 500 else context,
                processing_time=processing_time
            )
        except Exception as e:
            logger.error(f"Error during LLM processing: {str(e)}", exc_info=True)
            raise