import logging
import tempfile
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.text_splitter import TokenTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import json
from core.models import CompressionStrategy, ProcessedContent
from utils.token_counter import count_tokens
from utils.config import get_settings
from langchain.text_splitter import MarkdownHeaderTextSplitter

# 获取日志记录器
logger = logging.getLogger(__name__)

class SiliconFlowEmbeddings:
    """硅基流动嵌入模型封装类"""
    def __init__(self, api_key, api_base, model_name):
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
    
    def embed_query(self, text):
        """嵌入单个查询文本"""
        logger.debug(f"Embedding query: {text[:100]}...")
        return self.embed_documents([text])[0]
    
    def embed_documents(self, texts):
        """嵌入多个文档"""
        logger.debug(f"Embedding documents, count: {len(texts)}")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # 确保输入始终是列表
        if not isinstance(texts, list):
            texts = [texts]
        
        payload = {
            "model": self.model_name,
            "input": texts
        }
        
        logger.debug(f"Sending embedding request with payload: {json.dumps(payload)[:200]}...")
        
        try:
            response = requests.post(
                f"{self.api_base}/embeddings",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            logger.debug(f"Received {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"硅基流动嵌入API调用失败: {str(e)}")
            raise Exception(f"硅基流动嵌入API调用失败: {str(e)}")

class ContentCompressor:
    def __init__(self):
        self.settings = get_settings()
        self.llm = ChatOpenAI(
            temperature=0,
            openai_api_key=self.settings.api_key,
            openai_api_base=self.settings.api_base,
            model_name=self.settings.model_name
        )
        # 使用硅基流动的嵌入模型
        self.embeddings = SiliconFlowEmbeddings(
            api_key=self.settings.siliconflow_api_key,
            api_base=self.settings.siliconflow_api_base,
            model_name=self.settings.siliconflow_embedding_model
        )
        logger.info("ContentCompressor initialized successfully")
    
    def compress_content(
        self, 
        text: str, 
        question: str, 
        strategy: CompressionStrategy,
        max_tokens: int = 4000
    ) -> ProcessedContent:
        """根据策略压缩文本内容"""
        logger.info(f"Starting compression with strategy: {strategy}")
        logger.debug(f"Input text length: {len(text)} characters")
        logger.debug(f"Question: {question}")
        logger.debug(f"Max tokens limit: {max_tokens}")
        
        original_tokens = count_tokens(text)
        logger.debug(f"Original text token count: {original_tokens}")
        
        if original_tokens <= max_tokens:
            # 如果已经满足token限制，不需要压缩
            logger.info("Text already within token limit, no compression needed")
            logger.debug(f"Original token count {original_tokens} <= max_tokens {max_tokens}")
            return ProcessedContent(
                original_text=text,
                compressed_text=text,
                token_count=original_tokens,
                compression_ratio=1.0
            )
        
        if strategy == CompressionStrategy.EXTRACTIVE:
            logger.info("Using extractive compression")
            compressed_text = self._extractive_compression(text, question, max_tokens)
        elif strategy == CompressionStrategy.ABSTRACTIVE:
            logger.info("Using abstractive compression")
            compressed_text = self._abstractive_compression(text, question, max_tokens)
        elif strategy == CompressionStrategy.EMBEDDING:
            logger.info("Using embedding-based compression")
            compressed_text = self._embedding_based_compression(text, question, max_tokens)
        elif strategy == CompressionStrategy.KEYWORD:
            logger.info("Using keyword-based compression")
            compressed_text = self._keyword_based_compression(text, question, max_tokens)
        else:
            raise ValueError(f"Unknown compression strategy: {strategy}")
        
        # 将压缩后的内容输出到临时文件
        self._save_compressed_content_to_temp_file(compressed_text, strategy)
        
        compressed_tokens = count_tokens(compressed_text)
        compression_ratio = compressed_tokens / original_tokens
        logger.info(f"Compression completed. Original tokens: {original_tokens}, Compressed tokens: {compressed_tokens}, Ratio: {compression_ratio}")
        
        return ProcessedContent(
            original_text=text,
            compressed_text=compressed_text,
            token_count=compressed_tokens,
            compression_ratio=compression_ratio
        )
        
    def _save_compressed_content_to_temp_file(self, compressed_text: str, strategy: CompressionStrategy):
        """将压缩后的内容保存到临时文件"""
        try:
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix=f'_{strategy.value}_compressed.txt', delete=False, encoding='utf-8')
            temp_file.write(compressed_text)
            temp_file.close()
            logger.info(f"Compressed content saved to temporary file: {temp_file.name}")
        except Exception as e:
            logger.error(f"Failed to save compressed content to temporary file: {str(e)}")
    
    # def _extractive_compression(self, text: str, question: str, max_tokens: int) -> str:
    #     """提取式摘要压缩"""
    #     logger.debug("Starting extractive compression")
    #     logger.debug(f"Input text length: {len(text)}, question: {question}")
    #     logger.debug(f"Max tokens: {max_tokens}")
        
    #     # 使用文本分割器将文本分成块
    #     text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
    #     chunks = text_splitter.split_text(text)
    #     logger.debug(f"Split text into {len(chunks)} chunks")
        
    #     # 使用TF-IDF找到与问题最相关的文本块
    #     logger.debug("Calculating TF-IDF similarities")
    #     vectorizer = TfidfVectorizer().fit_transform([question] + chunks)
    #     vectors = vectorizer.toarray()
    #     question_vector = vectors[0]
    #     chunk_vectors = vectors[1:]
        
    #     # 计算余弦相似度
    #     similarities = cosine_similarity([question_vector], chunk_vectors)[0]
    #     logger.debug(f"Similarities calculated: {similarities}")
        
    #     # 选择最相关的块直到达到token限制
    #     selected_indices = np.argsort(similarities)[::-1]
    #     logger.debug(f"Selected indices by relevance: {selected_indices}")
        
    #     selected_text = []
    #     current_tokens = 0
        
    #     for idx in selected_indices:
    #         chunk = chunks[idx]
    #         chunk_tokens = count_tokens(chunk)
    #         logger.debug(f"Evaluating chunk {idx}: {chunk_tokens} tokens, similarity: {similarities[idx]}")
            
    #         if current_tokens + chunk_tokens <= max_tokens:
    #             selected_text.append(chunk)
    #             current_tokens += chunk_tokens
    #             logger.debug(f"Added chunk {idx}, cumulative tokens: {current_tokens}")
    #         else:
    #             logger.debug(f"Skipping chunk {idx}: would exceed token limit ({current_tokens} + {chunk_tokens} > {max_tokens})")
    #             break
        
    #     result = "\n".join(selected_text)
    #     logger.debug(f"Extractive compression completed. Result length: {len(result)}")
    #     return result
    # 修改 _extractive_compression 方法
    def _extractive_compression(self, text: str, question: str, max_tokens: int) -> str:
        """提取式摘要压缩 - 针对Markdown文档优化"""
        logger.debug("Starting extractive compression for markdown")
        logger.debug(f"Input text length: {len(text)}, question: {question}")
        logger.debug(f"Max tokens: {max_tokens}")
        
        # 使用Markdown标题分割器
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            return_each_line=False
        )
        
        # 分割文档
        try:
            chunks = markdown_splitter.split_text(text)
            logger.debug(f"Split markdown into {len(chunks)} chunks based on headers")
        except Exception as e:
            logger.warning(f"Markdown header splitting failed: {str(e)}, falling back to token splitting")
            # 回退到token分割
            text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_text(text)
            logger.debug(f"Fallback: Split text into {len(chunks)} chunks using token splitter")
        
        # 提取问题中的关键词（包括可能的同义词和变体）
        question_keywords = self._extract_technical_keywords(question)
        
        # 计算每个块的相关性得分
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            chunk_text = chunk.page_content if hasattr(chunk, 'page_content') else str(chunk)
            score = 0
            
            # 1. 精确匹配得分
            for keyword in question_keywords:
                if keyword.lower() in chunk_text.lower():
                    score += 5  # 精确匹配得分较高
            
            # 2. 技术术语邻近度得分
            if self._check_technical_term_proximity(chunk_text, question_keywords):
                score += 3
            
            # 3. 标题级别得分（高级别标题更重要）
            if hasattr(chunk, 'metadata'):
                header_level = len(chunk.metadata.get('Header 1', '')) + \
                            len(chunk.metadata.get('Header 2', '')) * 0.8 + \
                            len(chunk.metadata.get('Header 3', '')) * 0.6 + \
                            len(chunk.metadata.get('Header 4', '')) * 0.4
                score += header_level
            
            chunk_scores.append((i, score, chunk_text))
        
        # 按得分排序
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择最相关的块直到达到token限制
        selected_text = []
        current_tokens = 0
        
        for idx, score, chunk_text in chunk_scores:
            chunk_tokens = count_tokens(chunk_text)
            
            if current_tokens + chunk_tokens <= max_tokens and score > 0:
                selected_text.append(chunk_text)
                current_tokens += chunk_tokens
                logger.debug(f"Added chunk {idx} with score {score}, cumulative tokens: {current_tokens}")
            elif current_tokens >= max_tokens:
                logger.debug(f"Reached token limit, stopping selection")
                break
            else:
                logger.debug(f"Skipping chunk {idx} with score {score}: would exceed token limit")
        
        # 如果没有找到相关块，回退到原始TF-IDF方法
        if not selected_text:
            logger.debug("No relevant chunks found, falling back to TF-IDF method")
            return self._fallback_extractive_compression(text, question, max_tokens)
        
        result = "\n\n".join(selected_text)
        logger.debug(f"Extractive compression completed. Result length: {len(result)}")
        return result

    def _extract_technical_keywords(self, text: str) -> list:
        """提取技术关键词，包括可能的同义词和变体"""
        # 基本关键词提取
        base_keywords = self._extract_keywords(text)
        
        # 技术术语映射表（可以根据需要扩展）
        technical_synonyms = {
            "intentupdateprocess": ["意向变更", "意向变更流程", "IntentUpdateProcess"],
            "意向变更": ["IntentUpdateProcess", "意向变更流程"],
            "意向变更流程": ["IntentUpdateProcess", "意向变更"],
            "模型": ["表", "数据表", "资产模型", "DWA"],
            "表": ["模型", "数据表", "资产模型"],
        }
        
        # 扩展关键词
        expanded_keywords = set(base_keywords)
        for keyword in base_keywords:
            lower_keyword = keyword.lower()
            if lower_keyword in technical_synonyms:
                expanded_keywords.update(technical_synonyms[lower_keyword])
        
        return list(expanded_keywords)

    def _check_technical_term_proximity(self, text: str, keywords: list) -> bool:
        """检查技术术语是否在相近的位置出现"""
        # 找到所有关键词在文本中的位置
        positions = []
        for keyword in keywords:
            pos = text.lower().find(keyword.lower())
            if pos != -1:
                positions.append(pos)
        
        # 如果至少有两个关键词，检查它们是否在合理距离内
        if len(positions) >= 2:
            positions.sort()
            # 检查相邻关键词的距离
            for i in range(len(positions) - 1):
                if positions[i+1] - positions[i] < 200:  # 200字符内的距离认为是相近的
                    return True
        
        return False

    def _fallback_extractive_compression(self, text: str, question: str, max_tokens: int) -> str:
        """回退到原始的TF-IDF方法"""
        # 使用文本分割器将文本分成块
        text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        
        # 使用TF-IDF找到与问题最相关的文本块
        vectorizer = TfidfVectorizer().fit_transform([question] + chunks)
        vectors = vectorizer.toarray()
        question_vector = vectors[0]
        chunk_vectors = vectors[1:]
        
        # 计算余弦相似度
        similarities = cosine_similarity([question_vector], chunk_vectors)[0]
        
        # 选择最相关的块直到达到token限制
        selected_indices = np.argsort(similarities)[::-1]
        selected_text = []
        current_tokens = 0
        
        for idx in selected_indices:
            chunk = chunks[idx]
            chunk_tokens = count_tokens(chunk)
            
            if current_tokens + chunk_tokens <= max_tokens:
                selected_text.append(chunk)
                current_tokens += chunk_tokens
            else:
                break
        
        return "\n".join(selected_text)    
    def _abstractive_compression(self, text: str, question: str, max_tokens: int) -> str:
        """生成式摘要压缩"""
        logger.debug("Starting abstractive compression")
        logger.debug(f"Input text length: {len(text)}, question: {question}")
        logger.debug(f"Max tokens: {max_tokens}")
        
        # 使用LLM生成摘要
        prompt_template = """请根据以下问题和上下文，生成一个简洁的摘要，保留所有关键信息。

问题: {question}

上下文: {text}

摘要:"""
        
        logger.debug(f"Prompt template: {prompt_template}")
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["question", "text"]
        )
        
        # 如果文本太长，需要分块处理
        text_tokens = count_tokens(text)
        logger.debug(f"Text token count: {text_tokens}")
        
        if text_tokens > 3000:  # 假设模型最大输入为4k
            logger.debug("Text too long for single processing, splitting into chunks")
            text_splitter = TokenTextSplitter(chunk_size=3000, chunk_overlap=200)
            chunks = text_splitter.split_text(text)
            logger.debug(f"Split into {len(chunks)} chunks")
            
            summaries = []
            
            for i, chunk in enumerate(chunks):
                logger.debug(f"Processing chunk {i+1}/{len(chunks)} with {len(chunk)} characters")
                try:
                    # 格式化提示词为字符串
                    formatted_prompt = prompt.format(question=question, text=chunk)
                    logger.debug(f"Formatted prompt length: {len(formatted_prompt)}")
                    logger.debug(f"Formatted prompt type: {type(formatted_prompt)}")
                    
                    # 额外检查确保格式化后的提示词是字符串
                    if not isinstance(formatted_prompt, str):
                        logger.error(f"Formatted prompt is not a string! Type: {type(formatted_prompt)}, Value: {formatted_prompt}")
                        raise ValueError(f"Formatted prompt is not a string! Type: {type(formatted_prompt)}")
                    
                    # 使用Chat模型调用方式
                    messages = [HumanMessage(content=formatted_prompt)]
                    summary = self.llm.invoke(messages)
                    summary = summary.content
                    summaries.append(summary)
                    logger.debug(f"Chunk {i+1} summary generated: {str(summary)[:100]}...")
                    logger.debug(f"Summary token count: {count_tokens(str(summary))}")
                    
                except Exception as e:
                    logger.error(f"Error processing chunk {i+1}: {str(e)}")
                    raise
            
            # 如果摘要仍然太长，递归压缩
            combined_summary = "\n".join(summaries)
            combined_tokens = count_tokens(combined_summary)
            logger.debug(f"Combined summary token count: {combined_tokens}")
            if combined_tokens > max_tokens:
                logger.debug("Combined summary too long, recursively compressing")
                return self._abstractive_compression(combined_summary, question, max_tokens)
            
            logger.debug("Abstractive compression completed")
            return combined_summary
        else:
            logger.debug("Text within limit, processing directly")
            try:
                # 格式化提示词为字符串
                formatted_prompt = prompt.format(question=question, text=text)
                logger.debug(f"Formatted prompt length: {len(formatted_prompt)}")
                logger.debug(f"Formatted prompt type: {type(formatted_prompt)}")
                
                # 额外检查确保格式化后的提示词是字符串
                if not isinstance(formatted_prompt, str):
                    logger.error(f"Formatted prompt is not a string! Type: {type(formatted_prompt)}, Value: {formatted_prompt}")
                    raise ValueError(f"Formatted prompt is not a string! Type: {type(formatted_prompt)}")
                
                message = HumanMessage(content=formatted_prompt)
                response = self.llm.invoke([message])
                result = response.content
                logger.debug(f"Direct result: {str(result)[:100]}...")
                return result
            except Exception as e:
                logger.error(f"Error in direct processing: {str(e)}")
                raise
    
    def _embedding_based_compression(self, text: str, question: str, max_tokens: int) -> str:
        """基于嵌入的压缩"""
        logger.debug("Starting embedding-based compression")
        logger.debug(f"Input text length: {len(text)}, question: {question}")
        logger.debug(f"Max tokens: {max_tokens}")
        
        # 将文本分成句子
        sentences = text.split('. ')
        logger.debug(f"Split text into {len(sentences)} sentences")
        question_embedding = self.embeddings.embed_query(question)
        sentence_embeddings = self.embeddings.embed_documents(sentences)
        
        # 计算每个句子与问题的相似度
        similarities = cosine_similarity([question_embedding], sentence_embeddings)[0]
        
        # 选择最相关的句子
        selected_indices = np.argsort(similarities)[::-1]
        selected_sentences = []
        current_tokens = 0
        
        for idx in selected_indices:
            sentence = sentences[idx]
            sentence_tokens = count_tokens(sentence)
            
            if current_tokens + sentence_tokens <= max_tokens:
                selected_sentences.append(sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        result = ". ".join(selected_sentences)
        logger.debug(f"Embedding-based compression completed. Result length: {len(result)}")
        return result
    
    def _keyword_based_compression(self, text: str, question: str, max_tokens: int) -> str:
        """基于关键词的压缩"""
        logger.debug("Starting keyword-based compression")
        # 提取问题中的关键词
        question_keywords = self._extract_keywords(question)
        logger.debug(f"Question keywords: {question_keywords}")
        
        # 将文本分成句子
        sentences = text.split('. ')
        logger.debug(f"Split text into {len(sentences)} sentences")
        selected_sentences = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_keywords = self._extract_keywords(sentence)
            # 计算句子与问题的关键词重叠
            overlap = len(set(question_keywords) & set(sentence_keywords))
            logger.debug(f"Sentence keywords: {sentence_keywords}, overlap: {overlap}")
            
            if overlap > 0 and current_tokens + count_tokens(sentence) <= max_tokens:
                selected_sentences.append(sentence)
                current_tokens += count_tokens(sentence)
        
        result = ". ".join(selected_sentences)
        logger.debug(f"Keyword-based compression completed. Result length: {len(result)}")
        return result
    
    def _extract_keywords(self, text: str) -> list:
        """提取文本中的关键词"""
        logger.debug(f"Extracting keywords from text: {text[:100]}...")
        # 使用TF-IDF提取关键词
        vectorizer = TfidfVectorizer(max_features=10, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            result = feature_names.tolist()
            logger.debug(f"Extracted keywords: {result}")
            return result
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []