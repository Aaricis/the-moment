from FlagEmbedding import FlagReranker
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.configs.base_config import device
from src.configs.rag_config import embedding_model_dir, vector_db_dir, rerank_model_dir


class EmoLLMRAG:
    def __init__(self, retrieval_num=3, rerank_flag=True, select_num=3) -> None:
        self.retrieval_num = retrieval_num
        self.rerank_flag = rerank_flag
        self.select_num = select_num
        self.embedding_model_dir = embedding_model_dir
        self.rerank_model_dir = rerank_model_dir

    def load_vector_db(self):
        """
        加载向量数据库
        :return: db
        """
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_dir,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        db = FAISS.load_local(
            vector_db_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return db

    def load_rerank_model(self):
        # 指定本地路径
        ranker = FlagReranker(
            self.rerank_model_dir,
            use_fp16=True  # 速度更快，占显存少
        )
        return ranker

    def get_retrieval_content(self, query) -> str:
        db = self.load_vector_db()
        documents = db.similarity_search(query, k=self.retrieval_num)

        content = []
        for document in documents:
            content.append(document.page_content)

        if self.rerank_flag:
            # 加载rerank模型
            ranker = self.load_rerank_model()

            # 构造 query-doc 对
            pairs = [[query, doc.page_content] for doc in documents]

            # 计算相关性得分
            scores = ranker.compute_score(pairs)

            # 把文档和得分打包在一起
            doc_scores = list(zip(documents, scores))

            # 按得分排序（高分在前）
            doc_scores = sorted(doc_scores, key=lambda x: x[1], reverse=True)

            # 取排序后的文档内容
            content = [doc.page_content for doc, _ in doc_scores[:self.select_num]]

        # 拼接成一个上下文字符串返回
        return "\n".join(content)
