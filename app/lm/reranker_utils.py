from app.conf.reranker_config import reranker_config
import os
from dashscope import TextReRank


class DashScopeReranker:
    """DashScope Rerank API 封装，兼容原有 FlagReranker 的 compute_score 接口。"""

    def __init__(self, api_key=None, model="gte-rerank-v2"):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model = model
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY 未配置，请在 .env 中添加")

    def compute_score(self, sentence_pairs):
        """
        兼容原有接口：接收 [(query, passage), ...] 格式，
        返回 [score, ...] 分数列表（与输入一一对应）。
        """
        if not sentence_pairs:
            return []

        query = sentence_pairs[0][0]
        documents = [pair[1] for pair in sentence_pairs]

        response = TextReRank.call(
            model=self.model,
            query=query,
            documents=documents,
            top_n=len(documents),
            api_key=self.api_key,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope Rerank API 调用失败: {response.status_code} - {response.message}"
            )

        # API 返回按 relevance_score 降序，需映射回原始顺序
        scores = [0.0] * len(documents)
        for result in response.output.results:
            scores[result.index] = result.relevance_score

        return scores


_reranker_model = None

def get_reranker_model():
    """加载重排序模型：优先 DashScope API，其次本地模型，都不可用返回 None"""
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model

    # 方案1：DashScope Rerank API（无需本地模型）
    dashscope_key = os.getenv("DASHSCOPE_API_KEY")
    if dashscope_key:
        print("[Reranker] 使用 DashScope Rerank API (gte-rerank-v2)")
        _reranker_model = DashScopeReranker(api_key=dashscope_key)
        return _reranker_model

    # 方案2：本地 FlagReranker
    model_path = reranker_config.bge_reranker_large
    if model_path and os.path.isdir(str(model_path).replace("\\", "/")):
        from FlagEmbedding import FlagReranker
        print(f"[Reranker] 使用本地模型: {model_path}")
        _reranker_model = FlagReranker(
            model_name_or_path=model_path,
            device=reranker_config.bge_reranker_device,
            use_fp16=reranker_config.bge_reranker_fp16
        )
        return _reranker_model

    print("[Reranker] 未配置可用的 Reranker，重排序将跳过")
    return None