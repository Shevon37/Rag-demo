# rag-retrieval

RAG向量召回示例

## 依赖

先安装适合自己环境的torch，再安装项目依赖:

```
pip install langchain pypdf rapidocr-onnxruntime modelscope transformers faiss-cpu tiktoken -i https://mirrors.aliyun.com/pypi/simple/
```

如果有量化相关的报错，从源码安装vllm-gpts、auto-gtpq

## 用法

1、使用本地部署vllm、ollama或使用服务商提供api接入llm

```

2、运行indexer.py，解析pdf生成向量库

```
python indexer.py
```

3、运行rag.py，开始体验RAG增强检索

```
python rag.py
```
