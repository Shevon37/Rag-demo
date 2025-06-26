from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ChatMessageHistory
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,AIMessagePromptTemplate,MessagesPlaceholder
from langchain.schema import HumanMessage,SystemMessage,AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
import os 

DASHSCOPE_API_KEY = "sk-3PardAPDOYQFDL0ji4z9bFWGi3CUCoAB5e2x8aSMro2Se0TO"

# 加载embedding模型，用于将query向量化
# embeddings=ModelScopeEmbeddings(model_id='iic/nlp_corom_sentence-embedding_chinese-base') 
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 加载faiss向量库，用于知识召回
vector_db=FAISS.load_local('LLM.faiss',embeddings,allow_dangerous_deserialization=True)  # 明确启用反序列化
retriever=vector_db.as_retriever(search_kwargs={"k":5})

# # 用vllm部署openai兼容的服务端接口，然后走ChatOpenAI客户端调用
# os.environ['VLLM_USE_MODELSCOPE']='True'
# chat=ChatOpenAI(
#     model="qwen/Qwen-7B-Chat-Int4",
#     openai_api_key="EMPTY",
#     openai_api_base='http://localhost:8000/v1',
#     stop=['<|im_end|>']
# )

# 修改 ChatOpenAI 初始化以调用通义千问 OpenAI 兼容 API，而不是采用本地大模型
chat = ChatOpenAI(
    model="claude-sonnet-4-20250514",  # 或者 "qwen-plus", "qwen-max", "qwen-max-1201" 等
                         # 请查阅阿里云 DashScope 文档获取最新的兼容模型列表和名称
    openai_api_key=DASHSCOPE_API_KEY,
    openai_api_base="https://api.qingyuntop.top/v1", # DashScope 的 OpenAI 兼容模式 API 地址
    # stop=['<|im_end|>'], # 这个 stop token 通常在调用云服务 API 时不是必需的，
                           # 模型通常已经过微调以正确结束对话。
                           # 建议先移除或注释掉，测试模型默认的停止行为。
                           # 如果发现模型输出不完整或持续输出，再查阅 DashScope 文档看是否有推荐的 stop token。
    temperature=0, # 可以根据需要设置温度等其他参数
    # max_tokens=1500, # 也可以按需设置最大输出token数
)

# Prompt模板
system_prompt=SystemMessagePromptTemplate.from_template('You are a helpful assistant.')
user_prompt=HumanMessagePromptTemplate.from_template('''
Answer the question based only on the following context:

{context}

Question: {query}
''')
full_chat_prompt=ChatPromptTemplate.from_messages([system_prompt,MessagesPlaceholder(variable_name="chat_history"),user_prompt])

'''
<|im_start|>system
You are a helpful assistant.
<|im_end|>
...
<|im_start|>user
Answer the question based only on the following context:

{context}

Question: {query}
<|im_end|>
<|im_start|>assitant
......
<|im_end|>
'''

# Chat chain
chat_chain={
        "context": itemgetter("query") | retriever,
        "query": itemgetter("query"),
        "chat_history":itemgetter("chat_history"),
    }|full_chat_prompt|chat

# 开始对话
chat_history=[]
while True:
    query=input('query:')
    response=chat_chain.invoke({'query':query,'chat_history':chat_history})
    chat_history.extend((HumanMessage(content=query),response))
    print(response.content)
    chat_history=chat_history[-20:] # 最新10轮对话