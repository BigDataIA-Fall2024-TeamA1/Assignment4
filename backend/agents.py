import os
from dotenv import load_dotenv
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools.base import BaseTool
from langchain.chains import RetrievalQA
from langchain.utilities import SerpAPIWrapper
from langchain.tools import ArxivQueryRun
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore

# 加载环境变量
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# 初始化 Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# 连接到已有的 Pinecone 索引
index = pinecone.Index(PINECONE_INDEX_NAME)

# 初始化嵌入模型和向量存储
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings.embed_query,
    text_key="text"
)

# 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 创建 RetrievalQA 链
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo"
    ),
    chain_type="stuff",
    retriever=retriever
)

# 定义 Academic QA Tool
class AcademicQATool(BaseTool):
    name = "Academic QA System"
    description = "使用文档内容回答学术问题。输入应为问题字符串。"

    def _run(self, query: str):
        return qa_chain.run(query)

    async def _arun(self, query: str):
        raise NotImplementedError("此工具不支持异步运行")

# 初始化 Arxiv Search Tool
arxiv_tool = ArxivQueryRun()
arxiv_search_tool = Tool(
    name="Arxiv Search",
    func=arxiv_tool.run,
    description="在 Arxiv 上搜索相关的研究论文。输入应为搜索查询字符串。"
)

# 初始化 Web Search Tool
search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
web_search_tool = Tool(
    name="Web Search",
    func=search.run,
    description="进行在线搜索以获取更广泛的背景信息。输入应为搜索查询字符串。"
)

# 定义工具列表
tools = [
    AcademicQATool(),
    arxiv_search_tool,
    web_search_tool
]

# 初始化代理
agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-3.5-turbo"
    ),
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True
)

# 定义问答函数
def ask_question(question):
    response = agent.run(question)
    return response
