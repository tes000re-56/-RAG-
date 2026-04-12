import streamlit as st
import os
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader,PyPDFLoader,JSONLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from FlagEmbedding import FlagReranker
from rank_bm25 import BM25Okapi
import jieba




SAVED_FILES = "saved_files.txt"




def load_saved_files():
    if os.path.exists(SAVED_FILES):
        with open(SAVED_FILES, "r", encoding="utf-8") as f:
            return f.read().splitlines()
    return []


def save_to_chroma(uploaded_files):
    if not os.path.exists("uploads"):
        os.mkdir("uploads")

    vector_stores = Chroma(
        collection_name="my_chroma",#文件夹名字
        embedding_function=OllamaEmbeddings (model="nomic-embed-text:latest"),#模型
        persist_directory="./chroma_db",#路径
    )

    saved=load_saved_files()
    # 保存文件到硬盘
    file_path = os.path.join("uploads", uploaded_files.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_files.getbuffer())

    if file_path in saved:
        st.warning("⚠️ 该文件已上传")
        return
    if file_path:
        ext =file_path.split(".")[-1].lower()
        if ext=="csv":
            loader = CSVLoader(file_path=file_path,
                               encoding="utf-8",
                               csv_args={
                "delimiter": "\n",
                "quotechar": '"',
                }
            )
            document = loader.load()
            splitter = RecursiveCharacterTextSplitter(  # 分割文本
                chunk_size=256,  # 最多字数
                chunk_overlap=50,  # 重叠字数
                separators=["\n\n", "\n", " ", ""],  # 怎么分
                length_function=len  # 分割标准
            )
            documents = splitter.split_documents(document)

        elif ext=="txt":
            loader = TextLoader(
                file_path=file_path,
                encoding="utf-8",
            )
            document = loader.load()
            splitter=RecursiveCharacterTextSplitter(#分割文本
                chunk_size=256,#最多字数
                chunk_overlap=50,#重叠字数
                separators=["\n\n", "\n", " ", ""],#怎么分
                length_function=len#分割标准
            )
            documents = splitter.split_documents(document)

        elif ext=="pdf":
            loader = PyPDFLoader(
                file_path=file_path,
                mode="page"
            )
            document = loader.load()
            splitter = RecursiveCharacterTextSplitter(  # 分割文本
                chunk_size=256,  # 最多字数
                chunk_overlap=70,  # 重叠字数
                separators=["\n\n", "\n", " ", ""],  # 怎么分
                length_function=len  # 分割标准
            )
            documents = splitter.split_documents(document)

        elif ext=="json":
            loader = JSONLoader(
                file_path=file_path,
                jq_schema=".",
                text_content=False
            )
            document = loader.load()
            splitter = RecursiveCharacterTextSplitter(  # 分割文本
                chunk_size=256,  # 最多字数
                chunk_overlap=50,  # 重叠字数
                separators=["\n\n", "\n", " ", ""],  # 怎么分
                length_function=len  # 分割标准
            )
            documents = splitter.split_documents(document)

        else:
            st.error("不支持的文件格式")
            return


    vector_stores.add_documents(documents)

    if file_path not in saved:
        with open(SAVED_FILES, "a", encoding="utf-8") as f:
            f.write(file_path + "\n")



def find_from_vector_store(user_input):
    vector_stores = Chroma(
        collection_name="my_chroma",  # 文件夹名字
        embedding_function=OllamaEmbeddings (model="nomic-embed-text:latest"),  # 模型
        persist_directory="./chroma_db",  # 路径
    )

    reranker = FlagReranker(#重排序模型
        "BAAI/bge-reranker-base",
        use_fp16=True,  # 有 GPU 建议开启，加速
        device="cuda"  # 或 "cpu"
    )

    doc_list =vector_stores.get()["documents"]
    tokenized_corpus = [jieba.lcut(text) for text in doc_list]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_texts = bm25.get_top_n(jieba.lcut(user_input), doc_list, 5)#关键词检索

    result = vector_stores.similarity_search(user_input, 5)
    vector_texts=[doc.page_content for doc in result]#向量检索

    real_result = list(set(vector_texts + bm25_texts))#结合+去重

    pairs = [[user_input, doc] for doc in real_result]#固定格式，用户问题和召回内容

    scores = reranker.compute_score(pairs)#打分

    ranked = sorted(zip(scores, real_result), key=lambda x: x[0], reverse=True)#reverse=True从大到小排序，key=lambda x: x[0]分数，zip(scores, real_result)分数和内容拼成列表
    # for i, (score, doc) in enumerate(ranked, 1):#遍历

    context="\n".join(doc for score, doc in ranked[:3])
    return context


def use_ai(user_input, context,history):

    client = OpenAI(
        api_key="nokey",
        # api_key=apikey,
        # base_url="https://api.deepseek.com")
        base_url = "http://localhost:11434/v1")

    prompt = f"""
    参考下面知识库资料回答用户问题：
    【知识库资料】
    {context}

    【用户问题】
    {user_input}
    【历史消息】
    {history}
    只根据上面资料回答，不要瞎编。
    """

    response = client.chat.completions.create(
        model="deepseek-r1:1.5b",
        # model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是本地知识库ai助手，只按照知识库回答"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    print(context)
    return (response.choices[0].message.content)



# -------------------------- 页面配置 --------------------------
st.set_page_config(
    page_title="本地知识问答",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)


with st.sidebar:
    st.button("🗑 清空对话", width="stretch",on_click=lambda: st.session_state.clear())


    #上传文档
    uploaded_files = st.file_uploader(
        "上传 PDF / TXT / CSV / JSON文件",
        type=["pdf", "txt","csv", "json"],  # 允许的格式
        accept_multiple_files=True  # 可以一次传多个
    )


    # 如果有文件上传
    if uploaded_files :
        for one_file in uploaded_files:
            save_to_chroma(one_file)

    # apikey = st.text_area("请输入deepseek的apikey")


# -------------------------- 初始化会话状态（必须有） --------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "你好！我是你的本地知识问答助手，请上传文档后向我提问～"}
    ]

# -------------------------- 页面标题 --------------------------
st.title("📚 本地知识问答系统")


st.markdown("---")

# -------------------------- 显示聊天历史 --------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------- 用户输入框 --------------------------
user_input = st.chat_input("请输入你的问题...")


# -------------------------- 处理用户提问 --------------------------
history = []
if user_input:
    # 1. 把用户消息加入历史
    # history = []
    st.session_state.messages.append({"role": "user", "content": user_input})
    # history.append({"role": "user", "content": user_input})
    history = st.session_state.messages[-10:]

    # 2. 前端显示用户消息
    with st.chat_message("user"):
        st.markdown(user_input)

    # 3. 显示 AI 正在思考的状态
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # ===================== 在这里写你的核心逻辑 =====================
        # 你可以在这里调用：向量库检索 + 大模型生成答案
        with st.spinner("正在检索知识库..."):
            # ========== 你的代码位置 ==========
            # 示例回答（替换成你的真实逻辑）
            found=find_from_vector_store(user_input)
            full_response=use_ai(user_input,found,history)
            # full_response = f"我收到了你的问题：{user_input}\n\n这里将返回本地知识库的答案。"
            # ============================================================

        # 4. 显示最终答案
        message_placeholder.markdown(full_response)
        print(type(full_response))

    # 5. 把 AI 回答存入聊天历史
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    # history=st.session_state.messages [-5:]


# -------------------------- 右侧清空按钮 --------------------------
# with st.sidebar:
#     st.button("🗑 清空对话", width="stretch",on_click=lambda: st.session_state.clear())
#
#
#     #上传文档
#     uploaded_files = st.file_uploader(
#         "上传 PDF / TXT / CSV / JSON文件",
#         type=["pdf", "txt","csv", "json"],  # 允许的格式
#         accept_multiple_files=True  # 可以一次传多个
#     )
#
#
#     # 如果有文件上传
#     if uploaded_files :
#         for one_file in uploaded_files:
#             save_to_chroma(one_file)
#
#     apikey = st.text_area("请输入deepseek的apikey")



