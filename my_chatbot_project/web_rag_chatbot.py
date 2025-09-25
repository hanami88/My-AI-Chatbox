import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import CUSTOM_SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE, MODEL_CONFIG, VECTOR_CONFIG

load_dotenv()


class WebRAGChatbot:
    def __init__(self):
        self.llm = None
        self.vectorstore = None
        self.qa_chain = None
        self.documents_path = "documents"
        self.vector_path = "vector_store"
        self.setup_directories()
        self.setup_llm()

    def setup_directories(self):
        """Tạo thư mục nếu chưa có"""
        os.makedirs(self.documents_path, exist_ok=True)
        os.makedirs(self.vector_path, exist_ok=True)

    def setup_llm(self):
        """Khởi tạo LLM"""
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            **MODEL_CONFIG
        )

    def save_uploaded_file(self, uploaded_file):
        """Lưu file được upload"""
        try:
            file_path = os.path.join(self.documents_path, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return file_path, True
        except Exception as e:
            st.error(f"Lỗi lưu file: {e}")
            return None, False

    def load_single_document(self, file_path):
        """Load một tài liệu"""
        try:
            if file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            else:
                return []

            return loader.load()
        except Exception as e:
            st.error(f"Lỗi đọc file {file_path}: {e}")
            return []

    def process_documents(self, force_reload=False):
        """Xử lý tất cả documents"""
        if not force_reload and self.vectorstore is not None:
            return

        # Load tất cả documents
        all_documents = []
        doc_files = [f for f in os.listdir(self.documents_path)
                     if f.endswith(('.txt', '.pdf', '.docx'))]

        if not doc_files:
            st.warning("⚠️ Không có tài liệu nào trong thư mục documents/")
            return

        for doc_file in doc_files:
            file_path = os.path.join(self.documents_path, doc_file)
            docs = self.load_single_document(file_path)
            all_documents.extend(docs)

        if not all_documents:
            return

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=VECTOR_CONFIG["chunk_size"],
            chunk_overlap=VECTOR_CONFIG["chunk_overlap"]
        )
        texts = text_splitter.split_documents(all_documents)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Xóa vector store cũ nếu có
        if os.path.exists(self.vector_path):
            shutil.rmtree(self.vector_path)

        # Create vector store mới
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=self.vector_path
        )

        # Setup QA chain
        self.setup_qa_chain()

        st.success(f"✅ Đã xử lý {len(texts)} đoạn văn từ {len(all_documents)} tài liệu!")

    def setup_qa_chain(self):
        """Setup QA chain"""
        if self.vectorstore is None:
            return

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=RAG_PROMPT_TEMPLATE
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={"k": VECTOR_CONFIG["k_retrieved_docs"]}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def chat(self, question):
        """Chat với AI"""
        if self.qa_chain:
            response = self.qa_chain({"query": question})
            return {
                "answer": response["result"],
                "sources": response["source_documents"]
            }
        else:
            system_message = f"{CUSTOM_SYSTEM_PROMPT}\n\nCâu hỏi: {question}"
            response = self.llm.invoke(system_message)
            return {
                "answer": response.content,
                "sources": []
            }

    def get_document_list(self):
        """Lấy danh sách tài liệu"""
        if not os.path.exists(self.documents_path):
            return []
        return [f for f in os.listdir(self.documents_path)
                if f.endswith(('.txt', '.pdf', '.docx'))]

    def delete_document(self, filename):
        """Xóa tài liệu"""
        try:
            file_path = os.path.join(self.documents_path, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
        except Exception as e:
            st.error(f"Lỗi xóa file: {e}")
        return False


# Streamlit App
st.set_page_config(
    page_title="MyAI - RAG Chatbot với Upload",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
.upload-section {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    border: 2px dashed #dee2e6;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🤖 MyAI - Trợ Lý Thông Minh của Luân</h1>
    <p>Upload tài liệu và chat với AI ngay lập tức!</p>
</div>
""", unsafe_allow_html=True)


# Initialize chatbot
@st.cache_resource
def get_chatbot():
    return WebRAGChatbot()


if "chatbot" not in st.session_state:
    st.session_state.chatbot = get_chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Xin chào! Tôi là MyAI. Hãy upload tài liệu hoặc hỏi tôi bất cứ điều gì!"
    })

# Sidebar - Document Management
with st.sidebar:
    st.header("📁 Quản Lý Tài Liệu")

    # File Upload Section
    st.subheader("📤 Upload Tài Liệu Mới")
    uploaded_files = st.file_uploader(
        "Chọn file để upload:",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx'],
        help="Hỗ trợ: .txt, .pdf, .docx"
    )

    if uploaded_files:
        success_count = 0
        for uploaded_file in uploaded_files:
            file_path, success = st.session_state.chatbot.save_uploaded_file(uploaded_file)
            if success:
                success_count += 1
                st.success(f"✅ Đã lưu: {uploaded_file.name}")

        if success_count > 0:
            if st.button("🔄 Xử Lý Tài Liệu Mới", type="primary"):
                with st.spinner("Đang xử lý tài liệu..."):
                    st.session_state.chatbot.process_documents(force_reload=True)
                st.rerun()

    st.divider()

    # Document List
    st.subheader("📚 Tài Liệu Hiện Có")
    doc_list = st.session_state.chatbot.get_document_list()

    if doc_list:
        for doc in doc_list:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"📄 {doc}")
            with col2:
                if st.button("🗑️", key=f"delete_{doc}", help="Xóa file"):
                    if st.session_state.chatbot.delete_document(doc):
                        st.success(f"Đã xóa {doc}")
                        st.rerun()

        st.info(f"💡 Tổng: {len(doc_list)} tài liệu")

        if st.button("🔄 Tải Lại Tài Liệu"):
            with st.spinner("Đang tải lại..."):
                st.session_state.chatbot.process_documents(force_reload=True)
            st.rerun()
    else:
        st.warning("Chưa có tài liệu nào")

    st.divider()

    # Stats
    st.subheader("📊 Thống Kê")
    if hasattr(st.session_state.chatbot, 'vectorstore') and st.session_state.chatbot.vectorstore:
        try:
            doc_count = st.session_state.chatbot.vectorstore._collection.count()
            st.metric("Đoạn văn đã học", doc_count)
            st.success("✅ AI đã sẵn sàng!")
        except:
            st.info("🔄 Đang khởi tạo...")
    else:
        st.warning("⚠️ Chưa xử lý tài liệu")

    if st.button("🗑️ Xóa Lịch Sử Chat"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

# Main Chat Interface
st.subheader("💬 Chat với MyAI")

# Auto-process documents on first load
if not hasattr(st.session_state, 'docs_processed'):
    if st.session_state.chatbot.get_document_list():
        with st.spinner("🔄 Đang xử lý tài liệu có sẵn..."):
            st.session_state.chatbot.process_documents()
        st.session_state.docs_processed = True

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander(f"📚 Nguồn tham khảo ({len(message['sources'])} tài liệu)"):
                for i, source in enumerate(message["sources"], 1):
                    st.write(f"**Nguồn {i}:**")
                    st.write(source.page_content[:200] + "...")

# Chat input
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("🤔 MyAI đang suy nghĩ..."):
            try:
                result = st.session_state.chatbot.chat(prompt)

                st.write(result["answer"])

                if result["sources"]:
                    with st.expander(f"📚 Nguồn tham khảo ({len(result['sources'])} tài liệu)"):
                        for i, source in enumerate(result["sources"], 1):
                            st.write(f"**Nguồn {i}:**")
                            st.write(source.page_content[:200] + "...")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })

            except Exception as e:
                error_msg = f"❌ Có lỗi xảy ra: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Instructions
with st.expander("💡 Hướng Dẫn Sử Dụng"):
    st.markdown("""
    **Cách thêm tài liệu:**

    1. **Upload trực tiếp:** Dùng file uploader bên trái
    2. **Copy file:** Thả file vào thư mục `documents/` và nhấn "Tải Lại"

    **Các định dạng hỗ trợ:**
    - 📄 Text files (.txt)
    - 📕 PDF files (.pdf)
    - 📘 Word files (.docx)

    **Tips:**
    - Upload xong nhớ nhấn "Xử Lý Tài Liệu Mới"
    - AI sẽ trả lời dựa trên nội dung tài liệu
    - Có thể xóa tài liệu không cần thiết để tăng độ chính xác
    """)