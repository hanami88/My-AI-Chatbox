import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    DirectoryLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from config import CUSTOM_SYSTEM_PROMPT, RAG_PROMPT_TEMPLATE, MODEL_CONFIG, VECTOR_CONFIG

load_dotenv()


class RAGChatbot:
    def __init__(self):
        self.llm = None
        self.vectorstore = None
        self.qa_chain = None
        self.setup_components()

    def setup_components(self):
        """Khởi tạo các component của RAG"""
        print("🔄 Đang khởi tạo AI chatbot...")

        # 1. Khởi tạo LLM
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            **MODEL_CONFIG
        )

        # 2. Load documents
        self.load_documents()

        # 3. Setup retrieval QA chain
        self.setup_qa_chain()

        print("✅ AI chatbot đã sẵn sàng!")

    def load_documents(self):
        """Load và xử lý tài liệu"""
        print("📚 Đang đọc tài liệu...")

        documents_path = "documents"
        if not os.path.exists(documents_path):
            os.makedirs(documents_path)
            print(f"📁 Đã tạo thư mục {documents_path}")
            print("💡 Hãy thêm tài liệu (.txt, .pdf, .docx) vào thư mục này!")
            return

        # Load các loại file khác nhau
        loaders = []

        # Text files
        if any(f.endswith('.txt') for f in os.listdir(documents_path)):
            txt_loader = DirectoryLoader(
                documents_path,
                glob="*.txt",
                loader_cls=TextLoader
            )
            loaders.append(txt_loader)

        # PDF files
        pdf_files = [f for f in os.listdir(documents_path) if f.endswith('.pdf')]
        for pdf_file in pdf_files:
            pdf_loader = PyPDFLoader(os.path.join(documents_path, pdf_file))
            loaders.append(pdf_loader)

        # DOCX files
        docx_files = [f for f in os.listdir(documents_path) if f.endswith('.docx')]
        for docx_file in docx_files:
            docx_loader = Docx2txtLoader(os.path.join(documents_path, docx_file))
            loaders.append(docx_loader)

        # Load tất cả documents
        all_documents = []
        for loader in loaders:
            try:
                docs = loader.load()
                all_documents.extend(docs)
            except Exception as e:
                print(f"❌ Lỗi khi đọc tài liệu: {e}")

        if not all_documents:
            print("⚠️ Không tìm thấy tài liệu nào!")
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

        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory="vector_store"
        )

        print(f"✅ Đã đọc và xử lý {len(texts)} đoạn văn từ {len(all_documents)} tài liệu")

    def setup_qa_chain(self):
        """Setup QA chain với custom prompt"""
        if self.vectorstore is None:
            print("⚠️ Chưa có vector store, sử dụng mode chat thường")
            return

        # Custom prompt
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=RAG_PROMPT_TEMPLATE
        )

        # Tạo retrieval QA chain
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
            # Sử dụng RAG
            response = self.qa_chain({"query": question})
            return {
                "answer": response["result"],
                "sources": response["source_documents"]
            }
        else:
            # Chat thường
            system_message = f"{CUSTOM_SYSTEM_PROMPT}\n\nCâu hỏi: {question}"
            response = self.llm.invoke(system_message)
            return {
                "answer": response.content,
                "sources": []
            }

    def run_console_chat(self):
        """Chạy chat console"""
        print("\n🤖 MyAI đã sẵn sàng! Gõ 'thoat' để kết thúc.\n")

        while True:
            user_input = input("👤 Bạn: ")

            if user_input.lower() in ['thoat', 'exit', 'quit', 'bye']:
                print("🤖 MyAI: Tạm biệt! Hẹn gặp lại bạn!")
                break

            try:
                print("🔍 Đang tìm kiếm thông tin...")
                result = self.chat(user_input)

                print(f"\n🤖 MyAI: {result['answer']}")

                # Hiển thị sources nếu có
                if result['sources']:
                    print(f"\n📚 Nguồn tham khảo: {len(result['sources'])} tài liệu")

                print("-" * 50)

            except Exception as e:
                print(f"❌ Có lỗi xảy ra: {e}")


if __name__ == "__main__":
    # Khởi tạo và chạy chatbot
    chatbot = RAGChatbot()
    chatbot.run_console_chat()