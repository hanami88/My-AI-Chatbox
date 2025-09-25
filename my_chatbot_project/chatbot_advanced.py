import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

load_dotenv()

# Khởi tạo Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-70b-versatile",
    temperature=0.7
)

# Template prompt tùy chỉnh
prompt_template = PromptTemplate(
    input_variables=["history", "input"],
    template="""Bạn là một trợ lý AI thông minh và hữu ích. Hãy trả lời bằng tiếng Việt một cách thân thiện, chi tiết và chính xác.

Lịch sử cuộc trò chuyện:
{history}

Người dùng: {input}
Trợ lý AI:"""
)

# Bộ nhớ để lưu lịch sử cuộc trò chuyện (giữ lại 10 tin nhắn gần nhất)
memory = ConversationBufferWindowMemory(k=10)

# Tạo chain conversation
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt_template,
    verbose=False
)


def chatbot_nang_cao():
    print("🤖 Chatbot nâng cao đã sẵn sàng! (có bộ nhớ)")
    print("📝 Tôi sẽ nhớ cuộc trò chuyện của chúng ta.")
    print("💡 Gõ 'thoat' để kết thúc.\n")

    while True:
        user_input = input("👤 Bạn: ")

        if user_input.lower() in ['thoat', 'exit', 'bye', 'quit']:
            print("🤖 Tạm biệt! Rất vui được trò chuyện với bạn!")
            break

        try:
            # Lấy phản hồi từ conversation chain
            response = conversation.predict(input=user_input)
            print(f"🤖 Bot: {response}\n")

        except Exception as e:
            print(f"❌ Có lỗi xảy ra: {e}\n")


if __name__ == "__main__":
    chatbot_nang_cao()