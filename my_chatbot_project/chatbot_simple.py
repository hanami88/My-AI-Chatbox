import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Load biến môi trường
load_dotenv()

# Khởi tạo Groq LLM
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-70b-versatile",  # hoặc "mixtral-8x7b-32768"
    temperature=0.7,
    max_tokens=1024
)


def chatbot_don_gian():
    print("🤖 Chatbot đã khởi động! Gõ 'thoat' để kết thúc.")

    # System message để định hướng AI
    messages = [
        SystemMessage(
            content="Bạn là một trợ lý AI hữu ích. Hãy trả lời bằng tiếng Việt một cách thân thiện và ngắn gọn.")
    ]

    while True:
        user_input = input("\n👤 Bạn: ")

        if user_input.lower() in ['thoat', 'exit', 'bye', 'quit']:
            print("🤖 Tạm biệt!")
            break

        # Thêm tin nhắn của user
        messages.append(HumanMessage(content=user_input))

        # Lấy phản hồi từ AI
        try:
            response = llm.invoke(messages)
            print(f"🤖 Bot: {response.content}")

            # Thêm phản hồi AI vào lịch sử cuộc trò chuyện
            messages.append(AIMessage(content=response.content))

        except Exception as e:
            print(f"❌ Lỗi: {e}")


if __name__ == "__main__":
    chatbot_don_gian()