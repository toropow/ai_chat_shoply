from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
import json
from dotenv import load_dotenv
from uuid import uuid4

import os
import logging

load_dotenv()

MODEL = os.getenv("OLLAMA_MODEL")
SESSION_ID = str(uuid4())
NUM_MESSAGE_HISTORY = os.getenv("NUM_MESSAGE_HISTORY")
FAQ = os.path.join("data", "faq.json")
ORDERS = os.path.join("data", "order.json")

logging.basicConfig(
    filename=f"logs/session_{SESSION_ID}.json",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8"
)


def load_json(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def prepare_faq() -> str:
    return "\n".join([f"Вопрос: {item['q']} Ответ: {item['a']}" for item in load_json(FAQ) ])


# Создаём класс для CLI-бота
class ClicBot:
    def __init__(self):
        # Создаём модель
        self.chat_model = ChatOllama(model=MODEL, temperature=0)

        # Создаём Хранилище истории
        self.store = {}

        # Создаем шаблон промпта
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"Ты полезный и вежливый ассистент. Отвечай кратко и по делу. Для ответов на вопросы пользователя используй официальный faq {prepare_faq()}"),
                MessagesPlaceholder(variable_name="history", n_messages=NUM_MESSAGE_HISTORY),
                ("human", "{question}"),
            ]
        )

        # Создаём цепочку (тут используется синтаксис LCEL*)
        self.chain = self.prompt | self.chat_model

        # Создаём цепочку с историей
        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,  # Цепочка с историей
            self.get_session_history,  # метод для получения истории
            input_messages_key="question",  # ключ для вопроса
            history_messages_key="history",  # ключ для истории
        )

    # Метод для получения истории по session_id
    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def __call__(self, session_id):
        logging.info("=== New session ===")
        while True:
            try:
                user_text = input("Вы: ").strip()
                logging.info(f"User: {user_text}")
            except (KeyboardInterrupt, EOFError):
                print("\nБот: Завершение работы.")
                break
            if not user_text:
                continue

            msg = user_text.lower()
            if msg in ("выход", "стоп", "конец"):
                print("Бот: До свидания!")
                break
            if msg == "сброс":
                if session_id in self.store:
                    del self.store[session_id]
                print("Бот: Контекст диалога очищен.")
                continue

            response = self.chain_with_history.invoke(
                {"question": user_text}, {"configurable": {"session_id": session_id}}
            )

            answer = response.content.strip()
            tokens = response.usage_metadata
            print("Бот: ", answer)
            logging.info(f"Bot: {answer}")
            logging.info(f"Tokens: input_tokens - {tokens['input_tokens']}, output_tokens - {tokens['output_tokens']}, total_tokens - {tokens['total_tokens']}")


        logging.info("=== End session ===")

if __name__ == "__main__":
    bot = ClicBot()
    bot(SESSION_ID)
