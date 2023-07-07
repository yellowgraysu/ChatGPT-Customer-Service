from dotenv import load_dotenv
from flask import Flask, request
from src.models import OpenAIModel
from src.hyper_db import (
    DATA_DIR,
    DB_FILE_NAME,
    DOCUMENTS_NAME,
    CustomizeHyperDB,
    initHyperDB,
    getHyperDocuments,
)
import os

load_dotenv()
app = Flask(__name__)
db = CustomizeHyperDB()
model = OpenAIModel(
    os.getenv("OPENAI_API_KEY"),
    os.getenv("OPENAI_API_BASE"),
    os.getenv("OPENAI_API_TYPE"),
    os.getenv("OPENAI_API_VERSION"),
)

TOKENS_LIMIT = 1800
DEFAULT_SYSTEM = "你是AI虛擬接待員個人資料 : 姓名：小青, 30歲，身高168cm，體重45kg，台灣人，大學畢業，是一名青青婚宴會館的接待員，工作是站在會館入口處解決賓客的問題。長相清純可愛，膚色白皙，身材高挑修長。個性大方、具備禮儀、可愛、帶一點俏皮，善於關心、照顧、體貼、逗客人開心。"
DEFAULT_ANSWER = "抱歉小青不太清楚，還有其他可以幫上忙的地方嗎？"


@app.route("/qa", methods=["POST"])
def handle_message():
    try:
        data = request.get_json()

        # 問題
        question = data.get("question")

        # 角色設定
        system = data.get("system") or DEFAULT_SYSTEM

        # 範例問答/聊天記錄
        history = data.get("history") or []

        # 不清楚時的預設回答
        default_answer = data.get("default") or DEFAULT_ANSWER

        # custom_messages = data.get("messages")
        documents = getHyperDocuments(db, model, question)
        messages = (
            [{"role": "system", "content": system}]
            + history
            + [
                {
                    "role": "user",
                    "content": f"""
            只回答存在文本或是和自己相關的問題，如答案不在以下文本中，請回答「{default_answer}」

            文本：{str(documents)[:TOKENS_LIMIT]}

            Q: {question}
            A:
            """,
                },
            ]
        )
        print("============\n")
        print(messages)
        print("============\n")
        _, content = model.chat_completion(messages)
    except Exception as e:
        print(str(e))
        content = default_answer
    response = {"answer": content}
    return response


# 更新文本
@app.route("/upload", methods=["POST"])
def updateInfo():
    print("updating...")
    if "file" not in request.files:
        return "No file found", 400

    file = request.files["file"]

    if file.filename == "":
        return "No selected file", 400

    # 將檔案儲存到指定路徑
    file.save(DATA_DIR + DOCUMENTS_NAME)
    os.remove(DATA_DIR + DB_FILE_NAME)
    global db
    db = CustomizeHyperDB()
    initHyperDB(db, model)
    return "ok"


if __name__ == "__main__":
    initHyperDB(db, model)
    app.run(host="0.0.0.0", port=8000)

# 流程介紹：
# 1. 把資料吃入 hyperDB / 已經存在就讀取 hyperDB (hyperdb.pickle.gz)
#   1.1 使用 OpenAIModel embedding 預處理資料，轉成向量資料
#   1.2 把文件/向量存入 hyperDB
# 2. 讀取 query, 使用 OpenAIModel embedding 預處理 query 成向量
# 3. 使用 query 向量比對 hyperDB 取得相似的部分文件
# 4. 使用部分文件 + query 詢問 chatGPT 取得回答
