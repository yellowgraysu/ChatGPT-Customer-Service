from dotenv import load_dotenv
import os
import openai

load_dotenv()

openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.ChatCompletion.create(
    engine="chinchin_test",  # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
    messages=[
        {
            "role": "system",
            "content": "Assistant is a large language model trained by OpenAI.",
        },
        {"role": "user", "content": "Who were the founders of Microsoft?"},
    ],
)

print(response)

print(response["choices"][0]["message"]["content"])
