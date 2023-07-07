import openai


class OpenAIModel:
    def __init__(self, api_key: str, api_base: str, api_type: str, api_version: str):
        openai.api_key = api_key
        openai.api_base = api_base
        openai.api_type = api_type
        openai.api_version = api_version

    def chat_completion(self, messages, model_engine="chinchin_test") -> str:
        try:
            response = openai.ChatCompletion.create(
                engine=model_engine, messages=messages
            )
            role = response["choices"][0]["message"]["role"]
            content = response["choices"][0]["message"]["content"].strip()
            return role, content
        except Exception as e:
            raise e

    def embedding(self, text, model_engine="chinchin_embedding") -> list:
        try:
            response = openai.Embedding.create(engine=model_engine, input=text)
            return response["data"][0]["embedding"]
        except Exception as e:
            raise e
