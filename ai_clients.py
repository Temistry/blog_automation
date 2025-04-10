import os
import requests
import openai
from config import DEEPINFRA_MODELS, OPENAI_MODELS

class BaseLLMClient:
    def chat_completion(self, model, messages, temperature=0.7, max_tokens=2048):
        raise NotImplementedError

class OpenAIClient(BaseLLMClient):
    def __init__(self):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.models = OPENAI_MODELS

    def chat_completion(self, model, messages, temperature=0.7, max_tokens=2048):
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        ).choices[0].message.content

class DeepInfraClient(BaseLLMClient):
    def __init__(self):
        self.api_key = os.getenv("DEEPINFRA_API_KEY")
        self.models = DEEPINFRA_MODELS
        # OpenAI 클라이언트 초기화 - DeepInfra 엔드포인트 사용
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepinfra.com/v1/openai"
        )
        
    def chat_completion(self, model, messages, temperature=0.7, max_tokens=2048):
        """OpenAI 클라이언트 라이브러리를 사용하여 DeepInfra API 호출"""
        response = self.client.chat.completions.create(
            model=model,  # DeepInfra 모델 이름(예: google/gemma-3-4b-it)
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content 