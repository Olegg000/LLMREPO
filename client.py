import os
import httpx # Используем httpx для асинхронных запросов
import json
import logging

# Настраиваем простую систему логирования для этого клиента
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class GeminiRepoClient:
    """
    Тестовый клиент для имитации Gemini API через репозиторий.
    Он принимает параметры, как это сделано в вашей системе,
    и имитирует запрос к Gemini.
    """
    def __init__(self, url: str, api_key: str, model: str = "gemini-pro"):
        self.base_url = url
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient() # Инициализируем асинхронный HTTP клиент
        log.info(f"GeminiRepoClient initialized for model '{self.model}' with base URL: {self.base_url}")

    async def generate(self, prompt: str, max_tokens: int) -> dict:
        """
        Имитирует асинхронную генерацию текста, как это делает Gemini API.
        """
        log.info(f"Repo client received prompt: '{prompt[:50]}...' (max_tokens: {max_tokens})")

        # Специальная логика для добавления API-ключа в URL для Gemini
        final_url = self.base_url
        if "key=GEMINI_API_KEY" in final_url:
            final_url = final_url.replace("GEMINI_API_KEY", self.api_key)
            log.info(f"API key successfully integrated into URL: {final_url}")
        else:
            log.warning("GEMINI_API_KEY placeholder not found in URL. Key might not be used correctly.")

        # Имитируем ответ Gemini
        response_text = f"Это сгенерированный ответ от Gemini (из репозитория) на ваш запрос: '{prompt}'. " \
                        f"Максимальное количество токенов было {max_tokens}. " \
                        f"Использована модель: {self.model}."

        # Имитируем использование токенов
        # Это очень грубая оценка, просто для примера
        input_tokens = len(prompt.split())
        output_tokens = len(response_text.split())
        total_tokens = input_tokens + output_tokens

        log.info(f"Simulated response generated. Tokens: {total_tokens}")

        # Возвращаем результат в формате, ожидаемом вашей системой
        return {
            "text": response_text,
            "usage_tokens": total_tokens
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose() # Закрываем асинхронный клиент
