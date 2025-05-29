import os
import httpx # Используем httpx для асинхронных запросов
import json
import logging
import asyncio # Добавляем импорт asyncio
import sys   # Добавляем импорт sys для работы со stdin/stdout

# Настраиваем систему логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class GeminiRepoClient:
    """
    Клиент для взаимодействия с Gemini API через репозиторий.
    Он принимает параметры, как это сделано в вашей системе,
    и выполняет реальный запрос к Gemini.
    """
    def __init__(self, url: str, api_key: str, model: str = "gemini-pro"):
        # Если API ключ приходит как переменная окружения, используем ее
        if "API_KEY_ENV" in api_key:
            self.api_key = os.environ.get(api_key.split('=')[1], api_key)
            log.info(f"Using API key from environment variable: {api_key.split('=')[1]}")
        else:
            self.api_key = api_key

        self.base_url = url.replace("GEMINI_API_KEY", self.api_key) # Вставляем ключ прямо в URL, если он там есть
        self.model = model
        self._client = None # Инициализируем как None, создадим при первом использовании

        log.info(f"GeminiRepoClient initialized for model '{self.model}' with base URL: {self.base_url}")


    async def _get_client(self):
        """Ленивая инициализация асинхронного HTTP клиента."""
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    async def generate(self, prompt: str, max_tokens: int = 8192) -> dict:
        """
        Выполняет реальный асинхронный запрос к Gemini API.
        """
        log.info(f"Repo client received prompt: '{prompt[:50]}...' (max_tokens: {max_tokens})")

        client = await self._get_client() # Получаем или создаем асинхронный клиент

        headers = {
            "Content-Type": "application/json"
        }
        # Gemini API часто принимает ключ как часть URL или как параметр запроса.
        # В вашем случае, судя по логам, ключ уже вставляется в URL.
        # Если это не так, возможно, вам нужно будет добавить его в параметры JSON или в заголовки.
        # Например, если ключ в заголовке: headers["x-goog-api-key"] = self.api_key

        request_body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens
                # Можно добавить другие параметры, если они есть в params_json
                # "temperature": 0.9,
                # "topP": 1.0,
                # "topK": 32
            }
        }

        try:
            log.info(f"Sending request to Gemini URL: {self.base_url}")
            log.debug(f"Request body: {json.dumps(request_body)}")

            response = await client.post(
                self.base_url,
                headers=headers,
                json=request_body,
                timeout=180 # Увеличим таймаут на всякий случай
            )
            response.raise_for_status() # Вызывает исключение для ошибок HTTP (4xx, 5xx)

            api_result = response.json()
            log.debug(f"Raw API response: {json.dumps(api_result, indent=2)}")

            generated_text = ""
            if api_result and 'candidates' in api_result and api_result['candidates']:
                # Ищем текст в ответе
                for part in api_result['candidates'][0]['content']['parts']:
                    if 'text' in part:
                        generated_text += part['text']
                        break # Предполагаем, что нужен первый текстовый блок

            # Извлекаем использование токенов, если оно есть в ответе Gemini
            # Это может отличаться в зависимости от версии API и модели
            usage_tokens = 0
            if 'usageMetadata' in api_result and 'totalTokenCount' in api_result['usageMetadata']:
                usage_tokens = api_result['usageMetadata']['totalTokenCount']
            else:
                # Если Gemini не возвращает usageMetadata, сделаем грубую оценку
                input_tokens = len(prompt.split()) # Очень грубая оценка
                output_tokens = len(generated_text.split()) # Очень грубая оценка
                usage_tokens = input_tokens + output_tokens
                log.warning("usageMetadata not found in Gemini response. Estimating token count.")

            log.info(f"Gemini API call successful. Generated text length: {len(generated_text)}. Tokens used: {usage_tokens}")

            return {
                "text": generated_text,
                "usage_tokens": usage_tokens
            }
        except httpx.RequestError as exc:
            log.error(f"An error occurred while requesting {exc.request.url!r}: {exc}")
            raise RuntimeError(f"Network or request error: {exc}")
        except httpx.HTTPStatusError as exc:
            log.error(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}: {exc.response.text}")
            raise RuntimeError(f"API error: {exc.response.status_code} - {exc.response.text}")
        except json.JSONDecodeError as exc:
            log.error(f"Failed to decode JSON from API response: {exc}. Response text: {response.text}")
            raise RuntimeError(f"Invalid JSON response from API: {exc}")
        except Exception as e:
            log.error(f"An unexpected error occurred during Gemini API call: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error: {e}")


    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose() # Закрываем асинхронный клиент

# --- Блок для запуска из subprocess ---
if __name__ == "__main__":
    # Считываем входные данные из stdin
    input_data_str = sys.stdin.read()
    try:
        parsed_input = json.loads(input_data_str)

        prompt = parsed_input.get("prompt")
        max_tokens = parsed_input.get("max_tokens", 8192)

        # Извлекаем параметры для инициализации клиента из parsed_input
        client_params = {
            "url": parsed_input.get("url"),
            "api_key": parsed_input.get("api_key"),
            "model": parsed_input.get("model")
            # Добавьте любые другие параметры, которые ожидает ваш клиент
        }

        # Создаем экземпляр клиента
        client_instance = GeminiRepoClient(**client_params)

        # Запускаем асинхронную функцию generate
        # asyncio.run() нужен для запуска async функций из синхронного контекста
        result = asyncio.run(client_instance.generate(prompt, max_tokens))

        # Выводим результат в stdout в формате JSON
        print(json.dumps(result))

    except json.JSONDecodeError:
        log.error("Error: Invalid JSON input from stdin. This usually means the input from the main script was not valid JSON.")
        sys.exit(1)
    except Exception as e:
        log.error(f"An unexpected error occurred in client.py execution: {e}", exc_info=True)
        sys.exit(1)
