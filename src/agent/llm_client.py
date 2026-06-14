import httpx
from typing import Optional

from src.config import (
    OLLAMA_HOST, OLLAMA_MODEL,
    VLLM_API_URL, VLLM_API_KEY,
    INFERENCE_BACKEND, setup_logging,
)

logger = setup_logging(__name__)


class LLMClient:
    def __init__(self, model: Optional[str] = None, host: Optional[str] = None,
                 backend: Optional[str] = None):
        self.model = model or OLLAMA_MODEL
        self.host = host or OLLAMA_HOST
        self.backend = backend or INFERENCE_BACKEND

        if self.backend == "vllm":
            self.vllm_url = VLLM_API_URL or self._build_vllm_url(self.host)
            self.vllm_key = VLLM_API_KEY

    @staticmethod
    def _build_vllm_url(host: str) -> str:
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(host)
        port = 8000
        netloc = f"{parsed.hostname}:{port}" if parsed.hostname else f"localhost:{port}"
        return urlunparse(("http", netloc, "/v1", "", "", ""))

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.3,
             max_tokens: int = 4096, response_format: Optional[dict] = None,
             timeout: float = 60.0) -> Optional[str]:
        if self.backend == "vllm":
            return self._call_vllm(messages, temperature, max_tokens, response_format, timeout)
        return self._call_ollama(messages, temperature, max_tokens, timeout)

    def _call_ollama(self, messages: list[dict[str, str]], temperature: float,
                     max_tokens: int, timeout: float) -> Optional[str]:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        try:
            response = httpx.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()["message"]["content"]
        except httpx.TimeoutException:
            logger.warning(f"Ollama 调用超时 ({timeout}s)")
            return None
        except httpx.HTTPStatusError as e:
            logger.warning(f"Ollama 返回错误状态: {e.response.status_code}")
            return None
        except httpx.ConnectError:
            logger.warning("Ollama 连接失败，请确认服务已启动")
            return None
        except Exception as e:
            logger.warning(f"Ollama 调用失败: {e}")
            return None

    def _call_vllm(self, messages: list[dict[str, str]], temperature: float,
                   max_tokens: int, response_format: Optional[dict],
                   timeout: float) -> Optional[str]:
        url = f"{self.vllm_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.vllm_key:
            headers["Authorization"] = f"Bearer {self.vllm_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format

        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except httpx.TimeoutException:
            logger.warning(f"vLLM 调用超时 ({timeout}s)")
            return None
        except httpx.HTTPStatusError as e:
            logger.warning(f"vLLM 返回错误状态: {e.response.status_code}")
            return None
        except httpx.ConnectError:
            logger.warning("vLLM 连接失败，请确认服务已启动")
            return None
        except Exception as e:
            logger.warning(f"vLLM 调用失败: {e}")
            return None
