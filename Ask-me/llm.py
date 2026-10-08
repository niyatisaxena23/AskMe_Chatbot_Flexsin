"""Groq chat-completions client for Qwen3."""

import os

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3.8-27b"


class LLMError(RuntimeError):
    pass


class GroqClient:
    def __init__(self, api_key=None, model=None, timeout=45):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        self.timeout = timeout

    @property
    def configured(self):
        return bool(self.api_key)

    def _payload(self, messages, temperature=0.2, max_tokens=700, stream=False):
        return {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.9,
            # Qwen3 is a reasoning model; keep the thinking out of the reply.
            "reasoning_format": "hidden",
            "stream": stream,
        }

    def _headers(self):
        return {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }

    def complete(self, messages, temperature=0.2, max_tokens=700):
        """Return the complete Groq response as one string."""
        if not self.configured:
            raise LLMError("GROQ_API_KEY is not set")

        payload = self._payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )

        try:
            response = requests.post(
                GROQ_URL,
                json=payload,
                headers=self._headers(),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise LLMError("could not reach Groq: {}".format(exc)) from exc

        if response.status_code != 200:
            raise LLMError(
                "Groq returned {}: {}".format(
                    response.status_code,
                    response.text[:400],
                )
            )

        data = response.json()

        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("unexpected Groq response shape") from exc

    def stream(self, messages, temperature=0.2, max_tokens=700):
        """Yield text deltas from Groq's streaming response.

        This method yields raw model deltas to the caller. The Flask
        application is responsible for applying reasoning stripping and
        answer anonymization before anything is sent to the browser.
        """
        if not self.configured:
            raise LLMError("GROQ_API_KEY is not set")

        payload = self._payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        try:
            response = requests.post(
                GROQ_URL,
                json=payload,
                headers=self._headers(),
                timeout=(10, self.timeout),
                stream=True,
            )
        except requests.RequestException as exc:
            raise LLMError("could not reach Groq: {}".format(exc)) from exc

        if response.status_code != 200:
            try:
                error_text = response.text[:400]
            except Exception:
                error_text = "unknown error"

            response.close()

            raise LLMError(
                "Groq returned {}: {}".format(
                    response.status_code,
                    error_text,
                )
            )

        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue

                if not line.startswith("data:"):
                    continue

                data = line[5:].strip()

                if data == "[DONE]":
                    break

                try:
                    chunk = requests.models.complexjson.loads(data)
                except ValueError as exc:
                    raise LLMError(
                        "invalid streaming response from Groq"
                    ) from exc

                try:
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    content = delta.get("content")

                    if content:
                        yield content

                except (AttributeError, IndexError, TypeError) as exc:
                    raise LLMError(
                        "unexpected Groq streaming response shape"
                    ) from exc

        except requests.RequestException as exc:
            raise LLMError(
                "Groq streaming request failed: {}".format(exc)
            ) from exc

        finally:
            response.close()