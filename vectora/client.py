from __future__ import annotations

import time
from typing import Any

import requests

from vectora.exceptions import (
    VectoraAuthError,
    VectoraConfigError,
    VectoraConnectionError,
    VectoraNotFoundError,
    VectoraRateLimitError,
    VectoraServerError,
)


class VectoraClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://vectora.ai",
        timeout: float = 5.0,
        max_retries: int = 2,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.startswith("vct_"):
            raise VectoraConfigError("Vectora API keys must start with 'vct_'.")

        if len(api_key.strip()) < 20:
            raise VectoraConfigError("Vectora API keys must be at least 20 characters long.")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()

    def _post(self, path: str, json_payload: dict[str, Any]) -> dict[str, Any]:
        if not path.startswith("/"):
            raise VectoraConfigError("Vectora client paths must start with '/'.")

        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "vectora-python-sdk/0.1.0",
        }

        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.post(
                    url,
                    json=json_payload,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.Timeout as exc:
                last_error = VectoraConnectionError(
                    "Timed out while reaching the Vectora API. Check your network or increase the timeout."
                )
            except requests.RequestException as exc:
                last_error = VectoraConnectionError(
                    f"Couldn't reach the Vectora API: {exc}"
                )
            else:
                if response.status_code == 401:
                    raise VectoraAuthError(
                        "Vectora rejected the API key. Verify that your key is current and starts with 'vct_'."
                    )

                if response.status_code == 404:
                    raise VectoraNotFoundError(
                        "The requested Vectora endpoint or resource was not found."
                    )

                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After")
                    detail = (
                        f" Retry after {retry_after} seconds."
                        if retry_after and retry_after.isdigit()
                        else ""
                    )
                    raise VectoraRateLimitError(
                        f"Vectora rate-limited this request.{detail}"
                    )

                if 500 <= response.status_code:
                    raise VectoraServerError(
                        f"Vectora returned a server error ({response.status_code}). Try again shortly."
                    )

                if not response.ok:
                    raise VectoraServerError(
                        f"Vectora rejected the request with status {response.status_code}: {response.text}"
                    )

                if not response.content:
                    return {}

                data = response.json()
                return data if isinstance(data, dict) else {"data": data}

            if attempt < self.max_retries:
                time.sleep(0.25 * (attempt + 1))

        raise last_error or VectoraConnectionError("Couldn't reach the Vectora API.")
