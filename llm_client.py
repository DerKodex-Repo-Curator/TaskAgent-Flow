import requests

DEFAULT_HOST = "http://localhost:11434"


class OllamaClient:
    def __init__(self, host: str = DEFAULT_HOST, model: str = None):
        self.host = host.rstrip('/')
        self.model = model or self._resolve_model()

    def _resolve_model(self) -> str:
        """Checks local Ollama server to find an available model matching candidates."""
        try:
            res = requests.get(f"{self.host}/api/tags", timeout=3)
            if res.status_code == 200:
                models = res.json().get('models', [])
                # Normalize tags (e.g. 'llama3:latest' -> 'llama3')
                installed_names = [m['name'] for m in models]
                installed_bases = [m['name'].split(':')[0] for m in models]

                # Check preferences
                if 'llama3' in installed_bases:
                    idx = installed_bases.index('llama3')
                    return installed_names[idx]
                if 'mistral' in installed_bases:
                    idx = installed_bases.index('mistral')
                    return installed_names[idx]
                if installed_names:
                    return installed_names[0]
        except Exception:
            pass
        return "llama3"

    def generate(self, prompt: str) -> str:
        """Sends a generation request to the local Ollama instance."""
        url = f"{self.host}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "").strip()
        raise RuntimeError(f"Ollama returned HTTP error code {response.status_code}")