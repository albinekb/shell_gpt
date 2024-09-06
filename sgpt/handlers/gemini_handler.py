import google.generativeai as genai
from typing import Any, Dict, Generator, List, Optional

from .handler import Handler
from ..config import cfg
from ..role import SystemRole

def model_name_to_model_id(model_name: str) -> str:
    model_map = {
        "gemini-flash": "gemini-1.5-flash",
        "gemini-pro": "gemini-1.5-pro",
    }
    
    if model_name in model_map.values():
        result = model_name
    else:
        result = model_map.get(model_name)
    if result is None:
        raise ValueError(f"Model {model_name} not found")
    return result

class GeminiBaseHandler(Handler):
    def __init__(self, role: SystemRole, md: bool):
        super().__init__(role, md)
        self.model_id = model_name_to_model_id(cfg.get("GEMINI_MODEL"))

class GeminiHandler(GeminiBaseHandler):

    def __init__(self, role: SystemRole, md: bool):
        super().__init__(role, md)
        
        genai.configure(api_key=cfg.get("GEMINI_API_KEY"))
        
        self.model = genai.GenerativeModel(self.model_id)

    def make_messages(self, prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.role.role},
            {"role": "user", "content": prompt}
        ]

    def get_completion(
        self,
        model: str,
        temperature: float,
        top_p: float,
        messages: List[Dict[str, Any]],
        functions: Optional[List[Dict[str, str]]],
        caching: bool,
    ) -> Generator[str, None, None]:
        prompt = "\n".join([m["content"] for m in messages])
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
            ),
            stream=True,
        )
        for chunk in response:
            yield chunk.text

class GeminiChatHandler(GeminiBaseHandler):
    def __init__(self, chat: bool, role: SystemRole, md: bool):
        super().__init__(role, md)
        genai.configure(api_key=cfg.get("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(self.model_id)
        self.chat = self.model.start_chat(history=[])

    def get_completion(
        self,
        model: str,
        temperature: float,
        top_p: float,
        prompt: str,
        functions: Optional[List[Dict[str, str]]],
        caching: bool,
    ) -> Generator[str, None, None]:
        response = self.chat.send_message(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                top_p=top_p,
            ),
            stream=True,
        )
        for chunk in response:
            yield chunk.text
