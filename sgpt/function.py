import importlib.util
import sys
from abc import ABCMeta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .config import cfg
from .role import SystemRole


class Function:
    def __init__(self, path: str):
        module = self._read(path)
        self._function = module.Function.execute
        self._openai_schema = module.Function.openai_schema
        self._name = self._openai_schema["name"]
        self._tags = getattr(module.Function.Config, "tags", None)

    @property
    def name(self) -> str:
        return self._name  # type: ignore

    @property
    def tags(self) -> Optional[List[str]]:
        return self._tags

    @property
    def openai_schema(self) -> dict[str, Any]:
        return self._openai_schema  # type: ignore

    @property
    def execute(self) -> Callable[..., str]:
        return self._function  # type: ignore

    @classmethod
    def _read(cls, path: str) -> Any:
        module_name = path.replace("/", ".").rstrip(".py")
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)  # type: ignore
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore

        if not isinstance(module.Function, ABCMeta):
            raise TypeError(
                f"Function {module_name} must be a subclass of pydantic.BaseModel"
            )
        if not hasattr(module.Function, "execute"):
            raise TypeError(
                f"Function {module_name} must have a 'execute' static method"
            )

        return module


functions_folder = Path(cfg.get("OPENAI_FUNCTIONS_PATH"))
functions_folder.mkdir(parents=True, exist_ok=True)
functions = [Function(str(path)) for path in functions_folder.glob("*.py")]


def get_function(name: str) -> Callable[..., Any]:
    for function in functions:
        if function.name == name:
            return function.execute
    raise ValueError(f"Function {name} not found")


def get_openai_schemas(role: Optional[SystemRole]) -> List[Dict[str, Any]]:
    transformed_schemas = []
    
    for function in functions:
        if function.tags and len(function.tags) > 0:
            if not role:
                continue
            if role.tags and len(role.tags) > 0:
                if not any(tag in role.tags for tag in function.tags):
                    continue
        # else:
        #     print(f"Function {function.name} does not have roles.")
        schema = {
            "type": "function",
            "function": {
                "name": function.openai_schema["name"],
                "description": function.openai_schema.get("description", ""),
                "parameters": function.openai_schema.get("parameters", {}),
            },
        }
        transformed_schemas.append(schema)
    return transformed_schemas
