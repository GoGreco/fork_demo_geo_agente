import configparser
import json
import os
import uuid
from openai import AsyncOpenAI
from wms import get_all_layers, search_layers, get_layer_info

_config_path = os.path.join(os.path.dirname(__file__), "..", ".config")
_config = configparser.ConfigParser()
_config.read(_config_path)

API_KEY = _config.get("llm", "api_key", fallback="")
BASE_URL = _config.get("llm", "base_url", fallback="https://openrouter.ai/api/v1")
MODEL = _config.get("llm", "model", fallback="qwen/qwen3-8b")

client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

SYSTEM_PROMPT = """Você é um assistente de geoprocessamento que ajuda usuários a explorar dados geográficos do IBGE.
Você tem acesso a ferramentas para listar, buscar e manipular camadas WMS no mapa.

IMPORTANTE - Seu fluxo de trabalho:
- Quando o usuário pedir para adicionar/mostrar/exibir uma camada, SEMPRE use search_layers primeiro para encontrar o nome técnico, depois use add_layer com o resultado mais relevante.
- Quando houver múltiplos resultados de busca, escolha o mais relevante e adicione-o. Se realmente houver ambiguidade, apresente no máximo 5 opções.
- Quando o usuário pedir para listar camadas, use list_layers.
- Quando o usuário pedir para remover uma camada, use remove_layer.
- Use as ferramentas proativamente. NÃO peça informações que você pode descobrir usando as ferramentas.
- Responda sempre em português. Seja conciso e útil.
- Quando o usuário mencionar "estados", busque por "estado" ou "UF". Quando mencionar "municípios", busque por "municipio". Use termos simples nas buscas."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_layers",
            "description": "Lista uma amostra das camadas WMS disponíveis (são mais de 9000). Para encontrar camadas específicas, prefira usar search_layers.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_layers",
            "description": "Busca camadas WMS por palavra-chave no nome, título ou descrição.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Termo de busca"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_layer_info",
            "description": "Obtém metadados detalhados de uma camada específica (nome, título, descrição, bbox).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nome técnico da camada (ex: CCAR:BC250_Capital_P)",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_layer",
            "description": "Adiciona uma camada WMS ao mapa do usuário. Use após identificar qual camada o usuário deseja visualizar.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nome técnico da camada WMS",
                    },
                    "title": {
                        "type": "string",
                        "description": "Título legível da camada",
                    },
                },
                "required": ["name", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_layer",
            "description": "Remove uma camada WMS do mapa do usuário.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nome técnico da camada WMS a remover",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "zoom_to_layer",
            "description": "Ajusta o zoom do mapa para a extensão geográfica de uma camada.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nome técnico da camada WMS",
                    },
                },
                "required": ["name"],
            },
        },
    },
]

# In-memory session store
_sessions: dict[str, list] = {}

MAX_TOOL_ITERATIONS = 10


def create_session() -> str:
    session_id = uuid.uuid4().hex[:12]
    _sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return session_id


def _get_history(session_id: str) -> list:
    return _sessions[session_id]


def _tool_list_layers(_args: dict) -> tuple[str, dict | None]:
    layers = get_all_layers()
    summary = f"Total de {len(layers)} camadas disponíveis. Primeiras 30:\n"
    summary += json.dumps(layers[:30], ensure_ascii=False)
    summary += (
        "\n\nUse search_layers para buscar camadas específicas por palavra-chave."
    )
    return summary, None


def _tool_search_layers(args: dict) -> tuple[str, dict | None]:
    results = search_layers(args.get("query", ""))
    return json.dumps(results, ensure_ascii=False), None


def _tool_get_layer_info(args: dict) -> tuple[str, dict | None]:
    info = get_layer_info(args.get("name", ""))
    if info:
        return json.dumps(info, ensure_ascii=False), None
    return json.dumps({"error": "Camada não encontrada"}), None


def _tool_add_layer(args: dict) -> tuple[str, dict | None]:
    layer_name = args.get("name", "")
    title = args.get("title", layer_name)
    info = get_layer_info(layer_name)
    bbox = info["bbox"] if info else None
    action = {"type": "add_layer", "name": layer_name, "title": title, "bbox": bbox}
    return json.dumps(
        {"status": "ok", "message": f"Camada {title} adicionada ao mapa"}
    ), action


def _tool_remove_layer(args: dict) -> tuple[str, dict | None]:
    layer_name = args.get("name", "")
    action = {"type": "remove_layer", "name": layer_name}
    return json.dumps(
        {"status": "ok", "message": f"Camada {layer_name} removida do mapa"}
    ), action


def _tool_zoom_to_layer(args: dict) -> tuple[str, dict | None]:
    layer_name = args.get("name", "")
    info = get_layer_info(layer_name)
    if info and info.get("bbox"):
        action = {"type": "zoom_to_layer", "name": layer_name, "bbox": info["bbox"]}
        return json.dumps(
            {"status": "ok", "message": f"Zoom ajustado para {layer_name}"}
        ), action
    return json.dumps({"error": "Camada não encontrada ou sem bbox"}), None


_TOOL_DISPATCH = {
    "list_layers": _tool_list_layers,
    "search_layers": _tool_search_layers,
    "get_layer_info": _tool_get_layer_info,
    "add_layer": _tool_add_layer,
    "remove_layer": _tool_remove_layer,
    "zoom_to_layer": _tool_zoom_to_layer,
}


def _execute_tool(name: str, args: dict) -> tuple[str, dict | None]:
    handler = _TOOL_DISPATCH.get(name)
    if handler:
        return handler(args)
    return json.dumps({"error": f"Tool desconhecida: {name}"}), None


def _build_context_message(active_layers: list[dict]) -> str:
    if not active_layers:
        return "Estado atual do mapa: nenhuma camada ativa."
    names = ", ".join(f"{layer['title']} ({layer['name']})" for layer in active_layers)
    return f"Estado atual do mapa — camadas ativas: {names}"


async def chat(
    session_id: str, user_message: str, active_layers: list[dict] | None = None
) -> tuple[str, list[dict]]:
    """Process a chat message. Returns (reply_text, actions_list)."""
    history = _get_history(session_id)
    history.append(
        {"role": "system", "content": _build_context_message(active_layers or [])}
    )
    history.append({"role": "user", "content": user_message})

    actions = []

    for _ in range(MAX_TOOL_ITERATIONS):
        response = await client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=TOOLS,
        )

        message = response.choices[0].message
        history.append(message)

        if not message.tool_calls:
            return message.content or "", actions

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            result_text, action = _execute_tool(fn_name, fn_args)
            if action:
                actions.append(action)

            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                }
            )

    return "Desculpe, não consegui processar sua solicitação.", actions
