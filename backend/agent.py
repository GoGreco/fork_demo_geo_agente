import configparser
import json
import logging
import os
import time
import uuid
from openai import AsyncOpenAI
from wms import get_all_layers, search_layers, get_layer_info, get_layer_columns

logger = logging.getLogger("agent")

_config_path = os.path.join(os.path.dirname(__file__), "..", ".config")
_config = configparser.ConfigParser()
_config.read(_config_path)

API_KEY = _config.get("llm", "api_key", fallback="")
BASE_URL = _config.get("llm", "base_url", fallback="http://10.61.85.149:4000/v1/")
MODEL = _config.get("llm", "model", fallback="qwen/qwen3.5-9b")

client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

SYSTEM_PROMPT = """Responda sempre em português. Você é um assistente para processamento de dados geográficos. 
Você lerá o que o usuário está digitando e vai retornar para o usuário qual função você acredita ser mais adequada para essa requisição.
- caso for filtrar UTILIZE APENAS as colunas conhecidas da camada """

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
    {
        "type": "function",
        "function": {
            "name": "get_layer_columns",
            "description": "obtém os nomes exatos das colunas/atributos da tabela de uma camada. IMPORTANTE usar antes de criar um filtro CQL para não errar o nome da coluna.",
            "parameters": {
                "type": "object",
                "properties":{
                    "name":{
                        "type": "string",
                        "description": "Nome técnico da camada WMS/WFS",
                    },
                },
                "required":["name"],
            },       
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_cql_filter",
            "description": "Aplica um filtro CQL a uma camada WMS no mapa.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nome técnico da camada WMS",
                    },
                    "filter": {
                        "type": "string",
                        "description": "Expressão CQL. Ex: NM_UF = 'São Paulo'",
                    },
                },
                "required": ["name", "filter"],
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
    top_results = results[:10]
    resultados_limpos =[
        {"name": r.get("name"), "title": r.get("title")}
        for r in top_results
    ]
    return json.dumps(resultados_limpos, ensure_ascii=False), None


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

#tentativa fitro CQL
def _tool_get_layer_columns(args: dict) -> tuple[str, dict | None]:
    layer_name = args.get("name", "")
    columns = get_layer_columns(layer_name)

    if columns:
        return json.dumps({"colunas_disponíveis": columns}, ensure_ascii=False), None
    return json.dumps({"error": "Não foi possível carregar as colunas"}), None

def _tool_apply_cql_filter(args: dict) -> tuple[str, dict | None]:
    layer_name = args.get("name", "")
    cql_filter = args.get("filter", "")

    action = {"type": "apply_cql_filter", "name": layer_name, "filter": cql_filter}

    return json.dumps({"status": "ok", "message": "Filtro CQL aplicado."}), action



_TOOL_DISPATCH = {
    "list_layers": _tool_list_layers,
    "search_layers": _tool_search_layers,
    "get_layer_info": _tool_get_layer_info,
    "add_layer": _tool_add_layer,
    "remove_layer": _tool_remove_layer,
    "zoom_to_layer": _tool_zoom_to_layer,
    "get_layer_columns": _tool_get_layer_columns,
    "apply_cql_filter": _tool_apply_cql_filter,
}


def _execute_tool(name: str, args: dict) -> tuple[str, dict | None]:
    handler = _TOOL_DISPATCH.get(name)
    if handler:
        return handler(args)
    return json.dumps({"error": f"Tool desconhecida: {name}"}), None

def _build_context_message(active_layers: list[dict]) -> str:
    if not active_layers:
        return "Estado atual do mapa: nenhuma camada ativa."
    
    details = []
    for layer in active_layers:
        status = f"{layer['title']} ({layer['name']})"
        if layer.get('filter'):
            status += f" [Filtro Ativo: {layer['filter']}]"
        details.append(status)
    
    return "Estado atual do mapa — camadas ativas: " + ", ".join(details)

async def chat(
    session_id: str, user_message: str, active_layers: list[dict] | None = None
) -> tuple[str, list[dict]]:
    """Process a chat message. Returns (reply_text, actions_list)."""
    history = _get_history(session_id)
    
    contexto_mapa = _build_context_message(active_layers or [])

    full_user_content = f"CONTEXTO ATUAL DO MAPA: {contexto_mapa}\n\nPERGUNTA: {user_message}"

    mensagem_do_usuario = {"role": "user", "content": full_user_content}

    history.append(mensagem_do_usuario)
    actions = []
    t_chat_start = time.perf_counter()

    for iteration in range(MAX_TOOL_ITERATIONS):
        t0 = time.perf_counter()
        response = await client.chat.completions.create(
            model=MODEL,
            messages=history, 
            tools=TOOLS,
        )
        t_llm = time.perf_counter() - t0

        message = response.choices[0].message
        history.append(message)
        usage = getattr(response, "usage", None)
        usage_str = (
            f" tokens(prompt={usage.prompt_tokens}, completion={usage.completion_tokens})"
            if usage
            else ""
        )
        logger.warning(
            f"[iter {iteration}] LLM call: {t_llm:.2f}s{usage_str} | msg_count={len(history)}"
        )

        if not message.tool_calls:
            total = time.perf_counter() - t_chat_start
            logger.warning(f"[DONE] total={total:.2f}s iterations={iteration + 1}")
            return message.content or "", actions

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            t_tool = time.perf_counter()
            result_text, action = _execute_tool(fn_name, fn_args)
            t_tool = time.perf_counter() - t_tool
            logger.warning(
                f"[iter {iteration}] tool {fn_name}({fn_args}): {t_tool:.4f}s"
            )

            if action:
                actions.append(action)

            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text,
                }
            )

    total = time.perf_counter() - t_chat_start
    logger.warning(f"[MAX_ITER] total={total:.2f}s")
    return "Desculpe, não consegui processar sua solicitação.", actions
