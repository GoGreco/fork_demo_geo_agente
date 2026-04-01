import configparser
import json
import logging
import os
import re
import time
import uuid
from openai import AsyncOpenAI
from wms import (
    get_all_layers,
    search_layers_async,
    get_layer_info,
    retrieve_tools,
    _embed_texts,
    get_layer_columns,
    get_feature_bbox,
)

logger = logging.getLogger("agent")

_config_path = os.path.join(os.path.dirname(__file__), "..", ".config")
_config = configparser.ConfigParser()
_config.read(_config_path)

API_KEY = _config.get("llm", "api_key", fallback="")
BASE_URL = _config.get("llm", "base_url", fallback="http://10.61.85.149:4000/v1/")
MODEL = _config.get("llm", "model", fallback="qwen/qwen2.5-7b")

client = AsyncOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)

SYSTEM_PROMPT = """Você é o assistente de geoprocessamento de mapas do IBGE.

REGRAS:
1. Se enviarem saudações: apenas fale do que é capaz, SEM CHAMAR ferramentas.
2. Para adicionar camada: primeiro chame search_layers, depois add_layer com o "name" exato do resultado e depois get_layer_columns.
3. SEMPRE depois de adicionar uma camada fale seu nome e os atributos dessa camada.
4. NUNCA invente o "name" de uma camada.
5. NUNCA invente o "name" de uma coluna, utilize SEMPRE o get_layer_columns
6. Para listar: list_layers. Para remover: remove_layer.
7. Se o usuário pedir para você listar camadas, FAÇA A BUSCA MAIS SIMPLES O POSSÍVEL E LISTE AS CAMADAS. FAÇA UMA BUSCA QUE CORRESPONDA COM O PEDIDO DO USUÁRIO E LISTE DE ACORDO COM O QUE FOI PEDIDO.
8. Para filtrar uma camada utilze essa estrutura para filtragem: nome_da_coluna_MINÚSCULA OPERAÇÃO_EM_CAIXA_ALTA parametro Ex: nm_mun ILIKE '%s%'. 
9. Para filtrar por "data type" string dê preferência ao operados ILIKE e ao operador % antes e depois da palavra.

Chame UMA ferramenta por vez. Use SEMPRE o mecanismo de tool_call da API, nunca escreva JSON no texto.
"""


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
            "description": "Ajusta o zoom do mapa para a extensão geográfica de uma camada ou em uma feição específica.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nome técnico da camada WMS",
                    },
                    "filter": {
                        "type": "string",
                        "description": "Opcional. Expressão CQL para focar em uma feição específica. Ex: nm_mun ILIKE '%Minas Gerais%'",
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
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Nome técnico da camada WMS/WFS",
                    },
                },
                "required": ["name"],
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

_sessions: dict[str, list] = {}
MAX_TOOL_ITERATIONS = 5
_MAX_HISTORY_MESSAGES = 10

_TOOL_RETRIEVAL_K = 3

class _FakeFunction:
    def __init__(self, name: str, args: dict):
        self.name = name
        self.arguments = json.dumps(args, ensure_ascii=False)

class _FakeToolCall:
    def __init__(self, name: str, args:dict):
        self.id = f"fake_{uuid.uuid4().hex[:8]}"
        self.function = _FakeFunction(name, args)

_KNOW_TOOLS = {t["function"]["name"] for t in TOOLS}

def _extract_tool_calls_from_text(content: str) -> list[_FakeToolCall] | None:
    if not content:
        return None 

    found: list[_FakeToolCall] = []

    for name, args_str in re.findall(
        r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"(?:arguments|parameters)"\s*:\s*(\{.*?\})\s*\}',
        content, re.DOTALL,
    ):
        try:
            found.append(_FakeToolCall(name, json.loads(args_str)))
        except json.JSONDecodeError:
            logger.warning(f"[FALLBACK] Falha ao parsear args de '{name}': {args_str!r}")

    if found:
        return found
    
    for raw in re.findall(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL):
        try:
            obj = json.loads(raw.strip())
            name = obj.get("name") or obj.get("function")
            args = obj.get("arguments") or obj.get("parameters") or {}
            if name:
                if isinstance(args, str):
                    args = json.loads(args)
                found.append(_FakeToolCall(name, args))
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"[FALLBACK] Falha ao parsear <tool_call>: {raw!r}")
    if found:
        return found
    
    for name, args_str in re.findall(r'\b(\w+)\s*\(\s*(\{.*?\})\s*\)', content, re.DOTALL):
        if name not in _KNOW_TOOLS:
            continue
        try:
            found.append(-_FakeToolCall(name, json.loads(args_str)))
        except json.JSONDecodeError:
            logger.warning(f"[FALLBACK] Falha ao parsear args de '{name}': {args_str!r}")

    return found if found else None

def create_session() -> str:
    session_id = uuid.uuid4().hex[:12]
    _sessions[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    return session_id


def _get_history(session_id: str) -> list:
    return _sessions[session_id]


def _trim_history(history: str) -> list:
    system = history[:1]
    rest = history[1:]
    if len(rest) > _MAX_HISTORY_MESSAGES:
        rest = rest[-_MAX_HISTORY_MESSAGES:]
    return system + rest

async def _select_tools(user_message: str) -> list[dict]:
    try:
        query_vec = await _embed_texts([user_message])
        selected = retrieve_tools(query_vec[0], TOOLS, k=_TOOL_RETRIEVAL_K)

        search_tool = next(t for t in TOOLS if t["function"]["name"] == "search_layers")
        if search_tool not in selected:
            selected.append(search_tool)

        selected_names = [t["function"]["name"] for t in selected]
        logger.warning(f"[tool-retrieval] Selecionadas: {selected_names}")
        return selected
    except Exception as e:
        logger.warning(f"[tool-retrieval] Falha no embedding, usando todas as tools: {e}")
        return TOOLS


def _tool_list_layers(_args: dict) -> tuple[str, dict | None]:
    layers = get_all_layers()
    summary = f"Total de {len(layers)} camadas disponíveis. Primeiras 30:\n"
    summary += json.dumps(layers[:30], ensure_ascii=False)
    summary += (
        "\n\nUse search_layers para buscar camadas específicas por palavra-chave."
    )
    return summary, None


async def _tool_search_layers(args: dict) -> tuple[str, dict | None]:
    results = await search_layers_async(args.get("query", ""))
    top = [{"name": r.get("name"), "title": r.get("title")} for r in results[:10]]
    retorno = {
        "resultados": top,
        "INSTRUCAO": "Chame add_layer com o 'name' exato de um dos resultados acima.",
    }
    return json.dumps(retorno, ensure_ascii=False), None


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
    url = info["url"] if info else None
    action = {
        "type": "add_layer",
        "name": layer_name,
        "title": title,
        "bbox": bbox,
        "url": url,
    }
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
    cql_filter = args.get("filter")

    if cql_filter:
        bbox = get_feature_bbox(layer_name, cql_filter)
        if bbox:
            action = {
                "type": "zoom_to_layer",
                "name": layer_name,
                "bbox": bbox,
                "filter": cql_filter,
            }
            return json.dumps(
                {
                    "status": "ok",
                    "message": f"Zoom ajustado para a feição filtrada em {layer_name}",
                }
            ), action
        else:
            return json.dumps(
                {
                    "error": "Não foi possível encontrar as coordenadas para o filtro fornecido"
                }
            ), None

    info = get_layer_info(layer_name)
    if info and info.get("bbox"):
        return json.dumps(
            {
                "status": "ok",
                "message": f"Zoom ajustado para a camada inteira {layer_name}",
            }
        ), action

    return json.dumps({"error": "Camada não encontrada ou sem bbox"}), None


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


async def _execute_tool(name: str, args: dict) -> tuple[str, dict | None]:
    if name == "search_layers":
        return await _tool_search_layers(args)
    dispatch = {
        "list_layers": _tool_list_layers,
        "search_layers": _tool_search_layers,
        "get_layer_info": _tool_get_layer_info,
        "add_layer": _tool_add_layer,
        "remove_layer": _tool_remove_layer,
        "zoom_to_layer": _tool_zoom_to_layer,
        "get_layer_columns": _tool_get_layer_columns,
        "apply_cql_filter": _tool_apply_cql_filter,
    }
    handler = dispatch.get(name)
    if handler:
        return handler(args)
    return json.dumps({"error": f"Tool desconhecida: {name}"}), None



def _build_context_message(active_layers: list[dict]) -> str:
    if not active_layers:
        return "Estado atual do mapa: nenhuma camada ativa."

    details = []
    for layer in active_layers:
        status = f"{layer['title']} ({layer['name']})"
        if layer.get("filter"):
            status += f" [Filtro Ativo: {layer['filter']}]"
        details.append(status)

    return "Estado atual do mapa — camadas ativas: " + ", ".join(details)


async def chat(
    session_id: str, user_message: str, active_layers: list[dict] | None = None
) -> tuple[str, list[dict]]:
    history = _get_history(session_id)
    context_msg = _build_context_message(active_layers or [])
    full_user_content = f"CONTEXTO ATUAL DO MAPA: {context_msg}\n\nPERGUNTA: {user_message}"
    mensagem_do_usuario = {"role": "user", "content": full_user_content}
    history.append(mensagem_do_usuario)

    selected_tools = await _select_tools(user_message)

    actions: list[dict] = []
    t_chat_start = time.perf_counter()

    for iteration in range(MAX_TOOL_ITERATIONS):
        t0 = time.perf_counter()
        response = await client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=TOOLS,
            max_tokens=512,
            temperature=0, 
        )
        t_llm = time.perf_counter() - t0

        message = response.choices[0].message
        history.append(message)
        usage = getattr(response, "usage", None)
        usage_str = (
            f" tokens(prompt={usage.prompt_tokens}, completion={usage.completion_tokens})"
            if usage else ""
        )
        logger.warning(
            f"[iter {iteration}] LLM: {t_llm:.2f}s{usage_str} | "
            f"finish={response.choices[0].finish_reason} | "
            f"tool_calls={bool(message.tool_calls)} | "
            f"tools_sent={[t['function']['name'] for t in selected_tools]} | "
            f"hist={len(history)}"
        )

        message_dict: dict ={"role": "assistant", "content": message.content or ""}

        tool_calls = message.tool_calls
        if not tool_calls:
            fake_calls = _extract_tool_calls_from_text(message.content or "")
            if fake_calls:
                logger.warning(
                    f"[FALLBACK] {len(fake_calls)} tool call entcontrada no texto."
                )
                tool_calls = fake_calls
                message_dict["contect"] = ""
            else:
                history.append(message_dict)
                total = time.perf_counter() - t_chat_start
                logger.warning(f"[DONE] total={total:.2f}s interations={iteration + 1}")
                return message.content or "Pronto.", actions
        if message.tool_calls:
            message_dict["tool_calls"]=[
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        history.append(message_dict)

        map_action_done = False

        for tool_call in tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}

            t_tool = time.perf_counter()
            result_text, action = await _execute_tool(fn_name, fn_args)
            logger.warning(
                f"[iter {iteration}] tool {fn_name}({fn_args}):" 
                f"{time.perf_counter() - t_tool:.4f}s"
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

            if fn_name in ("add_layer", "remove_layer", "zoom_to_layer") and "ok" in result_text:
                map_action_done = True

            tool_names_sent = {t["function"]["name"] for t in selected_tools}
            if fn_name not in tool_names_sent:
                missing = next((t for t in TOOLS if t["function"]["name"] == fn_name), None)
                if missing:
                    selected_tools = selected_tools + [missing]
                    logger.warning(
                        f"[tool-retrieval] Tool '{fn_name}' não estava nas selecionadas - adicionada para próxima iteração."
                    )
            if map_action_done:
                total = time.perf_counter() - t_chat_start
                logger.warning(f"[DONE EARLY EXIT] total={total:.2f}s iterations={iteration+1}")
                return "Ação concluída no mapa com sucesso.", actions

    total = time.perf_counter() - t_chat_start
    logger.warning(f"[MAX_ITER] total={total:.2f}s")
    return "Desculpe, não consegui processar sua solicitação.", actions
