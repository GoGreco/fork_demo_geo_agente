import os
import asyncio
import sqlite3
import json
import logging
import urllib.parse

import numpy as np
import httpx
import xml.etree.ElementTree as ET
from rapidfuzz import fuzz, process

logger = logging.getLogger("wms")

WMS_URL = "http://localhost:8180/geoserver/wms"
WFS_URL = "http://localhost:8180/geoserver/wfs"
CAPABILITIES_URL = f"{WMS_URL}?service=WMS&request=GetCapabilities"
NS = {"wms": "http://www.opengis.net/wms"}


EMBED_URL = os.environ.get("EMBED_URL", "http://10.61.85.149:8001/v1")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
EMBED_API_KEY = os.environ.get("EMBED_API_KEY", "none")

_db: sqlite3.Connection | None = None
_layers_cache: list[dict] = []

_embed_matrix: np.ndarray | None = None
_embed_names: list[str] = []


_TOOL_INTENT_PHRASES: dict[str, list[str]] = {
     "list_layers": [
        "listar camadas disponíveis",
        "quais camadas existem",
        "mostrar todas as camadas",
        "ver opções de camadas",
        "que mapas tem disponível",
        "liste camadas relacionadas a", 
        "me fale de camadas que", 
    ],
    "search_layers": [
        "buscar camada por nome",
        "pesquisar camada",
        "encontrar camada de municípios",
        "procurar camada de hidrografia",
        "achar camada sobre vegetação",
        "qual camada representa biomas",
        "localizar dado geográfico",
        "camadas relacionadas a",
    ],
    "get_layer_info": [
        "informações sobre a camada",
        "metadados da camada",
        "detalhes técnicos da camada",
        "descrição da camada",
        "bbox da camada",
        "o que é essa camada",
    ],
    "add_layer": [
        "adicionar camada ao mapa",
        "mostrar camada no mapa",
        "exibir mapa de municípios",
        "carregar camada",
        "colocar camada no mapa",
        "quero ver o mapa de",
        "exibir biomas",
        "visualizar hidrografia",
    ],
    "remove_layer": [
        "remover camada do mapa",
        "tirar camada",
        "deletar camada",
        "ocultar camada",
        "limpar mapa",
        "retirar do mapa",
    ],
    "zoom_to_layer": [
        "zoom na camada",
        "focar na camada",
        "centralizar mapa na camada",
        "ir para a camada",
        "navegar até a camada",
        "ajustar zoom",
        "vá para camada", 
    ],
}

_tool_embed_matrix: np.ndarray | None = None  
_tool_embed_index: list[str] = [] 

def _parse_bbox(layer_el):
    bb = layer_el.find("wms:EX_GeographicBoundingBox", NS)
    if bb is None:
        return None
    try:
        return [
            float(bb.findtext("wms:westBoundLongitude", namespaces=NS)),
            float(bb.findtext("wms:southBoundLatitude", namespaces=NS)),
            float(bb.findtext("wms:eastBoundLongitude", namespaces=NS)),
            float(bb.findtext("wms:northBoundLatitude", namespaces=NS)),
        ]
    except (TypeError, ValueError):
        return None


async def _embed_texts(texts: list[str]) -> np.ndarray:
    BATCH = 64
    all_vecs = []

    async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            resp = await client.post(
                f"{EMBED_URL}/embeddings",
                headers={"Authorization": f"Bearer {EMBED_API_KEY}"},
                json={"model": EMBED_MODEL, "input": batch},
            )
            resp.raise_for_status()
            data = resp.json()
            items = sorted(data["data"], key=lambda x: x["index"])
            vecs = np.array([item["embedding"] for item in items], dtype=np.float32)
            all_vecs.append(vecs)

    matrix = np.vstack(all_vecs)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


async def _build_embedding_index(rows: list[tuple]):
    global _embed_matrix, _embed_names

    texts = []
    names = []
    for name, title, abstract, _ in rows:
        abstract_seguro = abstract[:25000] if abstract else ""
        text = f"{title}. {abstract}".strip(". ") if abstract else title
        texts.append(text)
        names.append(name)

    logger.info(f"[embed] Gerando embeddings para {len(texts)} camadas...")
    matrix = await _embed_texts(texts)
    _embed_matrix = matrix
    _embed_names = names
    logger.info(f"[embed] Índice de camadas pronto — shape {matrix.shape}")


async def build_tool_index() -> None:
    global _tool_embed_matrix, _tool_embed_index

    all_phrases: list[str] = []
    all_tool_names: list[str] = []

    for tool_name, phrases in _TOOL_INTENT_PHRASES.items():
        for phrase in phrases:
            all_phrases.append(phrase)
            all_tool_names.append(tool_name)

    logger.info(f"[tool-embed] Gerando embeddings para {len(all_phrases)} frases de intenção...")
    try:
        matrix = await _embed_texts(all_phrases)
        _tool_embed_matrix = matrix
        _tool_embed_index = all_tool_names
        logger.info(f"[tool-embed] Índice de tools pronto — shape {matrix.shape}")
    except Exception as e:
        logger.warning(f"[tool-embed] Falha ao gerar embeddings de tools: {e}. Retrieval desativado.")



def retrieve_tools(query_vec: np.ndarray, tool_defs: list[dict], k: int = 3) -> list[dict]:
    if _tool_embed_matrix is None or not _tool_embed_index:
        logger.warning("[tool-embed] Índice não disponível — usando todas as tools.")
        return tool_defs

    # Similaridade cosine: query_vec (D,) @ matrix.T → (P,)
    scores = _tool_embed_matrix @ query_vec  # shape (P,)

    # Max-pooling por tool: agrupa scores por nome de tool
    tool_scores: dict[str, float] = {}
    for idx, tool_name in enumerate(_tool_embed_index):
        current = tool_scores.get(tool_name, -1.0)
        if scores[idx] > current:
            tool_scores[tool_name] = float(scores[idx])

    # Ordena por score e pega top-k
    ranked = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
    top_names = {name for name, _ in ranked[:k]}

    logger.info(
        f"[tool-embed] Retrieval top-{k}: "
        + ", ".join(f"{n}={s:.3f}" for n, s in ranked[:k])
    )

    # Filtra as tool_defs preservando a ordem original (mais previsível pro LLM)
    selected = [t for t in tool_defs if t["function"]["name"] in top_names]
    return selected


def _build_sqlite_index(raw_layers: list[dict]):
    global _db, _layers_cache

    _db = sqlite3.connect(":memory:")
    _db.execute("PRAGMA journal_mode=OFF")
    _db.execute("PRAGMA synchronous=OFF")

    _db.execute("CREATE TABLE layers (name TEXT, title TEXT, abstract TEXT, bbox TEXT)")
    _db.execute("CREATE UNIQUE INDEX idx_layers_name ON layers(name)")
    _db.execute(
        """CREATE VIRTUAL TABLE layers_fts USING fts5(
            name, title, abstract,
            content='layers', content_rowid='rowid',
            tokenize='unicode61 remove_diacritics 2'
        )"""
    )

    seen = set()
    rows = []
    for layer in raw_layers:
        if layer["name"] in seen:
            continue
        seen.add(layer["name"])
        rows.append(
            (
                layer["name"],
                layer["title"],
                layer["abstract"],
                json.dumps(layer["bbox"]),
            )
        )

    _db.executemany(
        "INSERT INTO layers (name, title, abstract, bbox) VALUES (?,?,?,?)", rows
    )
    _db.execute("INSERT INTO layers_fts(layers_fts) VALUES('rebuild')")
    _db.commit()

    _layers_cache = [{"name": r[0], "title": r[1]} for r in rows]


async def load_capabilities():
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.get(CAPABILITIES_URL)
        resp.raise_for_status()

    root = ET.fromstring(resp.text)
    raw_layers = []
    for layer_el in root.iter(f"{{{NS['wms']}}}Layer"):
        name = layer_el.findtext("wms:Name", namespaces=NS)
        if not name:
            continue
        raw_layers.append(
            {
                "name": name,
                "title": layer_el.findtext("wms:Title", namespaces=NS) or name,
                "abstract": layer_el.findtext("wms:Abstract", namespaces=NS) or "",
                "bbox": _parse_bbox(layer_el),
            }
        )

    rows = _build_sqlite_index(raw_layers)
    logger.info(f"WMS: loaded {len(_layers_cache)} layers")

    try:
        await _build_embedding_index(rows)
    except Exception as e:
        logger.warning(f"[embed] falha ao gerar embeding de camadas : {e}")

    await build_tool_index()

def get_all_layers() -> list[dict]:
    return _layers_cache


def _fts5_search(query: str, limit: int = 50) -> list[dict]:
    if _db is None:
        return []
    tokens = query.split()

    if not tokens:
        return []
    
    sanitized = []
    for t in tokens:
        t = t.replace('"', "").replace("*", "").replace("(", "").replace(")", "")
        if t:
            sanitized.append(f'"{t}"*')
    if not sanitized:
        return []
    fts_query = " AND ".join(sanitized)
    try:
        rows = _db.execute(
            "SELECT l.name, l.title FROM layers_fts f "
            "JOIN layers l ON l.rowid = f.rowid "
            "WHERE layers_fts MATCH ? "
            "ORDER BY bm25(layers_fts) LIMIT ?",
            (fts_query, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [{"name": r[0], "title": r[1]} for r in rows]

def _fuzzy_search(query: str, limit: int = 50) -> list[dict]:
    if _db is None:
        return []
    all_rows = _db.execute("SELECT name, title FROM layers").fetchall()
    choices = {i: f"{r[0]} {r[1]}" for i, r in enumerate(all_rows)}
    matches = process.extract(
        query, choices, scorer=fuzz.WRatio, limit=limit, score_cutoff=60
    )
    return [{"name": all_rows[m[2]][0], "title": all_rows[m[2]][1]} for m in matches]


def _embedding_search(query_vec: np.ndarray, limit: int = 10) -> list[dict]:
    if _embed_matrix is None or len(_embed_names) == 0:
        return []

    scores = _embed_matrix @ query_vec
    top_idx = np.argpartition(scores, -limit)[-limit:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    results = []
    for idx in top_idx:
        row = _db.execute(
            "SELECT name, title FROM layers WHERE name = ?",
            (_embed_names[idx],),
        ).fetchone()
        if row:
            results.append({"name": row[0], "title": row[1], "score": float(scores[idx])})
    return results


async def search_layers_async(query: str) -> list[dict]:
    seen: set[str] = set()
    merged: list[dict] = []

    if _embed_matrix is not None:
        try:
            query_vec = await _embed_texts([query])
            results = _embedding_search(query_vec[0], limit=10)
            for r in results:
                if r["name"] not in seen:
                    seen.add(r["name"])
                    merged.append({"name": r["name"], "title": r["title"]})
        except Exception as e:
            logger.warning(f"[embed] Erro na busca por embedding: {e}")

    for r in _fts5_search(query, limit=20):
        if r["name"] not in seen:
            seen.add(r["name"])
            merged.append(r)

    if len(merged) < 3:
        logger.warning(f"[search] Poucos resultados ({len(merged)}), ativando fuzzy")
        for r in _fuzzy_search(query):
            if r["name"] not in seen:
                seen.add(r["name"])
                merged.append(r)

    return merged[:10]


def search_layers(query: str) -> list[dict]:
    fts_results = _fts5_search(query)
    if len(fts_results) >= 5:
        return fts_results

    fuzzy_results = _fuzzy_search(query)
    seen = {r["name"] for r in fts_results}
    merged = list(fts_results)
    for r in fuzzy_results:
        if r["name"] not in seen:
            merged.append(r)
            seen.add(r["name"])
    return merged[:10]

def get_layer_info(name: str) -> dict | None:
    if _db is None:
        return None
    row = _db.execute(
        "SELECT name, title, abstract, bbox FROM layers WHERE name = ?", (name,)
    ).fetchone()
    if row is None:
        return None
    return {
        "name": row[0],
        "title": row[1],
        "abstract": row[2],
        "bbox": json.loads(row[3]) if row[3] else None,
        "url": WMS_URL,
    }

def get_layer_columns(layer_name: str) -> list[str]:
    url = f"{WFS_URL}?service=WFS&version=1.0.0&request=DescribeFeatureType&typeName={layer_name}"

    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(url)
            resp.raise_for_status()

        root = ET.fromstring(resp.text)

        columns =[]

        ns = {"xsd": "http://www.w3.org/2001/XMLSchema"}

        for el in root.findall(".//xsd:element", namespaces=ns):
            col_name = el.attrib.get("name")
            col_type = el.attrib.get("type", "")

            if col_name and not col_type.startswith("gml:"):
                columns.append(col_name)
        return columns
    except Exception as e:
        print(f"Erro ao buscar colunas da camada {layer_name}: {e}")
        return []
    
def get_feature_bbox(layer_name: str, cql_filter: str) -> list[float] | None:
    encoded_filter = urllib.parse.quote(cql_filter)

    url = f"{WFS_URL}?service=WFS&version=1.0.0&request=GetFeature&typeName={layer_name}&CQL_FILTER={encoded_filter}&outputFormat=application/json"

    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()

            if not data.get("features"):
                return None
            
            if "bbox" in data:
                return data["bbox"]
            
            coords = []
            def extract_coords(geom_data):
                if isinstance(geom_data, list):
                    if len(geom_data) >= 2 and isinstance(geom_data[0], (int, float)):
                        coords.append(geom_data)
                    else:
                        for item in geom_data:
                            extract_coords(item)
                
            for feature in data["features"]:
                geom = feature.get("geometry")
                if geom and "coordinates" in geom:
                    extract_coords(geom["coordinates"])
            
            if coords:
                min_x = min(c[0] for c in coords)
                min_y = min(c[1] for c in coords)
                max_x = max(c[0] for c in coords)
                max_y = max(c[1] for c in coords)
                return [min_x, min_y, max_x, max_y]
    
    except Exception as e:
        print(f"Erro ao buscar bbox da feição de {layer_name} com filtro {cql_filter}")
    
    return None

                