import sqlite3
import json
import urllib.parse

import httpx
import xml.etree.ElementTree as ET
from rapidfuzz import fuzz, process

#WMS_URL = "http://localhost:8180/geoserver/wms"
WMS_URL = "https://geoservicos.ibge.gov.br/geoserver/wms"
#TENTATIVA FILTRO
WFS_URL = "https://geoservicos.ibge.gov.br/geoserver/wfs"
CAPABILITIES_URL = f"{WMS_URL}?service=WMS&request=GetCapabilities"
NS = {"wms": "http://www.opengis.net/wms"}

_db: sqlite3.Connection | None = None
_layers_cache: list[dict] = []


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


def _build_index(raw_layers: list[dict]):
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
    async with httpx.AsyncClient(timeout=30) as client:
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

    _build_index(raw_layers)
    print(f"WMS: loaded {len(_layers_cache)} layers")


def get_all_layers() -> list[dict]:
    return _layers_cache


def _fts5_search(query: str, limit: int = 50) -> list[dict]:
    if _db is None:
        return []
    tokens = query.split()
    if not tokens:
        return []
    # Escape FTS5 operators and wrap each token as a prefix query
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
    return merged[:50]


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

#WFS TESTE
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

                