# Agente Geo IBGE

Aplicação web com um agente de IA conversacional que permite explorar camadas geográficas do IBGE diretamente em um mapa interativo.

## Como funciona

A aplicação é composta por duas partes:

**Backend (FastAPI)**

- Na inicialização, faz download do XML de capabilities do serviço WMS do IBGE (`geoservicos.ibge.gov.br`) e indexa as ~9000 camadas disponíveis em um banco SQLite em memória com busca full-text (FTS5) e fuzzy matching (rapidfuzz).
- Expõe um endpoint de chat (`POST /api/chat`) que recebe mensagens do usuário e as encaminha para um LLM (Qwen 3 8B via OpenRouter) com acesso a ferramentas: `search_layers`, `list_layers`, `get_layer_info`, `add_layer`, `remove_layer` e `zoom_to_layer`.
- O agente usa tool calling para buscar camadas relevantes e retorna ações (adicionar/remover camada, ajustar zoom) que o frontend executa no mapa.
- Também serve os arquivos estáticos do frontend.

**Frontend (HTML + OpenLayers)**

- Mapa interativo usando OpenLayers com base OSM.
- Painel de chat lateral onde o usuário conversa em linguagem natural.
- Ao receber ações do backend, adiciona/remove camadas WMS do IBGE no mapa e ajusta o zoom automaticamente.

## Fluxo de trabalho do agente

Abaixo está o ciclo completo de uma interação, desde a mensagem do usuário até a atualização do mapa.

```
Usuário digita mensagem
        │
        ▼
Frontend ── POST /api/chat { session_id, message } ──► Backend (agent.py)
                                                            │
                                                    ┌───────┴───────┐
                                                    │  Agentic loop │
                                                    │  (máx. 10x)   │
                                                    └───────┬───────┘
                                                            │
                                              Envia histórico + tools ao LLM
                                                            │
                                                  ┌─────────┴─────────┐
                                                  │                   │
                                            tool_calls          texto final
                                                  │                   │
                                          Executa tools          Encerra loop
                                                  │                   │
                             ┌─────────────────── ┤                   │
                             │                    │                   │
                        Consulta:             Ação (mapa):            │
                      search_layers          add_layer                │
                      list_layers            remove_layer             │
                      get_layer_info         zoom_to_layer            │
                             │                    │                   │
                     Resultado → histórico        │                   │
                     → próxima iteração           │                   │
                                                  │                   │
                                                  ▼                   │
Frontend ◄──────── { reply, actions[] } ──────────┴───────────────────┘
  │
  ├─ Exibe reply no chat
  └─ Executa actions no mapa (adicionar/remover camada, zoom)
```

### Exemplo concreto

O usuário digita: **"Mostre os municípios do Brasil"**

1. O LLM recebe a mensagem e decide chamar `search_layers("municipio")`.
2. O backend executa a busca FTS5/fuzzy e retorna uma lista de camadas que contêm "municipio" no nome ou título.
3. O LLM analisa os resultados e escolhe a camada mais relevante (ex: `CCAR:BC250_Municipio_A`). Chama `add_layer(name="CCAR:BC250_Municipio_A", title="Municípios")`.
4. O backend registra a ação `{ type: "add_layer", name: "CCAR:BC250_Municipio_A", bbox: [...] }` e retorna o resultado da tool ao LLM.
5. O LLM gera uma resposta final em texto: _"Adicionei a camada de municípios ao mapa."_
6. O backend retorna `{ reply: "Adicionei a camada...", actions: [{ type: "add_layer", ... }] }`.
7. O frontend exibe a mensagem no chat, adiciona a camada WMS ao mapa e ajusta o zoom para o Brasil.

### Detalhes do agentic loop

- O loop permite até **10 iterações** de tool calling por mensagem, o que possibilita fluxos multi-step (ex: buscar → inspecionar → adicionar → dar zoom).
- O histórico de conversa é mantido **em memória por sessão** (`session_id`), preservando contexto entre mensagens.
- As ferramentas `search_layers`, `list_layers` e `get_layer_info` são puramente informativas (não geram ações no mapa). Já `add_layer`, `remove_layer` e `zoom_to_layer` acumulam ações que serão executadas pelo frontend.

## Indexação e busca de camadas

Na inicialização do servidor, o backend faz download do XML de capabilities do serviço WMS do IBGE (`GetCapabilities`) e extrai nome, título, resumo e bounding box de cada camada. Essas ~9000 camadas são então indexadas em um banco SQLite em memória através de duas estruturas:

**Full-Text Search (FTS5)**

Uma tabela virtual FTS5 indexa os campos `name`, `title` e `abstract` de cada camada. O tokenizer `unicode61` é configurado com `remove_diacritics 2`, o que permite encontrar resultados independente de acentuação (ex: "municipio" encontra "Município"). Cada termo da busca é tratado como prefixo (`"termo"*`), e múltiplos termos são combinados com `AND`. Os resultados são ranqueados pelo algoritmo BM25, que prioriza as camadas mais relevantes.

**Fuzzy matching (rapidfuzz)**

Quando a busca FTS5 retorna poucos resultados (menos de 5), o sistema complementa com uma busca fuzzy usando a biblioteca `rapidfuzz`. Ela compara a query contra todas as camadas usando o scorer `WRatio` (que combina diferentes estratégias de similaridade de strings) e retorna matches com score acima de 60%. Isso permite encontrar camadas mesmo com erros de digitação ou variações de nomenclatura.

**Estratégia de busca combinada**

A função `search_layers` combina as duas abordagens: primeiro executa a busca FTS5 (rápida e precisa). Se os resultados forem suficientes (≥ 5), retorna apenas eles. Caso contrário, complementa com os resultados fuzzy, removendo duplicatas, e retorna até 50 camadas.

## Pré-requisitos

- Python 3.11+
- Uma API key do [OpenRouter](https://openrouter.ai)

## Configuração

1. Clone o repositório e entre no diretório:

```bash
git clone <url-do-repo>
cd agente_ui_demo
```

2. Crie e ative o ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Instale as dependências:

```bash
pip install -r backend/requirements.txt
```

4. Crie o arquivo `.config` na raiz do projeto com as configurações do LLM:

```ini
[llm]
api_key = sk-or-...
base_url = https://openrouter.ai/api/v1
model = qwen/qwen3-8b
```

Todos os campos possuem valores padrão (base_url aponta para OpenRouter e model para `qwen/qwen3-8b`), então apenas `api_key` é obrigatório.

## Executando

Inicie o servidor de desenvolvimento a partir do diretório `backend/`:

```bash
cd backend
uvicorn main:app --reload --port 8000 2>&1 | tee log.txt
```

Acesse a aplicação em [http://localhost:8000](http://localhost:8000).

## Estrutura do projeto

```
├── .config              # Configuração do LLM: api_key, base_url, model (não versionado)
├── backend/
│   ├── main.py          # Servidor FastAPI, rotas e servir frontend
│   ├── agent.py         # Agente LLM com tool calling
│   ├── wms.py           # Parser de WMS capabilities + busca FTS5/fuzzy
│   └── requirements.txt
└── frontend/
    ├── index.html       # Página principal
    ├── app.js           # Mapa OpenLayers + chat + execução de ações
    └── style.css        # Estilos
```
