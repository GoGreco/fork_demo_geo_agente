const WMS_BASE = "https://geoservicos.ibge.gov.br/geoserver/wms";

// --- Map setup ---
const map = new ol.Map({
  target: "map",
  layers: [
    new ol.layer.Tile({
      source: new ol.source.OSM(),
    }),
  ],
  view: new ol.View({
    center: ol.proj.fromLonLat([-47.9, -15.8]),
    zoom: 4,
  }),
});

// Track WMS layers by name
const wmsLayers = {};

function addWmsLayer(name, title) {
  if (wmsLayers[name]) return;
  const layer = new ol.layer.Tile({
    source: new ol.source.TileWMS({
      url: WMS_BASE,
      params: {
        LAYERS: name,
        TILED: true,
        FORMAT: "image/png",
        TRANSPARENT: true,
      },
      serverType: "geoserver",
    }),
    opacity: 0.7,
  });
  layer.set("name", name);
  layer.set("title", title);
  map.addLayer(layer);
  wmsLayers[name] = layer;
}

function removeWmsLayer(name) {
  const layer = wmsLayers[name];
  if (layer) {
    map.removeLayer(layer);
    delete wmsLayers[name];
  }
}

function zoomToExtent(bbox) {
  if (!bbox || bbox.length !== 4) return;
  const extent = ol.proj.transformExtent(bbox, "EPSG:4326", "EPSG:3857");
  map.getView().fit(extent, { duration: 500, padding: [50, 50, 50, 50] });
}

// --- Process actions from agent ---
function processActions(actions) {
  for (const action of actions) {
    switch (action.type) {
      case "add_layer":
        addWmsLayer(action.name, action.title);
        if (action.bbox) zoomToExtent(action.bbox);
        break;
      case "remove_layer":
        removeWmsLayer(action.name);
        break;
      case "zoom_to_layer":
        if (action.bbox) zoomToExtent(action.bbox);
        break;
    }
  }
}

// --- Chat ---
const messagesEl = document.getElementById("chat-messages");
const inputEl = document.getElementById("chat-input");
const sendBtn = document.getElementById("chat-send");

let sessionId = null;

async function ensureSession() {
  if (sessionId) return;
  const resp = await fetch("/api/session", { method: "POST" });
  const data = await resp.json();
  sessionId = data.session_id;
}

function appendMessage(text, role) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.value = "";
  sendBtn.disabled = true;
  appendMessage(text, "user");
  const loadingEl = appendMessage("Pensando...", "assistant loading");

  try {
    await ensureSession();
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message: text,
        active_layers: Object.entries(wmsLayers).map(([name, layer]) => ({
          name,
          title: layer.get("title") || name,
        })),
      }),
    });
    const data = await resp.json();

    loadingEl.remove();
    if (data.reply) {
      appendMessage(data.reply, "assistant");
    }
    if (data.actions && data.actions.length > 0) {
      processActions(data.actions);
    }
  } catch (err) {
    loadingEl.remove();
    appendMessage("Erro ao conectar com o servidor.", "assistant");
    console.error(err);
  } finally {
    sendBtn.disabled = false;
    inputEl.focus();
  }
}

sendBtn.addEventListener("click", sendMessage);
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});
