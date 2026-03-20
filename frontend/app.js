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
  renderLayerList();
}

function removeWmsLayer(name) {
  const layer = wmsLayers[name];
  if (layer) {
    map.removeLayer(layer);
    delete wmsLayers[name];
    renderLayerList();
  }
}

function zoomToExtent(bbox) {
  if (!bbox || bbox.length !== 4) return;
  const extent = ol.proj.transformExtent(bbox, "EPSG:4326", "EPSG:3857");
  map.getView().fit(extent, { duration: 500, padding: [50, 50, 50, 50] });
}

function applyLayerFilter(name, cqlFilter) {
  const layer = wmsLayers[name];
  if (layer) {
    const source = layer.getSource();
    layer.set('current_filter', cqlFilter || null); 

    source.updateParams({ 
      'CQL_FILTER': cqlFilter || undefined 
    });
    
    console.log(`Filtro CQL em ${name}: ${cqlFilter || 'REMOVIDO'}`);
    renderLayerList();
  }
}

// --- GetFeatureInfo popup ---
const popupEl = document.getElementById("popup");
const popupContent = document.getElementById("popup-content");
const popupClose = document.getElementById("popup-close");

const overlay = new ol.Overlay({
  element: popupEl,
  autoPan: { animation: { duration: 200 } },
  positioning: "bottom-center",
  offset: [0, -12],
});
map.addOverlay(overlay);

function closePopup() {
  popupEl.classList.remove("visible");
  overlay.setPosition(undefined);
}

popupClose.addEventListener("click", closePopup);

// Skip attributes that are not useful to display
const SKIP_KEYS = new Set(["geometry", "geom", "the_geom", "shape", "bbox", "gid"]);

function featureToTable(properties) {
  const rows = Object.entries(properties)
    .filter(([k, v]) => !SKIP_KEYS.has(k.toLowerCase()) && v != null && v !== "")
    .map(([k, v]) => `<tr><th>${k}</th><td>${v}</td></tr>`)
    .join("");
  return rows ? `<table>${rows}</table>` : "";
}

map.on("singleclick", async (evt) => {
  const visibleLayers = Object.entries(wmsLayers).filter(([, l]) => l.getVisible());
  if (visibleLayers.length === 0) return;

  const viewResolution = map.getView().getResolution();
  const coordinate = evt.coordinate;

  // Show loading state
  popupContent.innerHTML = '<div class="popup-loading">Consultando...</div>';
  popupEl.classList.add("visible");
  overlay.setPosition(coordinate);

  let html = "";

  const queries = visibleLayers.map(async ([name, layer]) => {
    const source = layer.getSource();
    const url = source.getFeatureInfoUrl(coordinate, viewResolution, "EPSG:3857", {
      INFO_FORMAT: "application/json",
      FEATURE_COUNT: 5,
    });
    if (!url) return null;

    try {
      const resp = await fetch(url);
      if (!resp.ok) return null;
      const data = await resp.json();
      if (!data.features || data.features.length === 0) return null;

      const title = layer.get("title") || name;
      let section = `<div class="popup-layer-title">${title}</div>`;
      for (const feature of data.features) {
        const table = featureToTable(feature.properties || {});
        if (table) section += `<div class="popup-feature">${table}</div>`;
      }
      return section;
    } catch {
      return null;
    }
  });

  const results = (await Promise.all(queries)).filter(Boolean);

  if (results.length > 0) {
    popupContent.innerHTML = results.join("");
  } else {
    closePopup();
  }
});

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
      case "apply_cql_filter":
        applyLayerFilter(action.name, action.filter);
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
  if (role.startsWith("assistant")) {
    div.innerHTML = marked.parse(text);
  } else {
    div.textContent = text;
  }
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
          filter: layer.get("current_filter") || null
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

document.getElementById("chat-clear").addEventListener("click", () => {
  sessionId = null;
  messagesEl.innerHTML = "";
  // Remove all WMS layers from map
  for (const name of Object.keys(wmsLayers)) {
    map.removeLayer(wmsLayers[name]);
    delete wmsLayers[name];
  }
  renderLayerList();
  inputEl.focus();
});

// --- Layer panel ---
const layerPanel = document.getElementById("layer-panel");
const layerListEl = document.getElementById("layer-list");
const layerCountEl = document.getElementById("layer-count");
const chatContainer = document.getElementById("chat-container");

document.getElementById("layer-panel-header").addEventListener("click", () => {
  layerPanel.classList.toggle("collapsed");
});

function renderLayerList() {
  const entries = Object.entries(wmsLayers);
  layerCountEl.textContent = entries.length;

  if (entries.length === 0) {
    chatContainer.classList.remove("has-layers");
  } else {
    chatContainer.classList.add("has-layers");
  }

  layerListEl.innerHTML = "";
  for (const [name, layer] of entries) {
    const div = document.createElement("div");
    div.className = "layer-item" + (layer.getVisible() ? "" : " hidden-layer");

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = layer.getVisible();
    cb.id = "layer-cb-" + name;
    cb.addEventListener("change", () => {
      layer.setVisible(cb.checked);
      div.classList.toggle("hidden-layer", !cb.checked);
    });

    const lbl = document.createElement("label");
    lbl.htmlFor = cb.id;
    lbl.textContent = layer.get("title") || name;

    div.appendChild(cb);
    div.appendChild(lbl);
    layerListEl.appendChild(div);
  }
}
