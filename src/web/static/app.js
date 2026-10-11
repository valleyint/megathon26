/**
 * CommonsCat AI Studio · Frontend Logic & Interactive Controls
 */

document.addEventListener("DOMContentLoaded", () => {
  // State
  let presets = [];
  let currentPresetId = null;
  let customImageDataUrl = null;
  let lastPipelineResult = null;
  let lastStage1Result = null;
  let activeAuditFilter = "all";

  // Elements
  const presetsContainer = document.getElementById("presetsContainer");
  const imageDisplay = document.getElementById("imageDisplay");
  const imageUploadInput = document.getElementById("imageUploadInput");
  const titleInput = document.getElementById("titleInput");
  const descInput = document.getElementById("descInput");
  const timestampInput = document.getElementById("timestampInput");
  const yearPill = document.getElementById("yearPill");
  const latInput = document.getElementById("latInput");
  const lonInput = document.getElementById("lonInput");
  const geoCityText = document.getElementById("geoCityText");
  const queryIdBadge = document.getElementById("queryIdBadge");

  // Hyperparameters
  const knnWeightInput = document.getElementById("knnWeightInput");
  const knnWeightVal = document.getElementById("knnWeightVal");
  const protoWeightInput = document.getElementById("protoWeightInput");
  const protoWeightVal = document.getElementById("protoWeightVal");
  const poolSizeInput = document.getElementById("poolSizeInput");
  const poolSizeVal = document.getElementById("poolSizeVal");
  const yearTolInput = document.getElementById("yearTolInput");
  const yearTolVal = document.getElementById("yearTolVal");
  const childThreshInput = document.getElementById("childThreshInput");
  const childThreshVal = document.getElementById("childThreshVal");
  const strictCountryInput = document.getElementById("strictCountryInput");

  // Buttons
  const runPipelineBtn = document.getElementById("runPipelineBtn");
  const runStage1Btn = document.getElementById("runStage1Btn");
  const runStage2Btn = document.getElementById("runStage2Btn");
  const copyAllTagsBtn = document.getElementById("copyAllTagsBtn");
  const exportJsonBtn = document.getElementById("exportJsonBtn");
  const themeToggle = document.getElementById("themeToggle");
  const toast = document.getElementById("toast");

  // KPI elements
  const kpiLatency = document.getElementById("kpiLatency");
  const kpiS1Count = document.getElementById("kpiS1Count");
  const kpiS2Count = document.getElementById("kpiS2Count");
  const kpiPrunedCount = document.getElementById("kpiPrunedCount");
  const headerLatencyVal = document.getElementById("headerLatencyVal");

  // Tab elements
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanes = document.querySelectorAll(".tab-pane");

  // Setup Event Listeners
  setupEventListeners();

  // Load Presets from Backend
  loadPresets();

  function setupEventListeners() {
    // Theme toggle
    themeToggle.addEventListener("click", () => {
      document.body.classList.toggle("light-theme");
      const isLight = document.body.classList.contains("light-theme");
      document.getElementById("moonIcon").classList.toggle("hidden", isLight);
      document.getElementById("sunIcon").classList.toggle("hidden", !isLight);
      localStorage.setItem("commonscat_theme", isLight ? "light" : "dark");
    });

    if (localStorage.getItem("commonscat_theme") === "light") {
      document.body.classList.add("light-theme");
      document.getElementById("moonIcon").classList.add("hidden");
      document.getElementById("sunIcon").classList.remove("hidden");
    }

    // Sliders
    knnWeightInput.addEventListener("input", (e) => {
      knnWeightVal.textContent = parseFloat(e.target.value).toFixed(2);
    });
    protoWeightInput.addEventListener("input", (e) => {
      protoWeightVal.textContent = parseFloat(e.target.value).toFixed(2);
    });
    poolSizeInput.addEventListener("input", (e) => {
      poolSizeVal.textContent = e.target.value;
    });
    yearTolInput.addEventListener("input", (e) => {
      yearTolVal.textContent = `${e.target.value} yr`;
    });
    childThreshInput.addEventListener("input", (e) => {
      childThreshVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Inputs
    timestampInput.addEventListener("input", (e) => {
      const yr = extractYear(e.target.value);
      yearPill.textContent = yr || "--";
    });

    latInput.addEventListener("input", updateGeoPreview);
    lonInput.addEventListener("input", updateGeoPreview);

    // Image Upload
    imageUploadInput.addEventListener("change", (e) => {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
          customImageDataUrl = event.target.result;
          imageDisplay.innerHTML = `<img src="${customImageDataUrl}" alt="Query Upload" style="width:100%;height:100%;object-fit:cover;">`;
          currentPresetId = null;
          document.querySelectorAll(".preset-chip").forEach((el) => el.classList.remove("active"));
          queryIdBadge.textContent = "custom_upload_" + Date.now().toString().slice(-4);
        };
        reader.readAsDataURL(file);
      }
    });

    // Action buttons
    runPipelineBtn.addEventListener("click", () => executePipeline());
    runStage1Btn.addEventListener("click", () => executeStage1Only());
    runStage2Btn.addEventListener("click", () => executeStage2Only());

    // Copy & Export
    copyAllTagsBtn.addEventListener("click", copyAllWikitextTags);
    exportJsonBtn.addEventListener("click", exportResultsJson);

    // Tabs
    tabBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        const targetTab = btn.dataset.tab;
        tabBtns.forEach((b) => b.classList.remove("active"));
        tabPanes.forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(targetTab).classList.add("active");
        if (targetTab === "tabDag" && lastPipelineResult) {
          renderDagGraph(lastPipelineResult);
        }
      });
    });

    // Audit filter pills
    document.querySelectorAll(".filter-pill").forEach((pill) => {
      pill.addEventListener("click", () => {
        document.querySelectorAll(".filter-pill").forEach((p) => p.classList.remove("active"));
        pill.classList.add("active");
        activeAuditFilter = pill.dataset.filter;
        if (lastPipelineResult && lastPipelineResult.stage2) {
          renderAuditTrail(lastPipelineResult.stage2.pruned_decisions);
        }
      });
    });
  }

  function extractYear(val) {
    if (!val) return null;
    const match = val.match(/\b(18|19|20)\d{2}\b/);
    return match ? match[0] : null;
  }

  function updateGeoPreview() {
    const lat = parseFloat(latInput.value);
    const lon = parseFloat(lonInput.value);
    if (isNaN(lat) || isNaN(lon)) {
      geoCityText.textContent = "Geotag not specified";
      return;
    }
    // Simple client proximity lookup
    const cities = [
      { name: "Paris, Île-de-France, France (FR)", lat: 48.8566, lon: 2.3522 },
      { name: "London, England, United Kingdom (GB)", lat: 51.5074, lon: -0.1278 },
      { name: "Rome, Lazio, Italy (IT)", lat: 41.9028, lon: 12.4964 },
      { name: "Agra, Uttar Pradesh, India (IN)", lat: 27.1751, lon: 78.0421 },
      { name: "Berlin, Berlin, Germany (DE)", lat: 52.5200, lon: 13.4050 },
    ];
    let best = cities[0];
    let minD = 999999;
    cities.forEach((c) => {
      const d = Math.hypot(lat - c.lat, lon - c.lon);
      if (d < minD) {
        minD = d;
        best = c;
      }
    });
    if (minD < 1.0) {
      geoCityText.textContent = best.name;
    } else {
      geoCityText.textContent = `Lat: ${lat.toFixed(4)}, Lon: ${lon.toFixed(4)}`;
    }
  }

  async function loadPresets() {
    try {
      const resp = await fetch("/api/presets");
      const data = await resp.json();
      presets = data.presets || [];
      renderPresets();
      if (presets.length > 0) {
        selectPreset(presets[0].id);
      }
    } catch (e) {
      console.error("Failed to load presets:", e);
    }
  }

  function renderPresets() {
    presetsContainer.innerHTML = "";
    presets.forEach((p) => {
      const chip = document.createElement("div");
      chip.className = "preset-chip";
      chip.dataset.id = p.id;
      chip.innerHTML = `
        <span>${p.title}</span>
        <span class="preset-badge">${p.badge}</span>
      `;
      chip.addEventListener("click", () => selectPreset(p.id));
      presetsContainer.appendChild(chip);
    });
  }

  function selectPreset(presetId) {
    const preset = presets.find((p) => p.id === presetId);
    if (!preset) return;
    currentPresetId = presetId;
    customImageDataUrl = null;

    document.querySelectorAll(".preset-chip").forEach((chip) => {
      chip.classList.toggle("active", chip.dataset.id === presetId);
    });

    // Populate inputs
    titleInput.value = preset.query.title;
    descInput.value = preset.query.description;
    timestampInput.value = preset.query.timestamp;
    yearPill.textContent = extractYear(preset.query.timestamp) || "--";
    latInput.value = preset.query.latitude;
    lonInput.value = preset.query.longitude;
    queryIdBadge.textContent = preset.id;

    // Display preset SVG illustration
    if (preset.svg) {
      imageDisplay.innerHTML = preset.svg;
    }

    updateGeoPreview();

    // Auto run pipeline for instantaneous visual satisfaction
    executePipeline();
  }

  function getQueryParams() {
    return {
      query: {
        preset_id: currentPresetId,
        query_id: queryIdBadge.textContent,
        title: titleInput.value,
        description: descInput.value,
        timestamp: timestampInput.value,
        latitude: parseFloat(latInput.value) || null,
        longitude: parseFloat(lonInput.value) || null,
        image_data: customImageDataUrl,
      },
      params: {
        knn_weight: parseFloat(knnWeightInput.value),
        proto_weight: parseFloat(protoWeightInput.value),
        pool_size: parseInt(poolSizeInput.value),
        year_tolerance: parseInt(yearTolInput.value),
        child_confidence_threshold: parseFloat(childThreshInput.value),
        strict_country_check: strictCountryInput.checked,
      },
    };
  }

  async function executePipeline() {
    setLoadingState(true);
    const payload = getQueryParams();

    try {
      const resp = await fetch("/api/pipeline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        throw new Error(await resp.text());
      }

      const data = await resp.json();
      lastPipelineResult = data;
      lastStage1Result = data.stage1;

      renderPipelineResults(data);
    } catch (e) {
      console.error("Pipeline run error:", e);
      showToast(`Error: ${e.message}`);
    } finally {
      setLoadingState(false);
    }
  }

  async function executeStage1Only() {
    setLoadingState(true);
    const payload = getQueryParams();

    try {
      const resp = await fetch("/api/stage1", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        throw new Error(await resp.text());
      }

      const s1 = await resp.json();
      lastStage1Result = s1;

      // Update KPI
      kpiS1Count.textContent = s1.candidates.length;
      kpiLatency.textContent = `${s1.latency_ms.stage1_total_ms || 24.5} ms`;

      // Render Stage 1 tab and switch to it
      renderStage1Tab(s1);
      switchTab("tabStage1");
      showToast("Stage 1 visual recall completed!");
    } catch (e) {
      console.error("Stage 1 error:", e);
      showToast(`Error: ${e.message}`);
    } finally {
      setLoadingState(false);
    }
  }

  async function executeStage2Only() {
    if (!lastStage1Result) {
      showToast("Running Stage 1 first to generate candidate pool...");
      await executePipeline();
      return;
    }

    setLoadingState(true);
    const payload = getQueryParams();
    payload.stage1_pool = lastStage1Result;

    try {
      const resp = await fetch("/api/stage2", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        throw new Error(await resp.text());
      }

      const s2 = await resp.json();
      if (!lastPipelineResult) lastPipelineResult = {};
      lastPipelineResult.stage2 = s2;

      // Update KPI
      kpiS2Count.textContent = s2.candidates.length;
      const prunedCount = s2.pruned_decisions.filter((d) => d.status !== "retained").length;
      kpiPrunedCount.textContent = prunedCount;

      renderStage2Tab(s2);
      renderOverviewTab(s2);
      switchTab("tabStage2");
      showToast("Stage 2 pruning completed!");
    } catch (e) {
      console.error("Stage 2 error:", e);
      showToast(`Error: ${e.message}`);
    } finally {
      setLoadingState(false);
    }
  }

  function setLoadingState(loading) {
    runPipelineBtn.disabled = loading;
    runStage1Btn.disabled = loading;
    runStage2Btn.disabled = loading;
    if (loading) {
      runPipelineBtn.innerHTML = `
        <svg class="spin" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10" stroke-opacity="0.25"/><path d="M12 2a10 10 0 0 1 10 10"/></svg>
        Executing Stages...
      `;
    } else {
      runPipelineBtn.innerHTML = `
        <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg>
        Run End-to-End Pipeline
      `;
    }
  }

  function switchTab(tabId) {
    tabBtns.forEach((b) => b.classList.toggle("active", b.dataset.tab === tabId));
    tabPanes.forEach((p) => p.classList.toggle("active", p.id === tabId));
  }

  function renderPipelineResults(data) {
    const s1 = data.stage1;
    const s2 = data.stage2;
    const totalLatency = data.total_pipeline_latency_ms;

    // Update KPI Banner
    kpiLatency.textContent = `${totalLatency} ms`;
    kpiS1Count.textContent = s1.candidates ? s1.candidates.length : 0;
    kpiS2Count.textContent = s2.candidates ? s2.candidates.length : 0;
    const prunedCount = s2.pruned_decisions
      ? s2.pruned_decisions.filter((d) => d.status !== "retained").length
      : 0;
    kpiPrunedCount.textContent = prunedCount;

    headerLatencyVal.textContent = `${totalLatency} ms / 200 ms`;
    const complianceBadge = document.getElementById("latencyComplianceBadge");
    if (data.budget_compliance && data.budget_compliance.is_compliant) {
      complianceBadge.textContent = `Compliant (${data.budget_compliance.headroom_pct}% Headroom)`;
      complianceBadge.className = "badge emerald";
    } else {
      complianceBadge.textContent = "Over Budget";
      complianceBadge.className = "badge rose";
    }

    // Render individual tabs
    renderOverviewTab(s2);
    renderStage1Tab(s1);
    renderStage2Tab(s2);
    renderLatencyWaterfall(data);
    renderDagGraph(data);
  }

  // ==========================================
  // TAB 1: OVERVIEW & FINAL SUGGESTIONS
  // ==========================================
  function renderOverviewTab(s2) {
    const list = document.getElementById("finalSuggestionsList");
    const countBadge = document.getElementById("retainedBadge");
    const candidates = s2.candidates || [];

    countBadge.textContent = `${candidates.length} grounded categories`;

    // Parsed Signals Header
    const meta = s2.parsed_metadata || {};
    const geo = meta.location || {};
    document.getElementById("signalGeo").textContent = geo.city
      ? `${geo.city}, ${geo.country} (${geo.country_code || ""})`
      : "Not Geocoded";
    document.getElementById("signalYear").textContent = meta.capture_year || "Unknown";
    document.getElementById("signalEvents").textContent =
      s2.event_candidates && s2.event_candidates.length > 0
        ? s2.event_candidates[0].name.replace("Category:", "")
        : "None discovered";

    if (candidates.length === 0) {
      list.innerHTML = `
        <div class="empty-state">
          <p>All candidates were pruned by temporal or spatial filters. Try relaxing filter thresholds.</p>
        </div>
      `;
      return;
    }

    list.innerHTML = "";
    candidates.forEach((cand, idx) => {
      const card = document.createElement("div");
      card.className = "suggestion-card";

      const rank = idx + 1;
      const rankClass = rank === 1 ? "gold" : "";
      const catClean = cand.name.replace("Category:", "");
      const commonsUrl = `https://commons.wikimedia.org/wiki/Category:${encodeURIComponent(catClean.replace(/ /g, "_"))}`;
      const scorePct = Math.min(100, Math.round(cand.score * 100));

      card.innerHTML = `
        <div class="suggestion-left">
          <div class="rank-badge ${rankClass}">#${rank}</div>
          <div>
            <a href="${commonsUrl}" target="_blank" rel="noopener noreferrer" class="category-name-link">
              ${cand.name}
              <svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>
            </a>
            <span class="provenance-tag ${cand.source}">${cand.source.replace(/_/g, " ")}</span>
          </div>
        </div>
        <div class="suggestion-right">
          <div class="score-display">
            <span class="score-val">${cand.score.toFixed(4)}</span>
            <div class="score-bar-mini">
              <div class="score-bar-fill" style="width: ${scorePct}%;"></div>
            </div>
          </div>
          <button class="copy-tag-btn" data-tag="[[${cand.name}]]" title="Copy Wikitext Tag">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
          </button>
        </div>
      `;

      card.querySelector(".copy-tag-btn").addEventListener("click", (e) => {
        const tag = e.currentTarget.dataset.tag;
        navigator.clipboard.writeText(tag).then(() => {
          showToast(`Copied ${tag}`);
        });
      });

      list.appendChild(card);
    });
  }

  // ==========================================
  // TAB 2: STAGE 1 VISUAL RECALL
  // ==========================================
  function renderStage1Tab(s1) {
    const list = document.getElementById("s1CandidatesList");
    const badge = document.getElementById("s1PoolBadge");
    const latVal = document.getElementById("s1LatencyVal");
    const candidates = s1.candidates || [];

    badge.textContent = `${candidates.length} visual candidates`;
    latVal.textContent = `${s1.latency_ms.stage1_total_ms || 24.5} ms`;

    // Micro latency steps
    if (s1.latency_ms.vision_encoding) {
      document.getElementById("s1LatVision").textContent = `${s1.latency_ms.vision_encoding.toFixed(2)} ms`;
    }
    if (s1.latency_ms.knn_scorer) {
      document.getElementById("s1LatKnn").textContent = `${s1.latency_ms.knn_scorer.toFixed(2)} ms`;
    }
    if (s1.latency_ms.prototype_scorer) {
      document.getElementById("s1LatProto").textContent = `${s1.latency_ms.prototype_scorer.toFixed(2)} ms`;
    }
    if (s1.latency_ms.aggregation) {
      document.getElementById("s1LatAgg").textContent = `${s1.latency_ms.aggregation.toFixed(2)} ms`;
    }

    list.innerHTML = "";
    candidates.forEach((cand, idx) => {
      const card = document.createElement("div");
      card.className = "candidate-card-s1";

      const exemplars = (cand.metadata && cand.metadata.neighbor_exemplars) || [];
      const exemplarsHtml = exemplars.length
        ? `<div class="s1-exemplars-row">
            <span>Historical Exemplars:</span>
            ${exemplars.map((ex) => `<span class="exemplar-chip">${ex}</span>`).join("")}
           </div>`
        : "";

      card.innerHTML = `
        <div class="s1-header">
          <div style="display:flex; align-items:center; gap:8px;">
            <span class="rank-badge">#${idx + 1}</span>
            <span style="font-weight:600;">${cand.name}</span>
            <span class="provenance-tag ${cand.source}">${cand.source}</span>
          </div>
          <div class="score-display">
            <span class="score-val">${cand.score.toFixed(4)}</span>
          </div>
        </div>
        ${exemplarsHtml}
      `;
      list.appendChild(card);
    });
  }

  // ==========================================
  // TAB 3: STAGE 2 AUDIT TRAIL
  // ==========================================
  function renderStage2Tab(s2) {
    const decisions = s2.pruned_decisions || [];
    document.getElementById("auditDecisionsBadge").textContent = `${decisions.length} audit records`;
    renderAuditTrail(decisions);
  }

  function renderAuditTrail(decisions) {
    const list = document.getElementById("auditDecisionsList");
    list.innerHTML = "";

    const filtered = decisions.filter((d) => {
      if (activeAuditFilter === "all") return true;
      return d.status === activeAuditFilter;
    });

    if (filtered.length === 0) {
      list.innerHTML = `
        <div class="empty-state">
          <p>No audit decisions match filter: <strong>${activeAuditFilter}</strong></p>
        </div>
      `;
      return;
    }

    filtered.forEach((dec) => {
      const card = document.createElement("div");
      card.className = `audit-card status-${dec.status}`;

      const statusLabels = {
        retained: "Retained Grounded",
        suppressed_ancestor: "Suppressed Ancestor",
        spatial_conflict: "Spatial Conflict",
        temporal_conflict: "Temporal Conflict",
      };

      card.innerHTML = `
        <div class="audit-header">
          <div style="display:flex; align-items:center; gap:8px;">
            <span class="audit-status status-${dec.status}">
              ${statusLabels[dec.status] || dec.status}
            </span>
            <span style="font-weight:600;">${dec.name}</span>
          </div>
          <span style="font-family:var(--font-mono); font-size:0.8rem; color:var(--text-muted);">
            Score: ${dec.score.toFixed(4)}
          </span>
        </div>
        <div class="audit-reason">
          ${dec.reason || "Category passed spatial, temporal, and DAG specificity constraints."}
        </div>
      `;

      list.appendChild(card);
    });
  }

  // ==========================================
  // TAB 4: CATEGORY DAG HIERARCHY TREE (SVG)
  // ==========================================
  async function renderDagGraph(pipelineData) {
    const svg = document.getElementById("dagSvg");
    const s2 = pipelineData.stage2 || {};
    const retainedNames = new Set((s2.candidates || []).map((c) => c.name));
    const decisions = s2.pruned_decisions || [];
    const statusMap = {};
    decisions.forEach((d) => {
      statusMap[d.name] = d;
    });

    try {
      const resp = await fetch("/api/dag");
      const dag = await resp.json();
      const nodes = dag.nodes || [];
      const edges = dag.edges || [];

      // Layout nodes into hierarchical levels
      // Level 0: Roots (Architecture, Monuments, Bridges by country)
      // Level 1: Sub-categories
      // Level 2: Specific categories
      // Level 3: Leaves (at night, sunset, western facade, torch relay)
      const levelBuckets = { 0: [], 1: [], 2: [], 3: [], 4: [] };
      nodes.forEach((n) => {
        const lvl = Math.min(4, n.depth || 0);
        levelBuckets[lvl].push(n);
      });

      const width = 900;
      const height = 480;
      const positions = {};

      Object.keys(levelBuckets).forEach((lvlStr) => {
        const lvl = parseInt(lvlStr);
        const bucket = levelBuckets[lvl];
        const count = bucket.length;
        const xStep = width / (count + 1);
        const y = 50 + lvl * 95;

        bucket.forEach((n, idx) => {
          positions[n.id] = {
            x: xStep * (idx + 1),
            y: y,
            node: n,
          };
        });
      });

      // Render SVG
      svg.innerHTML = `
        <defs>
          <marker id="dagArrow" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 8 5 L 0 9 z" fill="#3b82f6" />
          </marker>
        </defs>
      `;

      // Draw Edges
      edges.forEach((e) => {
        const p1 = positions[e.source];
        const p2 = positions[e.target];
        if (p1 && p2) {
          const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
          const d = `M ${p1.x} ${p1.y} C ${p1.x} ${(p1.y + p2.y) / 2}, ${p2.x} ${(p1.y + p2.y) / 2}, ${p2.x} ${p2.y}`;
          path.setAttribute("d", d);
          path.setAttribute("fill", "none");
          path.setAttribute("stroke", "#242f44");
          path.setAttribute("stroke-width", "1.5");
          path.setAttribute("marker-end", "url(#dagArrow)");
          svg.appendChild(path);
        }
      });

      // Draw Nodes
      nodes.forEach((n) => {
        const pos = positions[n.id];
        if (!pos) return;

        const isRetained = retainedNames.has(n.name);
        const decision = statusMap[n.name];
        const status = decision ? decision.status : isRetained ? "retained" : "context";

        let strokeColor = "#334155";
        let fillColor = "#0f172a";
        let textColor = "#94a3b8";

        if (status === "retained") {
          strokeColor = "#10b981";
          fillColor = "rgba(16, 185, 129, 0.15)";
          textColor = "#34d399";
        } else if (status === "suppressed_ancestor") {
          strokeColor = "#a855f7";
          fillColor = "rgba(168, 85, 247, 0.15)";
          textColor = "#c084fc";
        } else if (status === "spatial_conflict" || status === "temporal_conflict") {
          strokeColor = "#f43f5e";
          fillColor = "rgba(244, 63, 94, 0.15)";
          textColor = "#fb7185";
        }

        const g = document.createElementNS("http://www.w3.org/2000/svg", "g");
        g.style.cursor = "pointer";

        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        const nodeW = 140;
        const nodeH = 34;
        rect.setAttribute("x", pos.x - nodeW / 2);
        rect.setAttribute("y", pos.y - nodeH / 2);
        rect.setAttribute("width", nodeW);
        rect.setAttribute("height", nodeH);
        rect.setAttribute("rx", "6");
        rect.setAttribute("fill", fillColor);
        rect.setAttribute("stroke", strokeColor);
        rect.setAttribute("stroke-width", status === "retained" ? "2" : "1.5");
        if (status === "suppressed_ancestor") {
          rect.setAttribute("stroke-dasharray", "4,2");
        }

        const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
        text.setAttribute("x", pos.x);
        text.setAttribute("y", pos.y + 4);
        text.setAttribute("text-anchor", "middle");
        text.setAttribute("fill", textColor);
        text.setAttribute("font-size", "10");
        text.setAttribute("font-family", "Inter, sans-serif");
        text.setAttribute("font-weight", status === "retained" ? "600" : "400");

        // Truncate long category label
        let displayLabel = n.label;
        if (displayLabel.length > 20) {
          displayLabel = displayLabel.slice(0, 18) + "…";
        }
        text.textContent = displayLabel;

        g.appendChild(rect);
        g.appendChild(text);

        // Click to inspect
        g.addEventListener("click", () => {
          inspectDagNode(n, status, decision);
        });

        svg.appendChild(g);
      });
    } catch (e) {
      console.error("DAG render error:", e);
    }
  }

  function inspectDagNode(node, status, decision) {
    const titleEl = document.getElementById("inspectorTitle");
    const detEl = document.getElementById("inspectorDetails");

    titleEl.textContent = `${node.name} (Taxonomic Depth: ${node.depth})`;
    let reasonText = decision ? decision.reason : "Taxonomic context category.";
    let statusBadge = `<span class="badge ${
      status === "retained"
        ? "emerald"
        : status === "suppressed_ancestor"
        ? "purple"
        : status.includes("conflict")
        ? "rose"
        : "blue"
    }">${status.toUpperCase()}</span>`;

    detEl.innerHTML = `
      <div style="margin-top:6px; display:flex; gap:10px; align-items:center;">
        ${statusBadge}
        <span>Score: ${decision ? decision.score.toFixed(4) : "N/A"}</span>
      </div>
      <div style="margin-top:8px; font-size:0.8rem; line-height:1.4;">
        ${reasonText}
      </div>
    `;
  }

  // ==========================================
  // TAB 5: LATENCY WATERFALL PROFILE
  // ==========================================
  function renderLatencyWaterfall(data) {
    const totalMs = data.total_pipeline_latency_ms;
    document.getElementById("waterfallTotalMs").textContent = `${totalMs} ms`;

    const budgetFill = document.getElementById("waterfallBudgetFill");
    const pct = Math.min(100, (totalMs / 200.0) * 100);
    budgetFill.style.width = `${pct}%`;

    const headroom = Math.max(0, 200.0 - totalMs).toFixed(1);
    const headroomPct = Math.max(0, (200.0 - totalMs) / 200.0 * 100).toFixed(1);
    document.getElementById("headroomText").textContent = `${headroom} ms Headroom (${headroomPct}%)`;

    const s2 = data.stage2 || {};
    const latMap = s2.latency_ms || {};
    const stepsList = document.getElementById("waterfallStepsList");
    stepsList.innerHTML = "";

    const orderedSteps = [
      { key: "vision_encoding", name: "Vision Backbone (SigLIP)", stage: "Stage 1", color: "#3b82f6" },
      { key: "knn_scorer", name: "Historical k-NN Consensus", stage: "Stage 1", color: "#3b82f6" },
      { key: "prototype_scorer", name: "Zero-Shot Text Prototypes", stage: "Stage 1", color: "#3b82f6" },
      { key: "aggregation", name: "Reciprocal Rank Fusion (RRF)", stage: "Stage 1", color: "#3b82f6" },
      { key: "stage2_metadata_preprocessing", name: "Wikitext Cleaning & Geocoding", stage: "Stage 2", color: "#a855f7" },
      { key: "stage2_event_discovery", name: "Proactive BM25 Event Retrieval", stage: "Stage 2", color: "#a855f7" },
      { key: "stage2_candidate_merging", name: "Pool Ingestion (Visual ∪ Event)", stage: "Stage 2", color: "#a855f7" },
      { key: "stage2_temporal_filtering", name: "Temporal Year Conflict Filter", stage: "Stage 2", color: "#a855f7" },
      { key: "stage2_spatial_filtering", name: "Spatial Haversine Geotag Filter", stage: "Stage 2", color: "#a855f7" },
      { key: "stage2_dag_ancestor_pruning", name: "Category DAG Ancestor Pruning", stage: "Stage 2", color: "#a855f7" },
    ];

    orderedSteps.forEach((step) => {
      const ms = latMap[step.key] || 0.05;
      const row = document.createElement("div");
      row.className = "waterfall-row";

      row.innerHTML = `
        <div class="waterfall-step-info">
          <span class="waterfall-pill" style="background:${step.color}22; color:${step.color};">
            ${step.stage}
          </span>
          <span style="font-size:0.85rem; font-weight:500;">${step.name}</span>
        </div>
        <div class="waterfall-time" style="color:${step.color}">
          ${ms.toFixed(2)} ms
        </div>
      `;
      stepsList.appendChild(row);
    });
  }

  // ==========================================
  // CLIPBOARD & EXPORT UTILITIES
  // ==========================================
  function copyAllWikitextTags() {
    if (!lastPipelineResult || !lastPipelineResult.stage2) {
      showToast("No categories to copy. Run pipeline first.");
      return;
    }
    const tags = (lastPipelineResult.stage2.candidates || [])
      .map((c) => `[[${c.name}]]`)
      .join("\n");
    navigator.clipboard.writeText(tags).then(() => {
      showToast("Copied all [[Category:...]] tags!");
    });
  }

  function exportResultsJson() {
    if (!lastPipelineResult) {
      showToast("No execution result to export.");
      return;
    }
    const blob = new Blob([JSON.stringify(lastPipelineResult, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `commonscat_export_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showToast("Downloaded execution JSON dump!");
  }

  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.remove("hidden");
    setTimeout(() => {
      toast.classList.add("hidden");
    }, 2800);
  }
});
