const state = {
  data: null,
  selected: null,
  sortedRequests: [],
  referenceTokens: [],
  receivedTokens: [],
  referenceWordEls: [],
  receivedWordEls: [],
  referenceActiveCount: 0,
  receivedActiveCount: 0,
  replayTimeMs: 0,
  replayMaxMs: 0,
  replayPlaying: false,
  replayStartedAt: 0,
  replayBaseMs: 0,
  audioSeekToken: 0,
  raf: null,
  latencyChart: null,
  chartSeries: null,
  chartLibraryFailed: false,
};

const ECHARTS_URL = "https://cdn.jsdelivr.net/npm/echarts@5.6.0/dist/echarts.min.js";

const els = {
  dataUrl: document.getElementById("data-url"),
  loadUrl: document.getElementById("load-url"),
  fileInput: document.getElementById("file-input"),
  runSubtitle: document.getElementById("run-subtitle"),
  requestTitle: document.getElementById("request-title"),
  requestMeta: document.getElementById("request-meta"),
  latencyValue: document.getElementById("latency-value"),
  latencyDetail: document.getElementById("latency-detail"),
  latencyPanel: document.getElementById("latency-panel"),
  metricStrip: document.getElementById("metric-strip"),
  audio: document.getElementById("audio-player"),
  playToggle: document.getElementById("play-toggle"),
  replayButton: document.getElementById("replay-button"),
  timeSlider: document.getElementById("time-slider"),
  timeLabel: document.getElementById("time-label"),
  durationLabel: document.getElementById("duration-label"),
  latencyChart: document.getElementById("latency-chart"),
  latencyChartEmpty: document.getElementById("latency-chart-empty"),
  chartLiveMean: document.getElementById("chart-live-mean"),
  chartLatest: document.getElementById("chart-latest"),
  chartCount: document.getElementById("chart-count"),
  chartPlayhead: document.getElementById("chart-playhead"),
  chartPlayheadLabel: document.getElementById("chart-playhead-label"),
  referenceStream: document.getElementById("reference-stream"),
  transcriptStream: document.getElementById("transcript-stream"),
  replayWarning: document.getElementById("replay-warning"),
  sortSelect: document.getElementById("sort-select"),
  requestTable: document.getElementById("request-table"),
};

let chartLibraryLoad = null;

els.loadUrl.addEventListener("click", () => loadFromUrl(els.dataUrl.value.trim()));
els.fileInput.addEventListener("change", loadFromFile);
els.playToggle.addEventListener("click", toggleReplay);
els.replayButton.addEventListener("click", replayFromStart);
els.sortSelect.addEventListener("change", () => {
  sortRequests();
  renderRequestTable();
});
els.timeSlider.addEventListener("input", () => {
  pauseReplay();
  setReplayTime(Number(els.timeSlider.value));
  seekAudioToReplayTime();
});
window.addEventListener("resize", () => {
  state.latencyChart?.resize();
  renderLatencyChart();
  updateLatencyChartProgress();
});

ensureLatencyChartLibrary().catch(() => {});

const queryData = new URLSearchParams(window.location.search).get("data");
if (queryData) {
  els.dataUrl.value = queryData;
  loadFromUrl(queryData);
}

async function loadFromUrl(path) {
  if (!path) {
    return;
  }
  const url = new URL(path, `${window.location.origin}/`);
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.status}`);
  }
  loadData(await response.json());
}

function loadFromFile(event) {
  const file = event.target.files?.[0];
  if (!file) {
    return;
  }
  const reader = new FileReader();
  reader.onload = () => loadData(JSON.parse(String(reader.result || "{}")));
  reader.readAsText(file);
}

function loadData(data) {
  state.data = data;
  els.runSubtitle.textContent = `${data.run_dir || "run"} · ${data.requests?.length || 0} ASR requests`;
  sortRequests();
  selectRequest(state.sortedRequests[0] || null);
}

function sortRequests() {
  const requests = [...(state.data?.requests || [])];
  const sortKey = els.sortSelect.value;
  const metric = (request, name) => Number(request.metrics?.[name] ?? -Infinity);
  requests.sort((a, b) => {
    if (sortKey === "request_id_asc") {
      return Number(a.request_id ?? 0) - Number(b.request_id ?? 0);
    }
    if (sortKey === "final_wer_desc") {
      return metric(b, "final_wer") - metric(a, "final_wer");
    }
    if (sortKey === "rtf_desc") {
      return metric(b, "rtf") - metric(a, "rtf");
    }
    return metric(b, "interactivity") - metric(a, "interactivity");
  });
  state.sortedRequests = requests;
}

function renderRequestTable() {
  els.requestTable.innerHTML = state.sortedRequests
    .map((request) => {
      const selected = state.selected?.request_id === request.request_id ? " selected" : "";
      return `
        <button class="request-row${selected}" type="button" data-request-id="${escapeHtml(request.request_id)}">
          <span class="request-number">#${escapeHtml(request.request_id)}</span>
          <span class="request-text">${escapeHtml(request.dataset || "")} · ${escapeHtml(request.sample_id || "")}</span>
          <span class="request-latency">${formatMetric(request.metrics?.interactivity, "ms")}</span>
        </button>
      `;
    })
    .join("");

  for (const row of els.requestTable.querySelectorAll(".request-row")) {
    row.addEventListener("click", () => {
      const requestId = row.getAttribute("data-request-id");
      selectRequest(state.sortedRequests.find((item) => String(item.request_id) === requestId));
    });
  }
}

function selectRequest(request) {
  pauseReplay();
  state.selected = request || null;
  state.referenceTokens = buildReferenceTokens(request || {});
  state.receivedTokens = buildReceivedTokens(request || {});
  state.referenceWordEls = [];
  state.receivedWordEls = [];
  state.referenceActiveCount = 0;
  state.receivedActiveCount = 0;
  state.chartSeries = null;
  renderRequestTable();
  if (!request) {
    return;
  }

  els.requestTitle.textContent = "Live call";
  els.requestMeta.textContent = `Request #${request.request_id} · ${request.dataset || "unknown"} · ${request.sample_id || request.source_id || ""}`;
  els.audio.src = request.audio_url ? new URL(request.audio_url, `${window.location.origin}/`).href : "";

  configureTimeline(request);
  renderMetricStrip(request);
  renderLatencyPanel(request);
  renderStreams();
  renderLatencyChart(true);
  setReplayTime(0);
  seekAudioToReplayTime();
}

function configureTimeline(request) {
  const durationMs = Number(request.duration_ms || 0);
  const lastReferenceMs = Math.max(0, ...state.referenceTokens.map((word) => word.end_ms));
  const lastReceivedMs = Math.max(0, ...state.receivedTokens.map((word) => word.time_ms));
  state.replayMaxMs = Math.max(durationMs, lastReferenceMs, lastReceivedMs) + 300;
  els.timeSlider.max = String(Math.max(100, Math.ceil(state.replayMaxMs)));
  els.durationLabel.textContent = formatClock(state.replayMaxMs);
}

function renderMetricStrip(request) {
  const metrics = request.metrics || {};
  const items = [
    ["TTFC", metrics.ttfc, "ms"],
    ["First partial", metrics.time_to_first_partial, "ms"],
    ["Final after EOF", metrics.time_to_final_transcript, "ms"],
    ["Final WER", metrics.final_wer, "%"],
    ["RTF", metrics.rtf],
  ];
  els.metricStrip.innerHTML = items
    .map(
      ([label, value, unit]) => `
        <div class="metric">
          <span>${escapeHtml(label)}</span>
          <strong>${formatMetric(value, unit)}</strong>
        </div>
      `,
    )
    .join("");
}

function renderLatencyPanel(request) {
  const metrics = request.metrics || {};
  const interactivity = metrics.interactivity;
  const wordCount = metrics.interactivity_word_count ?? state.receivedTokens.length;
  els.latencyValue.textContent = formatMetric(interactivity, "ms");
  els.latencyDetail.textContent = `${wordCount} matched words · ${formatMetric(metrics.ttfc, "ms")} TTFC`;
  els.latencyPanel.className = `latency-panel ${latencyClass(interactivity)}`;
}

function setReplayTime(ms) {
  state.replayTimeMs = Math.max(0, Math.min(ms, state.replayMaxMs));
  els.timeSlider.value = String(Math.round(state.replayTimeMs));
  els.timeLabel.textContent = formatClock(state.replayTimeMs);
  updateStreamProgress();
  updateLatencyChartProgress();
}

function toggleReplay() {
  if (state.replayPlaying) {
    pauseReplay();
  } else {
    startReplay();
  }
}

function replayFromStart() {
  pauseReplay();
  setReplayTime(0);
  startReplay();
}

async function startReplay() {
  if (!state.selected) {
    return;
  }
  if (state.replayTimeMs >= state.replayMaxMs) {
    setReplayTime(0);
  }
  await seekAudioToReplayTime();
  state.replayPlaying = true;
  state.replayBaseMs = state.replayTimeMs;
  state.replayStartedAt = performance.now();
  updatePlaybackControls();
  if (els.audio.src) {
    els.audio.play().catch(() => {});
  }
  tickReplay();
}

function pauseReplay() {
  state.replayPlaying = false;
  if (state.raf !== null) {
    cancelAnimationFrame(state.raf);
    state.raf = null;
  }
  els.audio.pause();
  updatePlaybackControls();
}

function tickReplay() {
  if (!state.replayPlaying) {
    return;
  }
  const nextMs = state.replayBaseMs + performance.now() - state.replayStartedAt;
  setReplayTime(nextMs);
  if (nextMs >= state.replayMaxMs) {
    pauseReplay();
    return;
  }
  state.raf = requestAnimationFrame(tickReplay);
}

async function seekAudioToReplayTime() {
  if (!els.audio.src) {
    return;
  }
  const token = ++state.audioSeekToken;
  const seconds = Math.max(0, state.replayTimeMs / 1000);
  if (els.audio.readyState < 1) {
    els.audio.load();
    await new Promise((resolve) => {
      els.audio.addEventListener("loadedmetadata", resolve, { once: true });
    });
    if (token !== state.audioSeekToken) {
      return;
    }
  }
  await seekAudioToSeconds(seconds, token);
}

async function seekAudioToSeconds(seconds, token) {
  if (!els.audio.src || token !== state.audioSeekToken) {
    return;
  }
  const target = Number.isFinite(els.audio.duration)
    ? Math.min(seconds, els.audio.duration)
    : seconds;
  if (Math.abs(els.audio.currentTime - target) < 0.05) {
    return;
  }
  const seeked = new Promise((resolve) => {
    const done = () => {
      clearTimeout(timeout);
      els.audio.removeEventListener("seeked", done);
      resolve();
    };
    const timeout = setTimeout(done, 400);
    els.audio.addEventListener("seeked", done);
  });
  if (Number.isFinite(els.audio.duration)) {
    els.audio.currentTime = target;
  } else {
    els.audio.currentTime = target;
  }
  await seeked;
}

function updatePlaybackControls() {
  els.playToggle.classList.toggle("is-playing", state.replayPlaying);
  els.playToggle.setAttribute("aria-label", state.replayPlaying ? "Pause" : "Play");
  els.playToggle.title = state.replayPlaying ? "Pause" : "Play";
}

function renderStreams() {
  els.referenceStream.innerHTML = renderTimedWords(state.referenceTokens, "reference");
  els.transcriptStream.innerHTML = renderTimedWords(state.receivedTokens, "received");
  state.referenceWordEls = [...els.referenceStream.querySelectorAll(".stream-word")];
  state.receivedWordEls = [...els.transcriptStream.querySelectorAll(".stream-word")];
  state.referenceActiveCount = 0;
  state.receivedActiveCount = 0;
  updateStreamProgress();
  els.replayWarning.textContent = state.selected?.has_replay
    ? ""
    : "This row has scalar metrics but no precomputed replay words.";
}

function renderTimedWords(tokens, kind) {
  return tokens
    .map((token) => {
      const active = Number(token.time_ms || 0) <= state.replayTimeMs;
      const detail = kind === "reference"
        ? formatClock(Number(token.time_ms || 0))
        : `${formatClock(Number(token.time_ms || 0))}${latencySuffix(token.latency_ms)}`;
      return `
        <span class="stream-word ${active ? "active" : ""}" title="${escapeHtml(detail)}">
          ${escapeHtml(token.word)}
        </span>
      `;
    })
    .join(" ");
}

function updateStreamProgress() {
  state.referenceActiveCount = updateActiveWords(
    state.referenceWordEls,
    state.referenceActiveCount,
    upperBoundByTime(state.referenceTokens, state.replayTimeMs),
  );
  state.receivedActiveCount = updateActiveWords(
    state.receivedWordEls,
    state.receivedActiveCount,
    upperBoundByTime(state.receivedTokens, state.replayTimeMs),
  );
}

function updateActiveWords(elements, previousCount, nextCount) {
  if (nextCount > previousCount) {
    for (let index = previousCount; index < nextCount; index += 1) {
      elements[index]?.classList.add("active");
    }
  } else if (nextCount < previousCount) {
    for (let index = nextCount; index < previousCount; index += 1) {
      elements[index]?.classList.remove("active");
    }
  }
  return nextCount;
}

function buildReferenceTokens(request) {
  return (request.reference_words || []).map((word) => ({
    word: String(word.word || ""),
    time_ms: Number(word.start_ms ?? word.end_ms ?? 0),
    end_ms: Number(word.end_ms ?? word.start_ms ?? 0),
  }));
}

function buildReceivedTokens(request) {
  return (request.received_words || [])
    .map((word, index) => ({
      index,
      word: String(word.word || ""),
      time_ms: Number(word.time_ms ?? 0),
      reference_end_ms: Number(word.reference_end_ms ?? 0),
      latency_ms: Number(word.latency_ms ?? 0),
    }))
    .filter((word) => Number.isFinite(word.time_ms) && Number.isFinite(word.latency_ms))
    .sort((a, b) => a.time_ms - b.time_ms);
}

function initLatencyChart() {
  if (state.latencyChart) {
    return;
  }
  if (!window.echarts || !els.latencyChart) {
    return;
  }
  state.chartLibraryFailed = false;
  state.latencyChart = window.echarts.init(els.latencyChart, null, {
    renderer: "canvas",
  });
  state.latencyChart.on("click", (params) => {
    const value = Array.isArray(params.value) ? params.value : null;
    if (!value) {
      return;
    }
    pauseReplay();
    setReplayTime(Number(value[0]));
    seekAudioToReplayTime();
  });
}

function ensureLatencyChartLibrary() {
  if (window.echarts) {
    initLatencyChart();
    return Promise.resolve(window.echarts);
  }
  if (chartLibraryLoad) {
    return chartLibraryLoad;
  }

  chartLibraryLoad = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = ECHARTS_URL;
    script.async = true;
    script.crossOrigin = "anonymous";
    script.onload = () => {
      state.chartLibraryFailed = false;
      initLatencyChart();
      renderLatencyChart();
      resolve(window.echarts);
    };
    script.onerror = () => {
      state.chartLibraryFailed = true;
      els.latencyChartEmpty.textContent = "Could not load chart library from CDN";
      els.latencyChartEmpty.classList.remove("hidden");
      reject(new Error(`Failed to load ${ECHARTS_URL}`));
    };
    document.head.appendChild(script);
  });

  return chartLibraryLoad;
}

function renderLatencyChart() {
  const series = getLatencySeries();
  els.latencyChartEmpty.textContent = state.chartLibraryFailed
    ? "Could not load chart library from CDN"
    : "Latency points appear as transcript words arrive";
  els.latencyChartEmpty.classList.toggle(
    "hidden",
    !state.chartLibraryFailed && series.points.length > 0,
  );

  if (!state.latencyChart) {
    ensureLatencyChartLibrary().catch(() => {});
  }
  if (!state.latencyChart) {
    return;
  }

  state.latencyChart.setOption(
    {
      backgroundColor: "#faf9f6",
      animation: false,
      animationEasingUpdate: "cubicOut",
      grid: {
        left: 68,
        right: 28,
        top: 74,
        bottom: 40,
      },
      tooltip: {
        trigger: "item",
        confine: true,
        borderWidth: 1,
        borderColor: "rgba(23, 23, 23, 0.14)",
        backgroundColor: "#ffffff",
        padding: 0,
        extraCssText: "box-shadow:0 12px 34px rgba(23,23,23,0.16);border-radius:8px;",
        formatter: formatChartTooltip,
      },
      xAxis: {
        type: "value",
        min: 0,
        max: series.xMax,
        axisLabel: {
          color: "#74716c",
          formatter: (value) => formatClock(Number(value)),
        },
        axisLine: { lineStyle: { color: "rgba(23, 23, 23, 0.44)" } },
        axisTick: { show: false },
        splitLine: { lineStyle: { color: "rgba(23, 23, 23, 0.08)" } },
      },
      yAxis: {
        type: "value",
        min: series.yMin,
        max: series.yMax,
        axisLabel: {
          color: "#74716c",
          formatter: (value) => `${Math.round(Number(value))} ms`,
        },
        axisLine: { lineStyle: { color: "rgba(23, 23, 23, 0.44)" } },
        axisTick: { show: false },
        splitLine: {
          lineStyle: {
            color: "rgba(23, 23, 23, 0.08)",
            type: "dashed",
          },
        },
      },
      series: [
        {
          name: "Word latency",
          type: "scatter",
          data: series.pointData,
          symbolSize: (value) => Math.max(9, Math.min(18, 9 + Math.abs(Number(value[1])) / 180)),
          itemStyle: {
            color: "#171717",
            borderColor: "#faf9f6",
            borderWidth: 2,
            shadowBlur: 8,
            shadowColor: "rgba(23, 23, 23, 0.16)",
          },
          emphasis: {
            scale: 1.45,
            itemStyle: {
              borderColor: "#171717",
              borderWidth: 2,
              shadowBlur: 16,
              shadowColor: "rgba(23, 23, 23, 0.28)",
            },
          },
        },
        {
          name: "Running mean",
          type: "line",
          data: series.meanData,
          showSymbol: false,
          smooth: 0.3,
          lineStyle: {
            color: "#195e53",
            width: 3,
            shadowBlur: 8,
            shadowColor: "rgba(25, 94, 83, 0.22)",
          },
        },
      ],
    },
    {
      notMerge: true,
      lazyUpdate: true,
    },
  );
  updateLatencyChartProgress();
}

function getLatencySeries() {
  if (state.chartSeries) {
    return state.chartSeries;
  }
  const points = runningMeanPoints(state.receivedTokens.map((token, index) => ({
    index,
    word: token.word,
    time_ms: token.time_ms,
    latency_ms: token.latency_ms,
    reference_end_ms: token.reference_end_ms,
  })));
  const latencies = points.map((point) => point.latency_ms);
  const maxLatency = Math.max(500, ...latencies);
  state.chartSeries = {
    points,
    pointData: points.map((point) => [
      point.time_ms,
      point.latency_ms,
      point.word,
      point.reference_end_ms,
      point.index,
      point.mean_ms,
    ]),
    meanData: points.map((point) => [
      point.time_ms,
      point.mean_ms,
      point.word,
      point.latency_ms,
      point.index,
    ]),
    xMax: Math.max(state.replayMaxMs, ...points.map((point) => point.time_ms), 1000),
    yMin: 0,
    yMax: Math.ceil((maxLatency + 150) / 100) * 100,
  };
  return state.chartSeries;
}

function updateLatencyChartProgress() {
  const series = getLatencySeries();
  const activeCount = upperBoundByTime(series.points, state.replayTimeMs);
  const activePoints = series.points.slice(0, activeCount);
  updateChartStats(activePoints);
  updateChartPlayhead(series);
}

function runningMeanPoints(points) {
  let total = 0;
  return points.map((point, index) => {
    total += point.latency_ms;
    return {
      ...point,
      mean_ms: total / (index + 1),
    };
  });
}

function upperBoundByTime(points, timeMs) {
  let lo = 0;
  let hi = points.length;
  while (lo < hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (points[mid].time_ms <= timeMs) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

function updateChartPlayhead(series) {
  if (!els.chartPlayhead) {
    return;
  }
  const chartRect = els.latencyChart.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, state.replayTimeMs / Math.max(series.xMax, 1)));
  const x = 68 + ratio * Math.max(0, chartRect.width - 96);
  els.chartPlayhead.style.transform = `translateX(${Math.round(x)}px)`;
  els.chartPlayheadLabel.textContent = formatClock(state.replayTimeMs);
}

function formatChartTooltip(params) {
  const value = Array.isArray(params.value) ? params.value : [];
  const seriesName = params.seriesName || "";
  const receivedMs = Number(value[0]);
  const yValue = Number(value[1]);
  const word = value[2] || "";
  const latencyMs = seriesName === "Word latency" ? yValue : Number(value[3]);
  const meanMs = seriesName === "Word latency" ? Number(value[5]) : yValue;
  const referenceEndMs = seriesName === "Word latency" ? Number(value[3]) : receivedMs - latencyMs;
  const rows = [
    ["spoken end", formatClock(referenceEndMs)],
    ["received", formatClock(receivedMs)],
    ["latency", formatMetric(latencyMs, "ms")],
    ["running mean", formatMetric(meanMs, "ms")],
  ];

  return `
    <div class="chart-tooltip">
      <strong>${escapeHtml(word || seriesName)}</strong>
      ${rows
        .map(([label, detail]) => `
          <div>
            <span>${escapeHtml(label)}</span>
            <b>${escapeHtml(detail)}</b>
          </div>
        `)
        .join("")}
    </div>
  `;
}

function updateChartStats(activePoints) {
  if (!activePoints.length) {
    els.chartLiveMean.textContent = "mean n/a";
    els.chartLatest.textContent = "latest n/a";
    els.chartCount.textContent = "0 words";
    return;
  }
  const latest = activePoints.at(-1);
  const mean = activePoints.reduce((sum, point) => sum + point.latency_ms, 0) / activePoints.length;
  els.chartLiveMean.textContent = `mean ${Math.round(mean)} ms`;
  els.chartLatest.textContent = `latest ${Math.round(latest.latency_ms)} ms`;
  els.chartCount.textContent = `${activePoints.length} words`;
}

function latencyClass(value) {
  const latency = Number(value);
  if (!Number.isFinite(latency)) {
    return "";
  }
  if (latency < 200) {
    return "good";
  }
  if (latency <= 500) {
    return "warn";
  }
  return "bad";
}

function latencySuffix(value) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) {
    return "";
  }
  return ` · ${Math.round(Number(value))} ms latency`;
}

function formatMetric(value, unit = "") {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) {
    return "n/a";
  }
  const number = Number(value);
  const formatted = Math.abs(number) >= 100 ? number.toFixed(0) : number.toFixed(2);
  return unit ? `${formatted} ${unit}` : formatted;
}

function formatClock(value) {
  if (!Number.isFinite(value)) {
    return "not seen";
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(2)} s`;
  }
  return `${Math.round(value)} ms`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
