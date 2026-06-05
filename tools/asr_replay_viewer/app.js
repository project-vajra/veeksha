const state = {
  data: null,
  selected: null,
  sortedRequests: [],
  replayTimeMs: 0,
  replayMaxMs: 0,
  replayPlaying: false,
  replayStartedAt: 0,
  replayBaseMs: 0,
  raf: null,
  latencyChart: null,
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
  playButton: document.getElementById("play-button"),
  pauseButton: document.getElementById("pause-button"),
  timeSlider: document.getElementById("time-slider"),
  timeLabel: document.getElementById("time-label"),
  durationLabel: document.getElementById("duration-label"),
  latencyChart: document.getElementById("latency-chart"),
  latencyChartEmpty: document.getElementById("latency-chart-empty"),
  chartLiveMean: document.getElementById("chart-live-mean"),
  chartLatest: document.getElementById("chart-latest"),
  chartCount: document.getElementById("chart-count"),
  referenceStream: document.getElementById("reference-stream"),
  transcriptStream: document.getElementById("transcript-stream"),
  replayWarning: document.getElementById("replay-warning"),
  sortSelect: document.getElementById("sort-select"),
  requestTable: document.getElementById("request-table"),
};

const replayStatsCache = new WeakMap();
let chartLibraryLoad = null;

els.loadUrl.addEventListener("click", () => loadFromUrl(els.dataUrl.value.trim()));
els.fileInput.addEventListener("change", loadFromFile);
els.playButton.addEventListener("click", startReplay);
els.pauseButton.addEventListener("click", pauseReplay);
els.sortSelect.addEventListener("change", () => {
  sortRequests();
  renderRequestTable();
});
els.timeSlider.addEventListener("input", () => {
  pauseReplay();
  setReplayTime(Number(els.timeSlider.value));
  syncAudioToTime();
});
window.addEventListener("resize", () => {
  state.latencyChart?.resize();
  renderLatencyChart();
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
  state.selected = data.requests?.[0] || null;
  els.runSubtitle.textContent = `${data.run_dir || "run"} · ${data.requests?.length || 0} ASR requests`;
  sortRequests();
  renderRequestTable();
  selectRequest(state.selected);
}

function sortRequests() {
  const requests = [...(state.data?.requests || [])];
  const sortKey = els.sortSelect.value;
  const metric = (request, name) => Number(request.metrics?.[name] ?? -Infinity);
  const interactivity = (request) => {
    const replayStats = replayInteractivityStats(request);
    return replayStats?.mean_latency_ms ?? metric(request, "interactivity");
  };
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
    return interactivity(b) - interactivity(a);
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
          <span class="request-latency">${formatMetric(replayInteractivityStats(request)?.mean_latency_ms ?? request.metrics?.interactivity, "ms")}</span>
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
  setReplayTime(0);
}

function configureTimeline(request) {
  const reference = request.reference_word_timestamps || [];
  const transcript = buildTranscriptTokens(request);
  const durationMs = Number(request.duration_ms || 0);
  const lastReferenceMs = Math.max(0, ...reference.map((word) => Number(word.end_ms || 0)));
  const lastTranscriptMs = Math.max(0, ...transcript.map((word) => Number(word.time_ms || 0)));
  state.replayMaxMs = Math.max(durationMs, lastReferenceMs, lastTranscriptMs) + 300;
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
  const replayStats = replayInteractivityStats(request);
  const interactivity = replayStats?.mean_latency_ms ?? metrics.interactivity;
  const wordCount = replayStats?.word_count ?? metrics.interactivity_word_count ?? 0;
  els.latencyValue.textContent = formatMetric(interactivity, "ms");
  els.latencyDetail.textContent = `${wordCount} matched words · ${formatMetric(metrics.ttfc, "ms")} TTFC`;
  els.latencyPanel.className = `latency-panel ${latencyClass(interactivity)}`;
}

function setReplayTime(ms) {
  state.replayTimeMs = Math.max(0, Math.min(ms, state.replayMaxMs));
  els.timeSlider.value = String(Math.round(state.replayTimeMs));
  els.timeLabel.textContent = formatClock(state.replayTimeMs);
  renderStreams();
  renderLatencyChart();
}

function startReplay() {
  if (!state.selected) {
    return;
  }
  if (state.replayTimeMs >= state.replayMaxMs) {
    setReplayTime(0);
  }
  state.replayPlaying = true;
  state.replayBaseMs = state.replayTimeMs;
  state.replayStartedAt = performance.now();
  syncAudioToTime();
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
}

function tickReplay() {
  if (!state.replayPlaying) {
    return;
  }
  const elapsed = performance.now() - state.replayStartedAt;
  const nextMs = state.replayBaseMs + elapsed;
  setReplayTime(nextMs);
  if (nextMs >= state.replayMaxMs) {
    pauseReplay();
    return;
  }
  state.raf = requestAnimationFrame(tickReplay);
}

function syncAudioToTime() {
  if (!els.audio.src) {
    return;
  }
  const seconds = Math.max(0, state.replayTimeMs / 1000);
  if (Number.isFinite(els.audio.duration)) {
    els.audio.currentTime = Math.min(seconds, els.audio.duration);
  } else {
    els.audio.currentTime = seconds;
  }
}

function renderStreams() {
  const request = state.selected;
  if (!request) {
    return;
  }
  const referenceTokens = buildReferenceTokens(request);
  const transcriptTokens = buildTranscriptTokens(request);
  els.referenceStream.innerHTML = renderTimedWords(referenceTokens, "reference");
  els.transcriptStream.innerHTML = renderTimedWords(transcriptTokens, "transcript");

  const hasReplay = Boolean(request.has_replay);
  els.replayWarning.textContent = hasReplay
    ? ""
    : "This row has scalar metrics but no transcript snapshots, so transcribed words cannot animate.";
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

function buildReferenceTokens(request) {
  return (request.reference_word_timestamps || []).map((word) => ({
    word: String(word.word || ""),
    time_ms: Number(word.start_ms ?? word.end_ms ?? 0),
    end_ms: Number(word.end_ms ?? word.start_ms ?? 0),
  }));
}

function buildTranscriptTokens(request) {
  const finalTokens = displayTokens(request.final_transcript || "");
  if (finalTokens.length === 0) {
    return [];
  }

  const normalizedFinal = finalTokens.map((token) => token.normalized);
  const seen = Array(finalTokens.length).fill(null);
  const snapshots = [...(request.transcript_snapshots || [])].sort(
    (a, b) => Number(a.elapsed_ms || 0) - Number(b.elapsed_ms || 0),
  );

  for (const snapshot of snapshots) {
    const hypothesis = displayTokens(snapshot.transcript || "").map((token) => token.normalized);
    const pairs = lcsPairs(normalizedFinal, hypothesis);
    for (const [finalIndex] of pairs) {
      if (seen[finalIndex] === null) {
        seen[finalIndex] = Number(snapshot.elapsed_ms || 0);
      }
    }
  }

  const reference = buildReferenceTokens(request);
  return finalTokens.map((token, index) => {
    const timeMs = seen[index] ?? Number.POSITIVE_INFINITY;
    const referenceWord = reference[index];
    const referenceEnd = Number(referenceWord?.end_ms ?? referenceWord?.time_ms ?? 0);
    return {
      word: token.display,
      time_ms: timeMs,
      latency_ms: Number.isFinite(timeMs) ? Math.max(0, timeMs - referenceEnd) : null,
    };
  });
}

function buildReferenceLatencyTokens(request) {
  if (request && replayStatsCache.has(request)) {
    return replayStatsCache.get(request).tokens;
  }
  const reference = buildReferenceTokens(request);
  const normalizedReference = reference.map((word) => normalizeWord(word.word));
  const firstSeenMs = Array(reference.length).fill(null);
  const snapshots = [...(request.transcript_snapshots || [])].sort(
    (a, b) => Number(a.elapsed_ms || 0) - Number(b.elapsed_ms || 0),
  );

  for (const snapshot of snapshots) {
    const elapsedMs = Number(snapshot.elapsed_ms || 0);
    const hypothesis = displayTokens(snapshot.transcript || "").map((token) => token.normalized);
    const pairs = lcsPairs(normalizedReference, hypothesis);
    for (const [referenceIndex] of pairs) {
      if (
        firstSeenMs[referenceIndex] === null
        && elapsedMs >= Number(reference[referenceIndex]?.time_ms ?? 0)
      ) {
        firstSeenMs[referenceIndex] = elapsedMs;
      }
    }
  }

  const tokens = reference
    .map((word, index) => {
      const timeMs = firstSeenMs[index];
      const endMs = Number(word.end_ms ?? word.time_ms ?? 0);
      return {
        word: word.word,
        time_ms: timeMs,
        reference_start_ms: Number(word.time_ms ?? 0),
        reference_end_ms: endMs,
        latency_ms: Number.isFinite(timeMs) ? Math.max(0, Number(timeMs) - endMs) : null,
      };
    })
    .filter((word) => Number.isFinite(word.time_ms) && Number.isFinite(word.latency_ms));
  if (request) {
    const meanLatencyMs = tokens.length
      ? tokens.reduce((sum, word) => sum + word.latency_ms, 0) / tokens.length
      : null;
    replayStatsCache.set(request, {
      tokens,
      mean_latency_ms: meanLatencyMs,
      word_count: tokens.length,
    });
  }
  return tokens;
}

function displayTokens(text) {
  return String(text || "")
    .split(/\s+/)
    .filter(Boolean)
    .map((display) => ({
      display,
      normalized: normalizeWord(display),
    }))
    .filter((token) => token.normalized);
}

function lcsPairs(a, b) {
  const rows = a.length + 1;
  const cols = b.length + 1;
  const dp = Array.from({ length: rows }, () => Array(cols).fill(0));
  for (let i = a.length - 1; i >= 0; i -= 1) {
    for (let j = b.length - 1; j >= 0; j -= 1) {
      dp[i][j] = a[i] === b[j]
        ? dp[i + 1][j + 1] + 1
        : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }

  const pairs = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) {
      pairs.push([i, j]);
      i += 1;
      j += 1;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      i += 1;
    } else {
      j += 1;
    }
  }
  return pairs;
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
    syncAudioToTime();
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
  const series = buildLatencySeries(state.selected);
  const activePoints = runningMeanPoints(
    series.points.filter((point) => point.time_ms <= state.replayTimeMs),
  );
  els.latencyChartEmpty.textContent = state.chartLibraryFailed
    ? "Could not load chart library from CDN"
    : "Latency points appear as transcript words arrive";
  els.latencyChartEmpty.classList.toggle(
    "hidden",
    !state.chartLibraryFailed && activePoints.length > 0,
  );
  updateChartStats(activePoints);

  if (!state.latencyChart) {
    ensureLatencyChartLibrary().catch(() => {});
  }
  if (!state.latencyChart) {
    return;
  }

  const pointData = activePoints.map((point) => [
    point.time_ms,
    point.latency_ms,
    point.word,
    point.reference_end_ms,
    point.index,
    point.mean_ms,
  ]);
  const meanData = activePoints.map((point) => [
    point.time_ms,
    point.mean_ms,
    point.word,
    point.latency_ms,
    point.index,
  ]);
  const latestMean = meanData.length ? [meanData.at(-1)] : [];

  state.latencyChart.setOption(
    {
      backgroundColor: "#faf9f6",
      animation: true,
      animationDurationUpdate: state.replayPlaying ? 80 : 180,
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
          data: pointData,
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
          markLine: {
            silent: true,
            symbol: ["none", "none"],
            data: [
              {
                xAxis: state.replayTimeMs,
                lineStyle: {
                  color: "#171717",
                  width: 2,
                  type: "dashed",
                  shadowBlur: 6,
                  shadowColor: "rgba(23, 23, 23, 0.18)",
                },
                label: {
                  formatter: formatClock(state.replayTimeMs),
                  color: "#171717",
                  backgroundColor: "#ffffff",
                  borderColor: "rgba(23, 23, 23, 0.12)",
                  borderWidth: 1,
                  borderRadius: 8,
                  padding: [3, 7],
                  position: "insideEndTop",
                },
              },
            ],
            label: {
              show: true,
            },
            lineStyle: {
              color: "#171717",
            },
          },
        },
        {
          name: "Running mean",
          type: "line",
          data: meanData,
          showSymbol: false,
          smooth: 0.3,
          lineStyle: {
            color: "#195e53",
            width: 3,
            shadowBlur: 8,
            shadowColor: "rgba(25, 94, 83, 0.22)",
          },
        },
        {
          name: "Current mean",
          type: "effectScatter",
          data: latestMean,
          symbolSize: 12,
          rippleEffect: {
            brushType: "stroke",
            scale: 2.8,
          },
          itemStyle: {
            color: "#195e53",
            borderColor: "#faf9f6",
            borderWidth: 2,
          },
          zlevel: 2,
        },
      ],
    },
    {
      notMerge: true,
      lazyUpdate: true,
    },
  );
}

function buildLatencySeries(request) {
  const points = buildReferenceLatencyTokens(request || {})
    .map((token, index) => ({
      index,
      word: token.word,
      time_ms: Number(token.time_ms),
      latency_ms: Number(token.latency_ms),
      reference_end_ms: Number(token.reference_end_ms),
    }))
    .sort((a, b) => a.time_ms - b.time_ms);
  const latencies = points.map((point) => point.latency_ms);
  const maxLatency = Math.max(500, ...latencies);
  return {
    points,
    xMax: Math.max(state.replayMaxMs, ...points.map((point) => point.time_ms), 1000),
    yMin: 0,
    yMax: Math.ceil((maxLatency + 150) / 100) * 100,
  };
}

function replayInteractivityStats(request) {
  if (!request) {
    return null;
  }
  if (!replayStatsCache.has(request)) {
    buildReferenceLatencyTokens(request);
  }
  const stats = replayStatsCache.get(request);
  return stats?.mean_latency_ms === null ? null : stats;
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

function normalizeWord(text) {
  const matches = String(text || "")
    .toLowerCase()
    .match(/[a-z0-9]+(?:'[a-z0-9]+)?/g);
  return matches ? matches.join("") : "";
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
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
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
