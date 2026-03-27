/* ============================================================
   SemantiCache Dashboard - Real-time metrics client
   ============================================================ */

(function () {
  "use strict";

  // --- DOM refs -----------------------------------------------------------
  const $hitRate         = document.getElementById("hitRate");
  const $hitRateSub      = document.getElementById("hitRateSub");
  const $costSaved       = document.getElementById("costSaved");
  const $totalRequests   = document.getElementById("totalRequests");
  const $totalRequestsSub = document.getElementById("totalRequestsSub");
  const $cacheSize       = document.getElementById("cacheSize");
  const $queryList       = document.getElementById("queryList");
  const $connStatus      = document.getElementById("connectionStatus");
  const $statusText      = $connStatus.querySelector(".status-text");

  // --- Chart setup --------------------------------------------------------
  const ctx = document.getElementById("similarityChart").getContext("2d");

  const CHART_LABELS = [
    "0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
    "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0",
  ];

  const chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: CHART_LABELS,
      datasets: [{
        label: "Queries",
        data: new Array(10).fill(0),
        backgroundColor: "rgba(59, 130, 246, 0.5)",
        borderColor: "rgba(59, 130, 246, 0.8)",
        borderWidth: 1,
        borderRadius: 4,
        maxBarThickness: 40,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400, easing: "easeOutQuart" },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#1a1a1a",
          titleColor: "#ededed",
          bodyColor: "#888",
          borderColor: "#2a2a2a",
          borderWidth: 1,
          cornerRadius: 8,
          padding: 10,
          titleFont: { family: "'Inter', sans-serif", weight: "600", size: 12 },
          bodyFont: { family: "'Inter', sans-serif", size: 11 },
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(255,255,255,0.04)", drawBorder: false },
          ticks: {
            color: "#555",
            font: { family: "'Inter', sans-serif", size: 10 },
            maxRotation: 0,
          },
          border: { display: false },
        },
        y: {
          beginAtZero: true,
          grid: { color: "rgba(255,255,255,0.04)", drawBorder: false },
          ticks: {
            color: "#555",
            font: { family: "'Inter', sans-serif", size: 10 },
            precision: 0,
          },
          border: { display: false },
        },
      },
    },
  });

  // --- Number animation helper --------------------------------------------
  function animateValue(el, newHTML) {
    if (el.innerHTML === newHTML) return;
    el.classList.add("updating");
    el.innerHTML = newHTML;
    requestAnimationFrame(() => {
      requestAnimationFrame(() => el.classList.remove("updating"));
    });
  }

  // --- Metric updater -----------------------------------------------------
  function applyMetrics(metrics) {
    if (!metrics) return;

    const hitRate = (metrics.hit_rate ?? 0) * 100;
    animateValue($hitRate, hitRate.toFixed(1) + '<span class="metric-unit">%</span>');

    const hits = metrics.cache_hits ?? 0;
    const misses = metrics.cache_misses ?? 0;
    $hitRateSub.textContent = hits + " hits / " + misses + " misses";

    const cost = metrics.cost_saved ?? 0;
    animateValue($costSaved, '<span class="metric-unit">$</span>' + cost.toFixed(2));

    const total = metrics.total_requests ?? 0;
    animateValue($totalRequests, total.toLocaleString());
    $totalRequestsSub.textContent = "Lifetime queries";

    animateValue($cacheSize, (metrics.cache_size ?? 0).toLocaleString());

    // Similarity distribution
    const dist = metrics.similarity_distribution;
    if (Array.isArray(dist) && dist.length > 0) {
      chart.data.datasets[0].data = dist.length === 10
        ? dist
        : bucketize(dist);
      chart.update("none");
    }
  }

  /** If the distribution comes as raw scores, bucket into 10 bins. */
  function bucketize(scores) {
    const bins = new Array(10).fill(0);
    for (const s of scores) {
      const idx = Math.min(Math.floor(s * 10), 9);
      bins[idx]++;
    }
    return bins;
  }

  // --- Top queries renderer -----------------------------------------------
  function applyTopQueries(queries) {
    if (!Array.isArray(queries) || queries.length === 0) {
      $queryList.innerHTML = '<div class="empty-state">No queries cached yet</div>';
      return;
    }

    $queryList.innerHTML = queries.map(function (q, i) {
      const text = sanitize(q.query ?? q.text ?? "");
      const hits = q.hits ?? q.count ?? 0;
      const sim  = q.avg_similarity ?? q.similarity ?? 0;
      return (
        '<div class="query-item">' +
          '<span class="query-rank">' + (i + 1) + '</span>' +
          '<span class="query-text" title="' + text + '">' + text + '</span>' +
          '<div class="query-meta">' +
            '<span class="query-hits">' + hits + ' hits</span>' +
            '<span class="query-similarity">' + (sim * 100).toFixed(1) + '%</span>' +
          '</div>' +
        '</div>'
      );
    }).join("");
  }

  function sanitize(str) {
    var el = document.createElement("span");
    el.textContent = str;
    return el.innerHTML;
  }

  // --- WebSocket with auto-reconnect --------------------------------------
  var ws = null;
  var reconnectDelay = 1000;
  var reconnectTimer = null;

  function setStatus(state) {
    $connStatus.className = "connection-status " + state;
    var labels = { connected: "Live", disconnected: "Disconnected", connecting: "Connecting" };
    $statusText.textContent = labels[state] || state;
  }

  function connect() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    setStatus("connecting");

    var proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(proto + "//" + location.host + "/ws");

    ws.onopen = function () {
      setStatus("connected");
      reconnectDelay = 1000;
    };

    ws.onmessage = function (event) {
      try {
        var data = JSON.parse(event.data);
        if (data.metrics)     applyMetrics(data.metrics);
        if (data.top_queries) applyTopQueries(data.top_queries);
      } catch (_) { /* ignore malformed frames */ }
    };

    ws.onclose = function () {
      setStatus("disconnected");
      scheduleReconnect();
    };

    ws.onerror = function () {
      ws.close();
    };
  }

  function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(function () {
      reconnectTimer = null;
      reconnectDelay = Math.min(reconnectDelay * 1.5, 15000);
      connect();
    }, reconnectDelay);
  }

  // --- Initial HTTP fetch (fallback if WS not yet ready) ------------------
  fetch("/api/metrics")
    .then(function (r) { return r.json(); })
    .then(applyMetrics)
    .catch(function () {});

  fetch("/api/top-queries")
    .then(function (r) { return r.json(); })
    .then(applyTopQueries)
    .catch(function () {});

  // --- Boot ---------------------------------------------------------------
  connect();
})();
