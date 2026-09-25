(() => {
  const byId = id => document.getElementById(id);

  async function getJson(path) {
    const response = await fetch(path, {headers: {'Accept': 'application/json'}});
    if (!response.ok) throw new Error(`${path} returned ${response.status}`);
    return response.json();
  }

  async function refreshSystemHealth() {
    const dot = document.querySelector('.status .dot');
    try {
      const [health, storage] = await Promise.all([
        getJson('/health/ready'),
        getJson('/v1/system/storage')
      ]);
      const dependencyLatency = Object.values(health.checks)
        .reduce((total, check) => total + check.latency_ms, 0);
      byId('readiness').textContent = `${health.status} · v${health.version}`;
      byId('storage-backend').textContent = `${storage.backend} · ${storage.indexed_chunks} chunks`;
      byId('benchmark-runs').textContent = `${storage.benchmark_runs} persisted runs`;
      byId('dependency-latency').textContent = `${dependencyLatency.toFixed(2)} ms`;
      dot.classList.remove('is-error');
    } catch (error) {
      byId('readiness').textContent = 'Unavailable';
      byId('dependency-latency').textContent = 'Health check failed';
      dot.classList.add('is-error');
    }
  }

  refreshSystemHealth();
  const timer = window.setInterval(refreshSystemHealth, 15000);
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) refreshSystemHealth();
  });
  window.addEventListener('pagehide', () => window.clearInterval(timer), {once: true});
  window.ragbenchHealth = {refresh: refreshSystemHealth};
})();
