/**
 * Sorptivity REST Client
 * Interacts with the Python aiohttp backend API.
 */

const API = {
  async getStatus() {
    const res = await fetch("/api/status");
    if (!res.ok) throw new Error(`Status check failed: ${res.statusText}`);
    return res.json();
  },

  async getVideos() {
    const res = await fetch("/api/catalog/videos");
    if (!res.ok) throw new Error(`Failed to fetch videos: ${res.statusText}`);
    return res.json();
  },

  async getFrames(video) {
    const res = await fetch(`/api/catalog/frames?video=${encodeURIComponent(video)}`);
    if (!res.ok) throw new Error(`Failed to fetch frames: ${res.statusText}`);
    return res.json();
  },

  async analyzeBenchmark(payload) {
    const res = await fetch("/api/analyze/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Benchmark analysis failed: ${res.statusText}`);
    }
    return res.json();
  },

  async analyzeUpload(formData) {
    const res = await fetch("/api/analyze/upload", {
      method: "POST",
      body: formData
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `Upload analysis failed: ${res.statusText}`);
    }
    return res.json();
  }
};
