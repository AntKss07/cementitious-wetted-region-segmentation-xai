export const pages = ['overview', 'segmentation', 'benchmark', 'xai', 'dataset', 'outputs', 'vlm'];
export function safePage(hash) { const p = hash.replace(/^#/, ''); return pages.includes(p) ? p : 'overview'; }
export function frameSelection(videos, split, video, stem) {
  const subset = videos.filter(v => split === 'all' || v.split === split);
  const selected = subset.find(v => v.video === video) || subset[0];
  return { video: selected?.video || '', stem: selected?.frames.includes(stem) ? stem : selected?.frames[0] || '' };
}
export function warValues(values, mode) {
  const prediction = values?.[mode === 'raw' ? 'war_pred_raw' : 'war_pred_phys'];
  const gt = values?.war_gt;
  return { gt, prediction, error: Number.isFinite(gt) && Number.isFinite(prediction) ? Math.abs(prediction - gt) : null };
}
export function displayNumber(value, places = 4) { return Number.isFinite(value) ? value.toFixed(places) : 'Unavailable'; }
export function canShowPanel(model, view, mode) { return Boolean(model.figure) && (view !== 'error' || model.error_mode === mode); }
export function escapeHTML(value) { return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
