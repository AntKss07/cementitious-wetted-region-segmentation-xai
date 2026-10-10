import {safePage, frameSelection, warValues, displayNumber as fmt, canShowPanel, escapeHTML as esc} from './state.mjs';

const main = document.querySelector('main');
let meta, frame, generation = 0;
let uploadedImage = null, uploadError = '', uploadBusy = false, uploadSequence = 0;
const state = {page: safePage(location.hash), split: 'test', video: 'video 21', stem: 'ezgif-frame-040', mode: 'raw', view: 'prediction', metric: 'dice', category: 'All cases', query: '', type: 'All types'};
const title = {overview: ['Research overview', 'A visual window into capillary water absorption.'], segmentation: ['Segmentation explorer', 'Inspect the same frame across four segmentation architectures.'], benchmark: ['Model benchmark', 'Compare measured performance on the locked test cohort.'], xai: ['Explainability laboratory', 'Inspect U-Net++ spatial attribution with Grad-CAM++.'], dataset: ['Dataset & annotations', 'Trace every frame to its video, split and human annotations.'], outputs: ['Research outputs', 'Explore and download the original research artifacts.'], vlm: ['VLM analysis', 'An evidence-grounded extension for future qualitative analysis.']};
const url = (path, params = {}) => `${path}?${new URLSearchParams(params)}`;
const params = () => ({video: state.video, stem: state.stem, mode: state.mode});
const lookup = path => meta.artifacts.find(a => a.path === path);
const source = path => { const a = lookup(path); return a ? `<a class="source" href="${a.url}?download=1">${esc(path)} ↗</a>` : `<span class="source">${esc(path)} · unavailable</span>`; };
const notice = (text, kind = '') => `<div class="notice ${kind}"><span aria-hidden="true">${kind === 'info' ? 'i' : '!'}</span><p>${esc(text)}</p></div>`;
const empty = text => `<div class="empty"><span>◫</span><strong>Unavailable</strong><p>${esc(text)}</p></div>`;
const pill = text => `<span class="pill">${esc(text)}</span>`;
const heading = (name, text = '', right = '') => `<div class="section-heading"><div><h2>${name}</h2>${text ? `<p>${text}</p>` : ''}</div>${right}</div>`;
function imageCard(label, artifact, note = '', extra = '') {
  return `<article class="image-card"><div class="image-title"><h3>${esc(label)}</h3>${extra}</div>${artifact ? `<button class="image-button" data-image="${esc(artifact.url)}" data-title="${esc(label)}" aria-label="Inspect ${esc(label)} at full size"><img loading="lazy" src="${esc(artifact.url)}" alt="${esc(label)}"><span class="expand">⤢ Inspect</span></button>` : empty(note || 'No saved image for this frame.')}<div class="image-caption">${esc(note)}${artifact ? `<a href="${esc(artifact.url)}${artifact.url.includes('?') ? '&' : '?'}download=1" download>Download ↗</a>` : ''}</div></article>`;
}
function stat(label, value, note) { return `<article class="stat"><span>${label}</span><strong>${value}</strong><small>${note}</small></article>`; }
function table(headers, rows) { return `<div class="table-scroll"><table><thead><tr>${headers.map(h => `<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rows.map(r => `<tr>${r.map(c => `<td>${c}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`; }
function modeControl() { return `<label>Prediction mode<select id="mode"><option value="raw" ${state.mode === 'raw' ? 'selected' : ''}>Raw prediction</option><option value="physics_enforced" ${state.mode === 'physics_enforced' ? 'selected' : ''}>Physics-enforced</option></select></label>`; }
function controls(dataset = false) {
  const videos = meta.videos.filter(v => v.split === state.split);
  const current = videos.find(v => v.video === state.video);
  return `<div class="controls">${dataset ? `<label>Dataset split<select id="split">${['train','val','test'].map(s => `<option value="${s}" ${state.split === s ? 'selected' : ''}>${s === 'val' ? 'Validation' : s[0].toUpperCase()+s.slice(1)}</option>`).join('')}</select></label>` : `<div class="control-static"><span>Evaluation split</span><strong>Test ${pill(`${meta.splits.test.frames} frames`)}</strong></div>`}<label>Video / specimen<select id="video">${videos.map(v => `<option ${v.video === state.video ? 'selected' : ''}>${esc(v.video)}</option>`).join('')}</select></label><label class="frame-select">Frame<select id="stem" aria-label="Frame">${(current?.frames || []).map(s => `<option ${s === state.stem ? 'selected' : ''}>${esc(s)}</option>`).join('')}</select></label><div class="step-buttons"><button id="prev" aria-label="Previous frame" ${current?.frames.indexOf(state.stem) <= 0 ? 'disabled' : ''}>←</button><button id="next" aria-label="Next frame" ${current?.frames.indexOf(state.stem) >= (current?.frames.length || 0)-1 ? 'disabled' : ''}>→</button></div>${dataset || state.page === 'xai' ? '' : modeControl()}</div>`;
}
function benchmarkTable(mode = state.mode) {
  const rows = meta.models.map(m => { const r = meta.benchmarks.find(b => b.model === m.id && b.mode === mode); return [`<strong>${m.name}</strong><small>${m.encoder}</small>`, ...Object.keys(meta.metrics).map(k => fmt(r?.[k]))]; });
  return table(['Architecture', ...Object.values(meta.metrics).map(m => m.label)], rows);
}
function overview() {
  const best = meta.benchmarks.filter(b => b.mode === 'raw' && Number.isFinite(b.dice)).sort((a,b) => b.dice-a.dice)[0];
  const bestModel = meta.models.find(m => m.id === best?.model);
  const total = Object.values(meta.splits).reduce((sum,s) => sum+s.frames, 0);
  return `<section class="hero"><div><div class="eyebrow">COMPUTER VISION × MATERIALS SCIENCE</div><h2>Making the wetted<br>region measurable.</h2><p>Explore how semantic segmentation identifies water ingress in cementitious specimens, quantifies wetted area and reveals spatial model attribution.</p><a class="button primary" href="#segmentation">Explore segmentation <span>↗</span></a><a class="text-link" href="#benchmark">View benchmarks →</a></div><div class="hero-visual">${imageCard('From RGB to attribution', lookup('xai_gradcam_outputs/best_cases/xai_panel_video 1_ezgif-frame-053.png'), 'Saved U-Net++ Grad-CAM++ case · video 1 / frame 053')}<div class="hero-tags">${pill('RGB → MASK → ATTRIBUTION')}${pill('PRECOMPUTED')}</div></div></section>
    <div class="stats">${stat('Frames in manifest', total.toLocaleString(), `${meta.videos.length} videos · video-level split`)}${stat('Benchmark architectures', meta.models.length, 'CNNs and a hierarchical transformer')}${stat('Highest raw test Dice', fmt(best?.dice), bestModel ? `${bestModel.name} · ${bestModel.encoder}` : 'Benchmark unavailable')}${stat('Locked test cohort', meta.splits.test.frames, 'Videos 1, 21 and 24')}</div>
    ${heading('From image to evidence', 'Four connected stages of the research workflow.')}
    <div class="pipeline">${[['01','Source frame','Video-level split and four human annotators'],['02','Segmentation','Four architectures predict the wet region'],['03','WAR quantification','Published ratios and prediction errors'],['04','Explainability','Post-hoc Grad-CAM++ for U-Net++']].map(([n,t,d]) => `<article><span>${n}</span><h3>${t}</h3><p>${d}</p></article>`).join('')}</div>
    <div class="two-columns"><section class="panel">${heading('Dataset partition', 'Authoritative source: root split manifest')}${table(['Partition','Videos','Frames'], Object.entries(meta.splits).map(([s,n]) => [s === 'val' ? 'Validation' : s[0].toUpperCase()+s.slice(1), n.videos, n.frames]))}${source('video_split_assignment.csv')}<p class="muted">Seed 42 is documented in the notebooks. Video-level splitting reduces leakage from correlated frames.</p></section><section class="panel">${heading('Research scope')}<p>Segmentation identifies wet pixels. WAR summarizes their area. Grad-CAM++ shows post-hoc attribution for selected predictions.</p><p>VLM interpretation is planned. Validated sorptivity regression is unavailable because image-to-time-to-specimen linkage has not been established.</p><a class="text-link" href="#xai">Inspect the explainability evidence →</a></section></div>
    ${heading('Headline benchmark', 'Raw predictions · test split · values read from model evaluation CSVs', '<a class="text-link" href="#benchmark">Full comparison →</a>')}${benchmarkTable('raw')}
    ${notice(meta.war_note)}
    <details class="panel provenance"><summary>Data provenance & known discrepancies <span>${meta.warnings.length} notes</span></summary>${meta.warnings.map(w => `<p>${esc(w)}</p>`).join('')}</details>
    ${heading('Research notebooks')}<div class="notebooks">${meta.models.map(m => `<a href="${lookup(m.notebook)?.url || '#outputs'}"><strong>${m.name}</strong><span>${m.encoder} · Notebook ↗</span></a>`).join('')}</div>`;
}
function segmentation() {
  if (uploadedImage) return uploadControls() + uploadedView();
  const view = state.view;
  return `${uploadControls()}${controls()}<div class="context-line"><strong>${esc(frame.video)} / ${esc(frame.stem)}</strong><span>${pill('TEST')}${pill('PRECOMPUTED FIGURES')}<a class="button" href="${url('/api/comparison', params())}">Download comparison ↓</a></span></div>
    ${notice(meta.war_note)}
    <div class="two-columns reference-row">${imageCard('Original input', frame.original, 'Source RGB · native aspect ratio')}${imageCard('Consensus ground truth', frame.consensus, 'Stored reference · wet if at least 2 of 4 annotators vote wet')}</div>
    ${heading('Synchronized model comparison', 'Rendered figure crops; numerical values come from CSVs. Click any image for full-size inspection.')}
    <div class="comparison-toolbar"><div class="segmented" role="group" aria-label="Panel view">${[['prediction','Predicted mask'],['error','Error map'],['probability','Probability']].map(([v,t]) => `<button data-view="${v}" aria-pressed="${view === v}">${t}</button>`).join('')}</div><div class="legend"><span><i class="tp"></i>True positive</span><span><i class="fp"></i>False positive</span><span><i class="fn"></i>False negative</span><span><i class="tn"></i>True negative</span></div></div>
    <div class="model-grid">${frame.models.map(m => {
      const war = warValues(m.values, state.mode);
      const available = view !== 'probability' && canShowPanel(m, view, state.mode);
      const img = available ? {url: url('/api/panel', {...params(), model:m.id, view})} : null;
      const reason = view === 'probability' ? 'Standalone probability arrays are not saved. Selected U-Net++ probability visualizations are available in the XAI laboratory.' : !m.figure ? 'No saved prediction figure for this frame. CSV measurements may still be available.' : `The saved error map uses ${m.error_mode === 'raw' ? 'raw' : 'physics-enforced'} predictions. No error map is saved for the selected mode.`;
      return `<article class="model-panel" style="--model:${m.color}"><div class="model-heading"><h3>${m.name}</h3><span>${m.encoder}</span></div>${img ? `<button class="image-button" data-image="${esc(img.url)}" data-title="${m.name} · ${esc(frame.video)} / ${esc(frame.stem)} · ${state.mode} · ${view}" aria-label="Inspect ${m.name} ${view}"><img src="${esc(img.url)}" alt="${m.name} ${state.mode} ${view}"><span class="expand">⤢ Inspect</span></button>` : empty(reason)}<div class="model-details"><div class="mode-label">${state.mode === 'raw' ? 'Raw prediction' : 'Physics-enforced prediction'}</div><dl><div><dt>Reference WAR</dt><dd>${fmt(war.gt)}</dd></div><div><dt>Predicted WAR</dt><dd>${fmt(war.prediction)}</dd></div><div><dt>Absolute WAR error</dt><dd>${fmt(war.error)}</dd></div></dl>${m.figure ? `<a href="${m.figure.url}?download=1">Original comparison figure ↗</a>` : ''}<details><summary>Measurement source</summary>${source(m.source)}<p>Checkpoint / run identity is not recorded. Segmentation metrics per frame are unavailable in this benchmark CSV.</p></details></div></article>`;
    }).join('')}</div>
    ${notice('All panels use this selected frame. WAR values are full-image ratios. Figure and CSV checkpoint identity is not recorded; SegFormer-B1 displayed values use its figure-export CSV. Error maps are shown only for their saved prediction mode.', 'info')}
    <p class="muted">All four saved benchmark predictions can be inspected together on video 21 / ezgif-frame-040. Raw and physics-enforced masks can differ; benchmark overlap metrics were reused from raw predictions.</p>`;
}
function uploadControls() {
  return `<section class="upload-bar" aria-label="Image upload"><div><strong>Inspect your own image</strong><p>JPEG, PNG or WebP · up to 20 MB · preview stays in this browser session</p></div><div class="upload-actions"><input id="image-upload" type="file" accept="image/jpeg,image/png,image/webp" hidden aria-label="Choose image"><button id="upload-image" ${uploadBusy ? 'disabled' : ''}>${uploadBusy ? 'Reading image…' : uploadedImage ? 'Replace image ↑' : 'Upload image ↑'}</button>${uploadedImage ? '<button id="clear-upload">Return to dataset</button>' : ''}</div><p class="upload-status" role="status">${uploadBusy ? 'Validating image…' : ''}</p>${uploadError ? `<p class="upload-error" role="alert">${esc(uploadError)}</p>` : ''}</section>`;
}
function uploadedView() {
  const image = uploadedImage;
  return `<div class="context-line"><strong>${esc(image.name)}</strong>${pill('UPLOADED IMAGE · PREVIEW ONLY')}</div><div class="two-columns reference-row"><article class="image-card"><div class="image-title"><h3>Uploaded image</h3></div><button class="image-button" data-image="${image.url}" data-title="${esc(image.name)}" aria-label="Inspect uploaded image at full size"><img src="${image.url}" alt="Uploaded specimen preview"><span class="expand">⤢ Inspect</span></button><div class="image-caption">${image.width} × ${image.height} pixels · ${(image.size / 1024 / 1024).toFixed(2)} MB · native aspect ratio</div></article><section class="panel">${heading('Ready for visual inspection')}<p>Your image is loaded locally. You can open it at full size or replace it with another image.</p>${notice('Live segmentation is unavailable: trained checkpoints and a verified inference adapter are not configured. This upload has no predicted mask, WAR, reference annotation or Grad-CAM++ result.')}<p class="muted">The image is kept only in this browser session and is cleared on refresh. Return to the dataset to explore saved research results.</p></section></div>${heading('Model predictions', 'No model has analyzed this uploaded image.')}<div class="model-grid">${meta.models.map(m => `<article class="model-panel" style="--model:${m.color}"><div class="model-heading"><h3>${m.name}</h3><span>${m.encoder}</span></div>${empty('Live inference is unavailable for uploaded images.')}</article>`).join('')}</div>`;
}
async function loadUpload(file) {
  if (!file) return;
  const ticket = ++uploadSequence;
  uploadBusy = true; uploadError = '';
  render();
  let objectURL;
  try {
    if (!['image/jpeg','image/png','image/webp'].includes(file.type)) throw Error('Choose a JPEG, PNG or WebP image.');
    if (!file.size || file.size > 20 * 1024 * 1024) throw Error('Choose a non-empty image no larger than 20 MB.');
    objectURL = URL.createObjectURL(file);
    const image = new Image();
    image.src = objectURL;
    try { await image.decode(); } catch { throw Error('This file could not be decoded as an image. Choose a valid JPEG, PNG or WebP.'); }
    if (!image.naturalWidth || !image.naturalHeight || image.naturalWidth * image.naturalHeight > 40000000) throw Error('Choose an image with no more than 40 million pixels.');
    if (ticket !== uploadSequence) return;
    if (uploadedImage) URL.revokeObjectURL(uploadedImage.url);
    uploadedImage = {url:objectURL, name:file.name, size:file.size, width:image.naturalWidth, height:image.naturalHeight};
    objectURL = null;
  } catch (error) {
    if (ticket === uploadSequence) uploadError = error.message;
  } finally {
    if (objectURL) URL.revokeObjectURL(objectURL);
    if (ticket === uploadSequence) { uploadBusy = false; render(); }
  }
}
function benchmark() {
  const key = state.metric, definition = meta.metrics[key];
  const items = meta.models.map(m => ({...m, value:meta.benchmarks.find(b => b.model === m.id && b.mode === state.mode)?.[key]}));
  const max = key.startsWith('war_') && key !== 'war_r2' ? Math.max(...items.map(i => i.value || 0), .001)*1.1 : 1;
  return `<div class="controls"><div class="control-static"><span>Evaluation cohort</span><strong>Test · ${meta.splits.test.frames} frames</strong></div>${modeControl()}<label>Compare metric<select id="metric">${Object.entries(meta.metrics).map(([k,m]) => `<option value="${k}" ${k===key?'selected':''}>${m.label}</option>`).join('')}</select></label>${pill('VIDEOS 1 · 21 · 24')}</div>
    ${notice(meta.war_note)}${state.mode === 'physics_enforced' ? notice('The published physics-enforced row reuses raw Dice, IoU, precision, recall and pixel accuracy. Its WAR errors are separately evaluated after mask post-processing.') : ''}
    <div class="two-columns"><section class="panel">${heading(`${definition.label} comparison`, definition.description)}<div class="bars">${items.map(i => `<div class="bar-row"><span>${i.name}</span><div class="bar-track"><div style="width:${Math.max(0,Math.min(100,(i.value || 0)/max*100))}%;background:${i.color}"></div></div><strong>${fmt(i.value)}</strong></div>`).join('')}</div><div class="axis"><span>0</span><span>${fmt(max,3)}</span></div></section><section class="panel">${heading('Evaluation provenance')}<p>Each row is read from the architecture’s <code>metrics_test.csv</code>. The root manifest defines the cohort; frame IDs are checked against each model’s prediction CSV.</p>${meta.models.map(m => `<p class="coverage"><strong>${m.name}</strong><span>${meta.coverage[m.id].frames} frames · ${meta.coverage[m.id].matches_test ? 'cohort matches' : 'cohort mismatch'}</span></p>`).join('')}<p class="muted">Checkpoint hashes and run IDs are unavailable. Cohort alignment does not establish checkpoint identity.</p></section></div>
    ${heading('Benchmark results', `${state.mode === 'raw' ? 'Raw' : 'Physics-enforced'} · dimensionless ratios · unavailable metrics shown explicitly`)}${benchmarkTable()}
    <div class="sources">${meta.models.map(m => source(`evaluation/${m.id}/metrics_test.csv`)).join('')}</div>
    ${heading('Reading the metrics')}<div class="metric-definitions">${Object.values(meta.metrics).map(m => `<article><h3>${m.label}</h3><p>${m.description}</p></article>`).join('')}</div>`;
}
function xai() {
  const metrics = frame.xai_metrics;
  const categories = ['All cases', ...new Set(meta.cases.map(c => c.category))];
  const cases = meta.cases.filter(c => state.category === 'All cases' || c.category === state.category);
  const summary = meta.xai_summary;
  const correlations = summary.correlation_diagnostics || {};
  return `${controls()}${notice(meta.xai_note)}<div class="context-line"><strong>U-Net++ · ResNet-34</strong>${pill('SEPARATE XAI EVALUATION')}<code>decoder.blocks["x_0_4"]</code></div>
    <p>Grad-CAM++ is a post-hoc attribution method. It is not a model uncertainty estimate or causal proof. All images below are saved notebook outputs; no new explanation is generated.</p>
    <div class="two-columns"><section class="panel">${heading('Representative cases')}<label>Case category<select id="category">${categories.map(c => `<option ${state.category===c?'selected':''}>${esc(c)}</option>`).join('')}</select></label><div class="case-list">${cases.map(c => `<button data-case-video="${esc(c.video)}" data-case-stem="${esc(c.stem)}" class="${c.video===state.video&&c.stem===state.stem?'selected':''}"><span>${esc(c.video)} / ${esc(c.stem)}</span><small>${esc(c.category)} · Dice ${fmt(Number(c.dice))}</small></button>`).join('') || '<p>No representative cases available.</p>'}</div></section><section class="panel">${heading('Selected-frame diagnostics', `${esc(state.video)} / ${esc(state.stem)} · XAI run`)}${metrics ? table(['Measurement','Value'], [['Dice',fmt(metrics.dice)],['IoU',fmt(metrics.iou)],['Published reference WAR',fmt(metrics.gt_war)],['Published prediction WAR',fmt(metrics.pred_war)],['Absolute WAR error',fmt(metrics.war_abs_error)],['CAM within approximate ROI',percent(metrics.roi_cam_fraction)],['CAM within predicted wet region',percent(metrics.wet_cam_fraction)],['CAM outside approximate ROI',percent(metrics.bg_cam_fraction)],['CAM around boundary',percent(metrics.boundary_cam_fraction)]]) : empty('No XAI diagnostics for this frame.')}${source('xai_gradcam_outputs/tables/xai_spatial_attribution_diagnostics.csv')}</section></div>
    ${heading('Saved six-panel explanation', 'RGB · ground truth · prediction · probability · Grad-CAM++ · overlay')}
    ${frame.xai_figures.length ? frame.xai_figures.map(f => `<div class="wide-figure">${imageCard(`${frame.video} / ${frame.stem}`, f, 'Precomputed XAI run · open full size to inspect panels and scales')}</div>`).join('') : empty('No six-panel XAI figure is saved for this frame. Choose a representative case above; diagnostics may exist for other test frames.')}
    ${heading('Layer comparison & supporting evidence', 'Saved figures may contain different, explicitly labelled frames. They are not synchronized predictions for the selected frame.')}
    <div class="two-columns">${imageCard('Layer comparison', lookup('xai_gradcam_outputs/layer_comparison/layer_comparison_sample.png'), 'encoder.layer4 · decoder x_0_3 · primary decoder x_0_4')}${imageCard('Attribution versus WAR error', lookup('xai_gradcam_outputs/figures/attribution_vs_war_error.png'), 'Across the saved XAI test cohort')}</div>
    <div class="two-columns"><section class="panel">${heading('Cohort correlations')}${table(['Diagnostic','Reported value'], [['Background CAM vs Dice · Pearson r',fmt(correlations.dice_vs_bg_cam_pearson_r)],['Background CAM vs absolute WAR error · Spearman ρ',fmt(correlations.war_error_vs_bg_cam_spearman_rho)]])}<p>Correlation describes association within this evaluation. It does not establish a causal explanation.</p>${source('xai_gradcam_outputs/tables/xai_summary.json')}</section><section class="panel">${heading('Run distinction')}<p>XAI summary Dice: <strong>${fmt(summary.benchmark_metrics?.dice,6)}</strong>. Benchmark U-Net++ Dice: <strong>${fmt(meta.benchmarks.find(b=>b.model==='unetpp'&&b.mode==='raw')?.dice,6)}</strong>.</p><p>Keep these evaluation sources separate. The XAI summary names a checkpoint path, but the checkpoint is ${meta.checkpoints.length ? 'not verified against these figures' : 'absent from this checkout'}.</p>${notice(meta.war_note)}</section></div>
    <div class="two-columns">${imageCard('Boundary analysis',lookup('xai_gradcam_outputs/figures/boundary_analysis_cases.png'),'Saved multi-frame boundary analysis')}${imageCard('Failure analysis',lookup('xai_gradcam_outputs/figures/failure_analysis_detailed.png'),'Saved RGB, ground truth, prediction, error and CAM evidence')}</div>`;
}
function percent(v) { return Number.isFinite(v) ? `${(100*v).toFixed(2)}%` : 'Unavailable'; }
function dataset() {
  const allAnnotations = frame.annotators.every(Boolean);
  return `${controls(true)}<div class="context-line"><strong>${esc(frame.video)} / ${esc(frame.stem)}</strong>${pill(frame.split)}${pill('FOUR HUMAN ANNOTATORS')}</div>
    <div class="two-columns">${imageCard('Original frame',frame.original,frame.original?.path || 'Source image unavailable')}${imageCard('Stored consensus mask',frame.consensus,'Reference from repository · ≥2 wet votes out of 4')}</div>
    ${heading('Independent annotations','Original colored masks are preserved. Foreground follows the notebook rule: any nonzero RGB channel.')}
    <div class="four-columns">${frame.annotators.map((a,i)=>imageCard(`Annotator ${i+1}`,a,a?.path || 'This annotator mask is missing.')).join('')}</div>
    <div class="two-columns">${imageCard('Annotator disagreement',allAnnotations ? {url:url('/api/agreement',params())} : null,'Computed from all four original masks at their native dimensions.')}<section class="panel">${heading('Agreement & consensus')}<div class="agreement-legend"><p><i style="background:#18273b"></i><strong>Unanimous dry</strong> 0 / 4 wet votes</p><p><i style="background:#f7bb53"></i><strong>Partial agreement</strong> 1–3 / 4 wet votes</p><p><i style="background:#319c94"></i><strong>Unanimous wet</strong> 4 / 4 wet votes</p></div><p>The consensus rule is wet at ≥2 votes. The stored consensus is shown directly; disagreement is calculated on demand. Missing masks are never treated as dry annotations.</p><p>These are wet-region annotations. They do not define the complete specimen ROI.</p>${source('video_split_assignment.csv')}</section></div>
    ${heading('Video inventory', 'Declared frame counts and files available in the checked-out dataset')}${table(['Video / specimen','Split','Manifest frames','Available frames'],meta.videos.map(v=>[esc(v.video),pill(v.split),v.declared_frames,v.frames.length]))}`;
}
function outputs() {
  const types = ['All types',...new Set(meta.artifacts.map(a=>a.type))].sort();
  const matches = meta.artifacts.filter(a => (state.type==='All types'||a.type===state.type) && a.path.toLowerCase().includes(state.query.toLowerCase()));
  return `<div class="controls"><label class="search">Search artifacts<input id="search" type="search" placeholder="Search filename, model or folder…" value="${esc(state.query)}"></label><label>File type<select id="type">${types.map(t=>`<option ${t===state.type?'selected':''}>${t}</option>`).join('')}</select></label><span id="result-count">${matches.length} artifacts</span></div><div id="artifact-results">${artifactTable(matches)}</div>
    <section class="panel">${heading('Artifact availability')}<p>Live inference: unavailable. ${meta.checkpoints.length ? `${meta.checkpoints.length} checkpoint file(s) detected, but no verified inference adapter is configured.` : 'No trained checkpoint files are present.'}</p><p>VLM backend: unavailable. <code>build_xai_notebook.py</code> is documented in the original README but absent from this checkout.</p><p>Standalone prediction masks and probability arrays are not committed. Saved figures are displayed as visualizations; they are never used to recover numerical measurements.</p></section>`;
}
function artifactTable(matches) {
  return matches.length ? table(['Artifact / relative source path','Type','Size','Actions'], matches.map(a=>[`<strong>${esc(a.name)}</strong><small>${esc(a.path)}</small>`,pill(a.type),`${(a.bytes/1024).toFixed(0)} KB`,`${a.type==='PNG'?`<button class="text-button" data-image="${a.url}" data-title="${esc(a.name)}">Preview</button> `:''}<a href="${a.url}?download=1">Download ↓</a>`])) : empty('No artifacts match these filters.');
}
function vlm() {
  return `<section class="panel planned"><div class="eyebrow">PROPOSED RESEARCH EXTENSION</div><h2>Evidence-grounded interpretation</h2><p>A future vision-language module could describe segmentation and attribution evidence alongside verified measurements. This repository currently has no VLM backend.</p>${pill('PLANNED · UNAVAILABLE')}<div class="pipeline"><article><span>01</span><h3>Visual evidence</h3><p>Original frame, prediction, reference, probability and Grad-CAM++.</p></article><article><span>02</span><h3>Verified context</h3><p>Frame ID, model identity, evaluation run, WAR definition and metrics.</p></article><article><span>03</span><h3>Structured observations</h3><p>Boundary quality, attribution location, background sensitivity and possible failure modes.</p></article></div><p>Future output must distinguish observations from hypotheses, cite supporting evidence and express uncertainty cautiously. Numerical measurements must come from verified computation. A single image cannot establish a physical absorption coefficient; Grad-CAM++ cannot establish causality.</p><p class="muted">Any future external integration must run on a secure backend with credentials supplied through environment variables.</p></section>`;
}
async function api(path) { const response = await fetch(path); const data = await response.json(); if(!response.ok) throw Error(data.error || 'Could not read artifacts'); return data; }
async function render() {
  const ticket = ++generation;
  state.page = safePage(location.hash);
  main.dataset.page = state.page;
  document.querySelectorAll('nav a').forEach(a=>a.setAttribute('aria-current',a.hash===`#${state.page}`?'page':'false'));
  main.setAttribute('aria-busy','true');
  main.innerHTML = '<div class="loading" role="status">Reading research artifacts…</div>';
  try {
    if(!meta) meta = await api('/api/meta');
    if(['segmentation','dataset','xai'].includes(state.page) && !(state.page === 'segmentation' && uploadedImage)) {
      if(state.page !== 'dataset') state.split='test';
      Object.assign(state,frameSelection(meta.videos,state.split,state.video,state.stem));
      if(!state.video || !state.stem) throw Error('No frames available for this split. Inspect the dataset manifest and files.');
      const result = await api(url('/api/frame',params()));
      if(ticket!==generation) return;
      frame = result;
    }
    if(ticket!==generation) return;
    main.innerHTML = `<div class="page-heading"><div><div class="eyebrow">CEMENTITIOUS WETTED-REGION SEGMENTATION</div><h1>${title[state.page][0]}</h1><p>${title[state.page][1]}</p></div><a class="help-link" href="#outputs">Source collection ↗</a></div>${({overview,segmentation,benchmark,xai,dataset,outputs,vlm})[state.page]()}`;
    bind();
  } catch(e) {
    if(ticket!==generation) return;
    main.innerHTML=`<section class="panel"><h1>Research data unavailable</h1><p role="alert">${esc(e.message)}</p><button id="retry">Retry</button></section>`;
    document.querySelector('#retry').onclick=()=>{meta=null;render();};
  } finally { if(ticket===generation) main.removeAttribute('aria-busy'); }
}
function bind() {
  const upload = document.getElementById('image-upload');
  if (upload) {
    document.getElementById('upload-image').onclick = () => upload.click();
    upload.onchange = () => loadUpload(upload.files[0]);
  }
  const clear = document.getElementById('clear-upload');
  if (clear) clear.onclick = () => {
    ++uploadSequence;
    URL.revokeObjectURL(uploadedImage.url);
    uploadedImage = null; uploadError = ''; uploadBusy = false;
    render();
  };
  for(const key of ['split','video','stem','mode','metric','category','type']) {
    const element=document.getElementById(key);
    if(element) element.onchange=()=>{state[key]=element.value;render();};
  }
  for(const [id,step] of [['prev',-1],['next',1]]) {
    const element=document.getElementById(id);
    if(element) element.onclick=()=>{const frames=meta.videos.find(v=>v.video===state.video).frames; state.stem=frames[Math.max(0,Math.min(frames.length-1,frames.indexOf(state.stem)+step))];render();};
  }
  document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{state.view=b.dataset.view;render();});
  document.querySelectorAll('[data-case-video]').forEach(b=>b.onclick=()=>{state.video=b.dataset.caseVideo;state.stem=b.dataset.caseStem;render();});
  const search=document.getElementById('search');
  if(search) search.oninput=()=>{state.query=search.value; const matches=meta.artifacts.filter(a=>(state.type==='All types'||a.type===state.type)&&a.path.toLowerCase().includes(state.query.toLowerCase()));document.getElementById('result-count').textContent=`${matches.length} artifacts`;document.getElementById('artifact-results').innerHTML=artifactTable(matches);};
}
const dialog=document.getElementById('image-dialog');
document.addEventListener('click',e=>{const b=e.target.closest('[data-image]'); if(!b)return; document.getElementById('dialog-title').textContent=b.dataset.title; const img=document.getElementById('dialog-image');img.src=b.dataset.image;img.alt=b.dataset.title;img.style.width='100%'; document.getElementById('dialog-download').href=b.dataset.image.startsWith('blob:') ? b.dataset.image : b.dataset.image+(b.dataset.image.includes('?')?'&':'?')+'download=1';document.getElementById('zoom').value=100;document.getElementById('zoom-value').value='100%';dialog.showModal();});
document.getElementById('close-dialog').onclick=()=>dialog.close();
document.getElementById('zoom').oninput=e=>{document.getElementById('dialog-image').style.width=`${e.target.value}%`;document.getElementById('zoom-value').value=`${e.target.value}%`;};
document.addEventListener('error',e=>{if(e.target.tagName==='IMG'){const p=document.createElement('p');p.className='image-error';p.textContent='Image unavailable or layout changed. Download the source figure from Research outputs.'; e.target.replaceWith(p);}},true);
window.addEventListener('hashchange',()=>{render();window.scrollTo(0,0);});
render();
