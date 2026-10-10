import test from 'node:test';
import assert from 'node:assert/strict';
import {frameSelection, warValues, canShowPanel, safePage, escapeHTML, displayNumber} from '../research_app/static/state.mjs';

test('navigation accepts only known pages', () => {
  assert.equal(safePage('#xai'), 'xai');
  assert.equal(safePage('#<script>'), 'overview');
});
test('changing split/video preserves only valid frame IDs', () => {
  const videos=[{video:'video 1',split:'test',frames:['001','002']},{video:'video 2',split:'train',frames:['003']}];
  assert.deepEqual(frameSelection(videos,'test','video 1','002'),{video:'video 1',stem:'002'});
  assert.deepEqual(frameSelection(videos,'train','video 1','002'),{video:'video 2',stem:'003'});
  assert.deepEqual(frameSelection(videos,'val','video 1','002'),{video:'',stem:''});
});
test('mode changes select actual WAR columns and preserve missing values', () => {
  const values={war_gt:.2,war_pred_raw:.3,war_pred_phys:.25};
  assert.ok(Math.abs(warValues(values,'physics_enforced').error-.05) < 1e-12);
  assert.equal(warValues(values,'raw').prediction,.3);
});
test('missing measurements cannot be rendered as zero', () => {
  assert.equal(warValues(null,'raw').error,null);
  assert.equal(displayNumber(null),'Unavailable');
  assert.equal(displayNumber(0),'0.0000');
});
test('error maps remain tied to their saved mode', () => {
  assert.equal(canShowPanel({figure:{},error_mode:'raw'},'error','physics_enforced'),false);
  assert.equal(canShowPanel({figure:{},error_mode:'raw'},'prediction','physics_enforced'),true);
  assert.equal(canShowPanel({figure:null},'prediction','raw'),false);
});
test('artifact metadata is escaped before HTML rendering', () => {
  assert.equal(escapeHTML('<img onerror="x">'), '&lt;img onerror=&quot;x&quot;&gt;');
});
