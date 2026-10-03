import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, mkdtempSync, mkdirSync, writeFileSync, rmSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { validateEvidence } from './evidence-data.mjs';
const snapshot = () => JSON.parse(readFileSync('content-data/customer-evidence.json', 'utf8'));
test('preserved evidence includes both modalities and missing-value markers', () => {
  const fixture = snapshot();
  fixture.tables.find(t=>t.id==='ecg-detail').rows[4][4]='0.176';
  fixture.tables.find(t=>t.id==='ppg-detail').rows[0][4]='-';
  const data = validateEvidence(fixture);
  assert.equal(data.tables.reduce((n,t) => n+t.rows.length,0),33);
  assert.equal(data.tables.find(t=>t.id==='ecg-detail').rows[4][4],'0.176');
  assert.equal(data.tables.find(t=>t.id==='ppg-detail').rows[0][4],'-');
});
test('rejects incomplete, duplicate and nonfinite evidence', () => {
  const missing=snapshot();missing.tables[0].rows.pop();assert.throws(()=>validateEvidence(missing));
  const duplicate=snapshot();duplicate.tables[1]=duplicate.tables[0];assert.throws(()=>validateEvidence(duplicate));
  const wrongHeader=snapshot();wrongHeader.tables[0].headers[2]='Different metric';assert.throws(()=>validateEvidence(wrongHeader));
  const prose=snapshot();prose.tables[0].rows[0][2]='unknown';assert.throws(()=>validateEvidence(prose));
  const invalid=snapshot();invalid.tables[0].rows[0][2]='NaN';assert.throws(()=>validateEvidence(invalid));
});
test('refresh requires every scorecard and keeps absent measurements missing', () => {
 const dir=mkdtempSync(join(tmpdir(),'ck-evidence-')), output=join(dir,'snapshot.json');
 try {
  for(const modality of ['ppg','ecg'])for(const cr of modality==='ppg'?[2,4,8,16,32]:[2,4,8,16,32,64]){
   const run=`${modality}_rvq_${modality==='ppg'?64:256}hz_${String(cr).padStart(2,'0')}x_golden`;
   mkdirSync(join(dir,run));writeFileSync(join(dir,run,'quality_scorecard.json'),JSON.stringify({num_samples:10,bitrate:{cr_codec_uniform:{mean:cr}},time_domain:{prd_percent:{mean:1.25}}}));
  }
  const deploy=join(dir,'ppg_rvq_64hz_02x_golden','deploy');mkdirSync(deploy);
  writeFileSync(join(deploy,'encoder.tflite'),Buffer.alloc(1024));
  writeFileSync(join(deploy,'deploy_manifest.json'),JSON.stringify({artifacts:{encoder_tflite:'encoder.tflite'}}));
  const run=()=>spawnSync('python3',['scripts/refresh-evidence.py','--results-dir',dir,'--output',output],{encoding:'utf8'});
  const good=run();assert.equal(good.status,0,good.stderr);
  const data=validateEvidence(JSON.parse(readFileSync(output,'utf8')));assert.equal(data.tables[0].rows[0][2],'-');assert.equal(data.tables[0].rows[0][3],'1.25');
  const partial=data.tables.find(t=>t.id==='ppg-files').rows[0];assert.equal(partial[1],'-');assert.equal(partial[2],'1 KiB');
  rmSync(output);rmSync(join(dir,'ppg_rvq_64hz_02x_golden','quality_scorecard.json'));
  assert.notEqual(run().status,0);assert.equal(existsSync(output),false);
 } finally {rmSync(dir,{recursive:true,force:true});}
});
