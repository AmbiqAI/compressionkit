import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,mkdtempSync,mkdirSync,writeFileSync,rmSync,existsSync} from 'node:fs';
import {join} from 'node:path';
import {tmpdir} from 'node:os';
import {spawnSync} from 'node:child_process';
import {validateFidelity} from './fidelity-data.mjs';
const snapshot=m=>JSON.parse(readFileSync(`content-data/fidelity-${m}.json`));
test('fidelity snapshots retain complete ratio and noise-bucket coverage',()=>{
 for(const m of ['ppg','ecg'])validateFidelity(snapshot(m));
 const invalid=snapshot('ecg');invalid.tables[3].rows.pop();assert.throws(()=>validateFidelity(invalid));
 const duplicate=snapshot('ppg');duplicate.tables[0].rows[1][0]='02x';assert.throws(()=>validateFidelity(duplicate));
});
test('refresh preserves zero, distinguishes missing and refuses incomplete input',()=>{
 const dir=mkdtempSync(join(tmpdir(),'ck-fidelity-')),out=join(dir,'output');
 try{
  for(const m of ['ppg','ecg'])for(const cr of m==='ppg'?[2,4,8,16,32]:[2,4,8,16,32,64]){
   const name=`${m}_rvq_${m==='ppg'?64:256}hz_${String(cr).padStart(2,'0')}x_golden`;
   mkdirSync(join(dir,name));writeFileSync(join(dir,name,'quality_scorecard.json'),JSON.stringify({num_samples:10,bitrate:{cr_codec_uniform:cr},long_recording:{seam_ratio:0,stitching:{seam_ratio:7}},time_domain:{prd_percent:{mean:0}}}));
  }
  const run=()=>spawnSync('python3',['scripts/refresh-fidelity.py','--results-dir',dir,'--output-dir',out],{encoding:'utf8'});
  const good=run();assert.equal(good.status,0,good.stderr);
  for(const m of ['ppg','ecg']){
   const d=validateFidelity(JSON.parse(readFileSync(join(out,`fidelity-${m}.json`))));
   assert.equal(d.tables[2].rows[0][1],'0.000');assert.equal(d.tables[1].rows[0][1],'—');assert.equal(d.tables[0].rows[0][5],'0.00');
  }
  rmSync(out,{recursive:true});rmSync(join(dir,'ecg_rvq_256hz_64x_golden','quality_scorecard.json'));
  assert.notEqual(run().status,0);assert.equal(existsSync(out),false);
 }finally{rmSync(dir,{recursive:true,force:true});}
});
