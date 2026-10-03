const ids = ['headline', 'physiology', 'seams', 'noise', 'noise-spectral'];
export function validateFidelity(data) {
  if (data.schemaVersion !== 1 || !['ppg','ecg'].includes(data.modality)) throw Error('Invalid fidelity schema/modality');
  if (!['preserved-documentation','scorecards'].includes(data.provenance?.kind) || !data.provenance.source || !/^[a-f0-9]{64}$/.test(data.provenance.sha256 ?? '')) throw Error('Invalid fidelity provenance');
  const band = data.modality === 'ppg' ? 'Pulse-band PSD err' : 'QRS-band PSD err';
  const headers = [
    ['CR','Codec CR','Effective CR','bits/tok','N','Faithful PRD%'],
    ['CR','Truth PRD% (clean)','PRDN-noise%','HR MAE (bpm)',band,'Coherence'],
    ['CR','Seam ratio'],
    ['CR','Tertile','N','PRD%','PRDN-noise%','HR MAE'],
    ['CR','Tertile',band,'Coherence'],
  ];
  if (data.tables.length !== ids.length) throw Error('Expected five fidelity tables');
  const ratios = data.modality === 'ppg' ? ['02x','04x','08x','16x','32x'] : ['02x','04x','08x','16x','32x','64x'];
  for (const [index,id] of ids.entries()) {
    const matches=data.tables.filter(table=>table.id===id);
    if(matches.length!==1)throw Error(`Missing/duplicate table: ${id}`);
    const table=matches[0], noise=index>=3;
    if(JSON.stringify(table.headers)!==JSON.stringify(headers[index]))throw Error(`Invalid columns: ${id}`);
    const keys=noise?ratios.flatMap(r=>['clean','median','noisy'].map(t=>[r,t])):ratios.map(r=>[r]);
    if(table.rows.length!==keys.length)throw Error(`Incomplete rows: ${id}`);
    table.rows.forEach((row,i)=>{
      if(row.length!==table.headers.length || keys[i].some((key,j)=>row[j]!==key) || row.slice(noise?2:1).some(value=>typeof value!=='string'||!(/^(?:\d+(?:\.\d+)?|-|—)$/.test(value))))throw Error(`Invalid fidelity row: ${id} ${i}`);
    });
  }
  return data;
}
