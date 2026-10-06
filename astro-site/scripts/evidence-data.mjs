export const tableIds = ['ppg-quality', 'ecg-quality', 'ppg-files', 'ecg-files', 'ppg-detail', 'ecg-detail'];
const headers = {
  quality: ['CR', 'N', 'Truth PRD (%)', 'Faithful PRD (%)', 'HR error (bpm)'],
  files: ['CR', 'Total', 'Encoder', 'Decoder', 'Codebook'],
  detail: ['CR', 'PRDN-noise (%)', 'Band error', 'Coherence', 'Seam ratio'],
};
export function validateEvidence(data) {
  if (data.schemaVersion !== 1 || !['preserved-documentation', 'scorecards'].includes(data.provenance?.kind)) throw Error('Unsupported evidence schema or provenance');
  if (!data.provenance.source || !/^[a-f0-9]{64}$/.test(data.provenance.sha256 ?? '')) throw Error('Evidence requires a source and SHA-256 provenance');
  if (!Array.isArray(data.tables) || data.tables.length !== tableIds.length) throw Error('Expected six evidence tables');
  for (const id of tableIds) {
    const matches = data.tables.filter(table => table.id === id);
    if (matches.length !== 1) throw Error(`Missing or duplicate evidence table: ${id}`);
    const table = matches[0];
    const ratios = id.startsWith('ppg') ? ['2x','4x','8x','16x','32x'] : ['2x','4x','8x','16x','32x','64x'];
    const kind = id.split('-')[1];
    if (JSON.stringify(table.headers) !== JSON.stringify(headers[kind])) throw Error(`Invalid headers: ${id}`);
    if (table.rows.length !== ratios.length) throw Error(`Incomplete evidence rows: ${id}`);
    table.rows.forEach((row, i) => {
      const numeric = kind === 'files' ? /^\d+(?:\.\d+)? KiB$/ : /^\d+(?:\.\d+)?$/;
      if (row.length !== 5 || row[0] !== ratios[i] || row.slice(1).some(x => typeof x !== 'string' || (x !== '-' && !numeric.test(x)))) throw Error(`Invalid evidence row: ${id} ${i}`);
    });
  }
  return data;
}
