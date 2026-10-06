import api from './data/api-sidebar.json' with { type: 'json' };

const page = (label, slug) => ({ label, slug });
const group = (label, items, collapsed = true) => ({ label, items, collapsed });
const experimentFamily = (signal, family, ratios, prior = false) => group(
  `${signal.toUpperCase()} ${prior ? 'two-stage' : { rvq: 'codecs', spiht: 'SPIHT', hybrid: 'hybrid' }[family]}`,
  ratios.map(ratio => page(`${ratio}× compression`, `experiments/${signal}-${family}-${ratio}x${prior ? '-prior' : ''}`)),
);

export const sections = [
  { label: 'Home', href: '/compressionkit/', sidebar: false },
  {
    label: 'Getting started', href: '/compressionkit/getting-started/',
    sidebar: [
      page('Use cases', 'use-cases'),
      page('Customer evidence', 'customer-evidence'),
      page('Getting started', 'getting-started'),
      page('Dataset setup', 'datasets'),
      page('Example notebooks', 'examples'),
      group('Notebook guides', [
        page('PPG codec quickstart', 'guides/01_quickstart_golden_codec'),
        page('Evaluate your recordings', 'guides/02_evaluate_on_your_data'),
        page('ECG codec quickstart', 'guides/03_quickstart_golden_codec_ecg'),
      ]),
      page('HuggingFace bundles', 'huggingface'),
      page('Signal compression demo', 'demo/ppg-codec'),
    ],
  },
  {
    label: 'User guide', href: '/compressionkit/signals/',
    sidebar: [
      group('Signals', [
        page('Signal overview', 'signals'),
        page('PPG', 'signals/ppg'),
        page('PPG workflow', 'signals/ppg-workflow'),
        page('ECG', 'signals/ecg'),
      ], false),
      group('Methods', [
        page('Method comparison', 'methods'),
        page('RVQ autoencoder', 'methods/rvq'),
        page('Wavelet and SPIHT', 'methods/spiht'),
        page('Hybrid compression', 'methods/hybrid'),
        page('PPG compression vs fidelity', 'methods/cr_vs_fidelity_ppg'),
        page('ECG compression vs fidelity', 'methods/cr_vs_fidelity_ecg'),
      ], false),
      page('Deployment', 'deployment'),
      page('Experiment architecture', 'experiment-architecture'),
      page('Adding a codec family', 'adding-a-codec-family'),
    ],
  },
  {
    label: 'Models & experiments', href: '/compressionkit/models/',
    sidebar: [
      group('Models', [
        page('Model zoo', 'models'),
        page('PPG models', 'models/ppg'),
        page('ECG models', 'models/ecg'),
      ], false),
      page('Experiment overview', 'experiments'),
      ...['ppg', 'ecg'].flatMap(signal => {
        const ratios = signal === 'ppg' ? [2, 4, 8, 16, 32] : [2, 4, 8, 16, 32, 64];
        return [
          experimentFamily(signal, 'rvq', ratios),
          experimentFamily(signal, 'rvq', [4, 8], true),
          experimentFamily(signal, 'spiht', ratios),
          experimentFamily(signal, 'hybrid', ratios),
        ];
      }),
    ],
  },
  {
    label: 'Reference', href: '/compressionkit/reference/',
    sidebar: [
      page('Python API', 'reference'),
      page('Command line', 'cli'),
      page('Artifact contract', 'release-contract'),
      page('Validation scorecard', 'validation-scorecard'),
      group('API topics', [
        page('Configuration', 'api/configs'), page('Datasets', 'api/datasets'),
        page('Preprocessing', 'api/preprocessing'), page('Models', 'api/models'),
        page('Evaluation', 'api/evaluation'), page('Export', 'api/export'),
      ]),
      group('API modules', api),
    ],
  },
];
const flatten = items => items.flatMap(item => item.items ? flatten(item.items) : [item]);
export const sectionByPath = Object.fromEntries(sections.flatMap(section =>
  section.sidebar === false ? [[section.href, section.href]] :
    flatten(section.sidebar).map(item => [`/compressionkit/${item.slug}/`, section.href]),
));
