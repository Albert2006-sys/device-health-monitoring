import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Download,
  Cpu,
  FileCode2,
  FileJson,
  FileArchive,
  ChevronDown,
  ChevronUp,
  Eye,
  Package,
} from 'lucide-react';

/* ────────────────────────────────────────────────
   Deployment artifact metadata
   ──────────────────────────────────────────────── */

const GITHUB_RAW_BASE =
  'https://raw.githubusercontent.com/Albert2006-sys/device-health-monitoring/DownloadableForDevice';

/** Model/data artifacts served from /public/deployment (Vercel-hosted) */
const LOCAL_BASE = '/deployment';

interface Artifact {
  filename: string;
  description: string;
  icon: React.ReactNode;
  /** URL to download the file from */
  url: string;
  /** If true, show a "Preview" toggle with inline code block */
  previewable?: boolean;
  sizeHint?: string;
}

const ARTIFACTS: Artifact[] = [
  {
    filename: 'engine_classifier.tflite',
    description: 'Quantized TFLite classifier model for fault identification (6 classes).',
    icon: <Cpu size={18} className="text-primary-blue" />,
    url: `${LOCAL_BASE}/engine_classifier.tflite`,
    sizeHint: '~384 KB',
  },
  {
    filename: 'engine_anomaly.tflite',
    description: 'Quantized TFLite autoencoder for anomaly detection (reconstruction error).',
    icon: <Cpu size={18} className="text-primary-green" />,
    url: `${LOCAL_BASE}/engine_anomaly.tflite`,
    sizeHint: '~18 KB',
  },
  {
    filename: 'scaler_mean.npy',
    description: 'Feature normalization mean values (13 features).',
    icon: <FileCode2 size={18} className="text-yellow-400" />,
    url: `${LOCAL_BASE}/scaler_mean.npy`,
    sizeHint: '~0.5 KB',
  },
  {
    filename: 'scaler_scale.npy',
    description: 'Feature normalization scale (std-dev) values (13 features).',
    icon: <FileCode2 size={18} className="text-yellow-400" />,
    url: `${LOCAL_BASE}/scaler_scale.npy`,
    sizeHint: '~0.5 KB',
  },
  {
    filename: 'anomaly_threshold.npy',
    description: 'Statistically derived anomaly threshold for the autoencoder.',
    icon: <FileCode2 size={18} className="text-orange-400" />,
    url: `${LOCAL_BASE}/anomaly_threshold.npy`,
    sizeHint: '~0.1 KB',
  },
  {
    filename: 'label_map.json',
    description: 'Class index to human-readable fault type mapping.',
    icon: <FileJson size={18} className="text-purple-400" />,
    url: `${LOCAL_BASE}/label_map.json`,
    previewable: true,
    sizeHint: '~1 KB',
  },
  {
    filename: 'constants.h',
    description: 'Embedded C header with inference constants for ESP32 firmware.',
    icon: <FileCode2 size={18} className="text-cyan-400" />,
    url: `${GITHUB_RAW_BASE}/constants.h`,
    previewable: true,
    sizeHint: '~2 KB',
  },
  {
    filename: 'esp32_engine_monitor.ino',
    description: 'Full ESP32 firmware: Dual-Trust inference + MQTT telemetry publishing.',
    icon: <FileCode2 size={18} className="text-emerald-400" />,
    url: `${GITHUB_RAW_BASE}/esp32_engine_monitor.ino`,
    previewable: true,
    sizeHint: '~8 KB',
  },
];

/* ────────────────────────────────────────────────
   Individual artifact row
   ──────────────────────────────────────────────── */

const ArtifactRow: React.FC<{ artifact: Artifact; index: number }> = ({
  artifact,
  index,
}) => {
  const [showPreview, setShowPreview] = useState(false);
  const [previewContent, setPreviewContent] = useState<string | null>(null);
  const [loadingPreview, setLoadingPreview] = useState(false);

  const handlePreview = async () => {
    if (showPreview) {
      setShowPreview(false);
      return;
    }
    if (previewContent) {
      setShowPreview(true);
      return;
    }
    setLoadingPreview(true);
    try {
      const res = await fetch(artifact.url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const text = await res.text();
      setPreviewContent(text.slice(0, 8000)); // cap at 8 KB of text
    } catch {
      setPreviewContent('// Unable to load preview. Download the file instead.');
    } finally {
      setLoadingPreview(false);
      setShowPreview(true);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      className="bg-dark-1 rounded-lg border border-white/5 overflow-hidden"
    >
      <div className="flex items-center gap-4 p-4">
        {/* Icon */}
        <div className="p-2.5 rounded-lg bg-white/5 shrink-0">{artifact.icon}</div>

        {/* Info */}
        <div className="flex-1 min-w-0">
          <p className="font-mono text-sm text-white truncate">{artifact.filename}</p>
          <p className="text-xs text-gray-400 mt-0.5">{artifact.description}</p>
          {artifact.sizeHint && (
            <p className="text-xs text-gray-600 mt-0.5">{artifact.sizeHint}</p>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2 shrink-0">
          {artifact.previewable && (
            <button
              onClick={handlePreview}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs
                         text-gray-400 hover:text-white hover:bg-white/5 transition-all
                         border border-white/5"
            >
              <Eye size={14} />
              {showPreview ? 'Hide' : 'Preview'}
              {showPreview ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            </button>
          )}
          <a
            href={artifact.url}
            download={artifact.filename}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs
                       font-semibold bg-primary-blue/20 text-primary-blue
                       hover:bg-primary-blue/30 transition-all border border-primary-blue/30"
          >
            <Download size={14} />
            Download
          </a>
        </div>
      </div>

      {/* Code Preview */}
      <AnimatePresence>
        {showPreview && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="border-t border-white/5"
          >
            <pre className="p-4 text-xs font-mono text-gray-300 overflow-x-auto max-h-80 overflow-y-auto bg-dark-1/80">
              {loadingPreview ? 'Loading...' : previewContent}
            </pre>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

/* ────────────────────────────────────────────────
   Main component
   ──────────────────────────────────────────────── */

export const DeployToDevice: React.FC = () => {
  const handleDownloadAll = () => {
    // Trigger sequential downloads (browser will bundle or prompt)
    ARTIFACTS.forEach((a, i) => {
      setTimeout(() => {
        const link = document.createElement('a');
        link.href = a.url;
        link.download = a.filename;
        link.rel = 'noopener noreferrer';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }, i * 400); // stagger to avoid browser blocking
    });
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="max-w-4xl mx-auto"
    >
      {/* Header */}
      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card mb-4">
          <Package className="w-4 h-4 text-primary-green" />
          <span className="text-sm text-gray-300">Edge Deployment</span>
        </div>
        <h2 className="text-3xl md:text-4xl font-heading font-bold mb-3">
          <span className="text-gradient-blue">Deploy to Device</span>
        </h2>
        <p className="text-gray-400 max-w-xl mx-auto">
          Download the deployment artifacts required to run the Dual-Trust AI
          engine health monitor on an ESP32 microcontroller.
        </p>
      </div>

      {/* Download All */}
      <div className="flex justify-end mb-4">
        <button
          onClick={handleDownloadAll}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold
                     bg-primary-green/20 text-primary-green border border-primary-green/40
                     hover:bg-primary-green/30 transition-all"
        >
          <FileArchive size={18} />
          Download All ({ARTIFACTS.length} files)
        </button>
      </div>

      {/* Artifact List */}
      <div className="space-y-3">
        {ARTIFACTS.map((a, i) => (
          <ArtifactRow key={a.filename} artifact={a} index={i} />
        ))}
      </div>

      {/* Footer note */}
      <p className="text-xs text-gray-600 mt-6 text-center italic">
        Artifacts are served from the{' '}
        <a
          href="https://github.com/Albert2006-sys/device-health-monitoring/tree/DownloadableForDevice"
          target="_blank"
          rel="noopener noreferrer"
          className="text-primary-blue hover:underline"
        >
          DownloadableForDevice
        </a>{' '}
        branch. Flash the .ino firmware via Arduino IDE with ESP32 board support.
      </p>
    </motion.div>
  );
};

export default DeployToDevice;
