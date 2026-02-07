import { motion, AnimatePresence } from 'framer-motion';
import { X, Copy, Check, Terminal, Server, Zap, Database } from 'lucide-react';
import { useState } from 'react';
import { API_BASE } from '../services/api';

interface ApiModalProps {
    isOpen: boolean;
    onClose: () => void;
}

const CodeBlock: React.FC<{ code: string; language?: string }> = ({ code, language = 'bash' }) => {
    const [copied, setCopied] = useState(false);

    const handleCopy = () => {
        navigator.clipboard.writeText(code);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <div className="relative group">
            <pre className="bg-dark-1 border border-white/10 rounded-lg p-4 overflow-x-auto text-sm">
                <code className="text-gray-300 font-mono">{code}</code>
            </pre>
            <button
                onClick={handleCopy}
                className="absolute top-2 right-2 p-2 rounded-lg bg-white/5 hover:bg-white/10 transition-colors opacity-0 group-hover:opacity-100"
            >
                {copied ? (
                    <Check className="w-4 h-4 text-primary-green" />
                ) : (
                    <Copy className="w-4 h-4 text-gray-400" />
                )}
            </button>
        </div>
    );
};

export const ApiModal: React.FC<ApiModalProps> = ({ isOpen, onClose }) => {
    return (
        <AnimatePresence>
            {isOpen && (
                <>
                    {/* Backdrop */}
                    <motion.div
                        className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        onClick={onClose}
                    />

                    {/* Modal */}
                    <motion.div
                        className="fixed inset-4 md:inset-10 lg:inset-20 z-50 overflow-hidden"
                        initial={{ opacity: 0, scale: 0.95, y: 20 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.95, y: 20 }}
                        transition={{ duration: 0.2 }}
                    >
                        <div className="h-full glass-card rounded-2xl flex flex-col overflow-hidden border border-white/10">
                            {/* Header */}
                            <div className="flex items-center justify-between p-6 border-b border-white/10">
                                <div className="flex items-center gap-3">
                                    <div className="p-2 rounded-lg bg-primary-blue/20">
                                        <Zap className="w-6 h-6 text-primary-blue" />
                                    </div>
                                    <div>
                                        <h2 className="text-2xl font-heading font-bold text-white">API Reference</h2>
                                        <p className="text-gray-400 text-sm">REST API documentation</p>
                                    </div>
                                </div>
                                <button
                                    onClick={onClose}
                                    className="p-2 rounded-lg hover:bg-white/10 transition-colors"
                                >
                                    <X className="w-6 h-6 text-gray-400" />
                                </button>
                            </div>

                            {/* Content */}
                            <div className="flex-1 overflow-y-auto p-6 space-y-8">
                                {/* Base URL */}
                                <section>
                                    <div className="flex items-center gap-2 mb-3">
                                        <Server className="w-5 h-5 text-primary-blue" />
                                        <h3 className="text-lg font-heading font-semibold text-white">Base URL</h3>
                                    </div>
                                    <div className="bg-dark-1 border border-white/10 rounded-lg px-4 py-3 font-mono text-primary-green">
                                        {API_BASE}
                                    </div>
                                </section>

                                {/* Endpoints */}
                                <section>
                                    <div className="flex items-center gap-2 mb-4">
                                        <Terminal className="w-5 h-5 text-primary-blue" />
                                        <h3 className="text-lg font-heading font-semibold text-white">Endpoints</h3>
                                    </div>

                                    <div className="space-y-4">
                                        {[
                                            { method: 'GET', path: '/health', desc: 'Health check endpoint' },
                                            { method: 'GET', path: '/analyze/demo?type=normal', desc: 'Analyze demo normal sample' },
                                            { method: 'GET', path: '/analyze/demo?type=faulty', desc: 'Analyze demo faulty sample' },
                                            { method: 'POST', path: '/analyze', desc: 'Analyze uploaded file (multipart/form-data)' },
                                        ].map((endpoint) => (
                                            <div key={endpoint.path} className="glass-card p-4 rounded-xl">
                                                <div className="flex items-center gap-3 mb-2">
                                                    <span className={`px-2 py-1 rounded text-xs font-bold ${endpoint.method === 'GET'
                                                            ? 'bg-primary-green/20 text-primary-green'
                                                            : 'bg-primary-blue/20 text-primary-blue'
                                                        }`}>
                                                        {endpoint.method}
                                                    </span>
                                                    <code className="text-white font-mono text-sm">{endpoint.path}</code>
                                                </div>
                                                <p className="text-gray-400 text-sm">{endpoint.desc}</p>
                                            </div>
                                        ))}
                                    </div>
                                </section>

                                {/* Example Requests */}
                                <section>
                                    <h3 className="text-lg font-heading font-semibold text-white mb-4">📝 Example Requests</h3>

                                    <div className="space-y-4">
                                        <div>
                                            <p className="text-gray-400 text-sm mb-2">Demo (Normal Sample):</p>
                                            <CodeBlock code={`curl ${API_BASE}/analyze/demo?type=normal`} />
                                        </div>

                                        <div>
                                            <p className="text-gray-400 text-sm mb-2">Demo (Faulty Sample):</p>
                                            <CodeBlock code={`curl ${API_BASE}/analyze/demo?type=faulty`} />
                                        </div>

                                        <div>
                                            <p className="text-gray-400 text-sm mb-2">File Upload:</p>
                                            <CodeBlock code={`curl -X POST -F "file=@machine_audio.wav" ${API_BASE}/analyze`} />
                                        </div>
                                    </div>
                                </section>

                                {/* Example Response */}
                                <section>
                                    <h3 className="text-lg font-heading font-semibold text-white mb-4">📦 Example Response</h3>
                                    <CodeBlock
                                        language="json"
                                        code={`{
  "status": "faulty",
  "health_score": 52,
  "anomaly_score": 0.1847,
  "failure_type": "bearing_fault",
  "confidence": 0.89,
  "explanation": "Anomaly detected in 4/5 windows. Impulsive vibration patterns indicate potential bearing wear.",
  "reasoning_data": {
    "windows_analyzed": 5,
    "anomalous_windows": 4,
    "threshold": 0.043,
    "rf_confidence": 0.89
  },
  "processing_ms": 1247
}`}
                                    />
                                </section>

                                {/* Dataset as API */}
                                <section className="glass-card p-6 rounded-xl border border-primary-blue/30">
                                    <div className="flex items-center gap-2 mb-3">
                                        <Database className="w-5 h-5 text-primary-blue" />
                                        <h3 className="text-lg font-heading font-semibold text-white">🔗 Dataset as API</h3>
                                    </div>
                                    <div className="space-y-3 text-gray-300 text-sm">
                                        <p>
                                            This API transforms raw audio/vibration data into structured health assessments,
                                            enabling integration with external systems.
                                        </p>
                                        <div className="space-y-2">
                                            <p><span className="text-primary-blue">→</span> Upload labeled files to get standardized JSON results</p>
                                            <p><span className="text-primary-blue">→</span> Batch process multiple files for fleet monitoring</p>
                                            <p><span className="text-primary-blue">→</span> Integrate with dashboards, alerting systems, or databases</p>
                                            <p><span className="text-primary-blue">→</span> Export predictions for ML pipeline training</p>
                                        </div>
                                    </div>
                                </section>

                                {/* Rate Limits */}
                                <section className="text-center text-gray-500 text-sm py-4">
                                    <p>&#9889; No rate limits for demo &#x2022; &#128225; REST API &#x2022; &#128275; No authentication required</p>
                                </section>
                            </div>
                        </div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
};

export default ApiModal;
