import { motion, AnimatePresence } from 'framer-motion';
import { X, FileAudio, Cpu, Layers, Search, Target, MessageSquare, Clock, CheckCircle, AlertTriangle, Trash2, ExternalLink } from 'lucide-react';
import { useState, useEffect } from 'react';
import { getHistory, clearHistory, formatTimestamp, AnalysisHistoryItem } from '../utils/analysisHistory';
import { AnalysisResult } from '../types/analysis';

interface DocsModalProps {
    isOpen: boolean;
    onClose: () => void;
    onLoadResult?: (result: AnalysisResult, fileName: string) => void;
}

export const DocsModal: React.FC<DocsModalProps> = ({ isOpen, onClose, onLoadResult }) => {
    const [history, setHistory] = useState<AnalysisHistoryItem[]>([]);
    const [activeTab, setActiveTab] = useState<'overview' | 'how' | 'history'>('overview');

    useEffect(() => {
        if (isOpen) {
            setHistory(getHistory());
        }
    }, [isOpen]);

    const handleClearHistory = () => {
        clearHistory();
        setHistory([]);
    };

    const handleLoadItem = (item: AnalysisHistoryItem) => {
        if (onLoadResult) {
            onLoadResult(item.result, item.fileName);
            onClose();
        }
    };

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
                                <div>
                                    <h2 className="text-2xl font-heading font-bold text-white">Documentation</h2>
                                    <p className="text-gray-400 text-sm">Device Health Monitoring System</p>
                                </div>
                                <button
                                    onClick={onClose}
                                    className="p-2 rounded-lg hover:bg-white/10 transition-colors"
                                >
                                    <X className="w-6 h-6 text-gray-400" />
                                </button>
                            </div>

                            {/* Tabs */}
                            <div className="flex gap-2 px-6 pt-4">
                                {[
                                    { id: 'overview', label: 'Overview' },
                                    { id: 'how', label: 'How It Works' },
                                    { id: 'history', label: 'Recent Analyses' },
                                ].map((tab) => (
                                    <button
                                        key={tab.id}
                                        onClick={() => setActiveTab(tab.id as typeof activeTab)}
                                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === tab.id
                                            ? 'bg-primary-blue/20 text-primary-blue'
                                            : 'text-gray-400 hover:text-white hover:bg-white/5'
                                            }`}
                                    >
                                        {tab.label}
                                    </button>
                                ))}
                            </div>

                            {/* Content */}
                            <div className="flex-1 overflow-y-auto p-6">
                                {activeTab === 'overview' && (
                                    <div className="space-y-6">
                                        <section>
                                            <h3 className="text-lg font-heading font-semibold text-white mb-3">
                                                🎯 Project Overview
                                            </h3>
                                            <p className="text-gray-300 leading-relaxed">
                                                The Device Health Monitoring System is an AI-powered predictive maintenance solution
                                                that analyzes audio and vibration signals to detect machine faults before they cause failures.
                                            </p>
                                            <div className="mt-3 p-3 bg-primary-blue/10 border border-primary-blue/30 rounded-lg">
                                                <p className="text-sm text-primary-blue font-medium">
                                                    🔬 Physics-Informed Machine Learning (PIML)
                                                </p>
                                                <p className="text-xs text-gray-300 mt-1">
                                                    This system combines data-driven ML models with mechanical validation to improve
                                                    trust and explainability. Predictions are verified against known mechanical behavior.
                                                </p>
                                            </div>
                                        </section>

                                        <section className="glass-card p-4 rounded-xl">
                                            <h4 className="text-primary-blue font-medium mb-3">📁 Supported Input Formats</h4>
                                            <div className="flex flex-wrap gap-2">
                                                {['.wav', '.mat', '.mp3', '.mp4', '.m4a', '.flac'].map((fmt) => (
                                                    <span key={fmt} className="px-3 py-1 bg-primary-blue/10 text-primary-blue rounded-full text-sm font-mono">
                                                        {fmt}
                                                    </span>
                                                ))}
                                            </div>
                                        </section>

                                        <section className="glass-card p-4 rounded-xl">
                                            <h4 className="text-primary-green font-medium mb-3">⚙️ Fault Classes Detected (6)</h4>
                                            <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                                                {[
                                                    { name: 'Healthy', color: 'text-primary-green' },
                                                    { name: 'Bearing Fault', color: 'text-primary-red' },
                                                    { name: 'Bad Ignition', color: 'text-primary-red' },
                                                    { name: 'Dead Battery', color: 'text-primary-red' },
                                                    { name: 'Worn Brakes', color: 'text-primary-red' },
                                                    { name: 'Mixed Faults', color: 'text-yellow-400' },
                                                ].map((fault) => (
                                                    <span key={fault.name} className={`${fault.color} text-sm`}>
                                                        • {fault.name}
                                                    </span>
                                                ))}
                                            </div>
                                        </section>

                                        <section className="glass-card p-4 rounded-xl">
                                            <h4 className="text-yellow-400 font-medium mb-3">🧠 Model Architecture</h4>
                                            <div className="space-y-2 text-sm text-gray-300">
                                                <p><span className="text-primary-blue">Autoencoder:</span> Anomaly detection via reconstruction error</p>
                                                <p><span className="text-primary-green">Random Forest:</span> Fault classification (93% accuracy)</p>
                                                <p><span className="text-gray-400">Training:</span> 1,431 samples across 6 fault types</p>
                                                <p><span className="text-gray-400">Features:</span> 13 sensor-agnostic signal features</p>
                                            </div>
                                        </section>

                                        <section className="glass-card p-4 rounded-xl border border-yellow-500/30">
                                            <h4 className="text-yellow-400 font-medium mb-3">🛡️ How the System Handles Unknown Audio</h4>
                                            <div className="space-y-3 text-sm text-gray-300">
                                                <p className="leading-relaxed">
                                                    <strong className="text-white">Our system is designed for machine health monitoring, not general audio classification.</strong>
                                                </p>
                                                <p className="leading-relaxed">
                                                    If a completely unseen sound is uploaded, the anomaly detector flags it and the confidence drops, preventing false diagnosis.
                                                </p>
                                                <div className="bg-dark-1 p-3 rounded-lg border border-white/5">
                                                    <p className="text-xs text-gray-400 mb-2">When unknown audio is detected:</p>
                                                    <ul className="space-y-1 text-xs">
                                                        <li className="flex items-center gap-2">
                                                            <span className="text-primary-red">●</span>
                                                            <span>Status marked as "Faulty" (anomalous)</span>
                                                        </li>
                                                        <li className="flex items-center gap-2">
                                                            <span className="text-yellow-400">●</span>
                                                            <span>No specific fault type assigned</span>
                                                        </li>
                                                        <li className="flex items-center gap-2">
                                                            <span className="text-yellow-400">●</span>
                                                            <span>Confidence intentionally lowered</span>
                                                        </li>
                                                        <li className="flex items-center gap-2">
                                                            <span className="text-primary-green">●</span>
                                                            <span>Clear warning shown to user</span>
                                                        </li>
                                                    </ul>
                                                </div>
                                                <p className="text-xs text-gray-500 italic">
                                                    This ensures safety and prevents false alarms when non-machine audio is analyzed.
                                                </p>
                                            </div>
                                        </section>
                                    </div>
                                )}

                                {activeTab === 'how' && (
                                    <div className="space-y-4">
                                        <h3 className="text-lg font-heading font-semibold text-white mb-4">
                                            🔄 Processing Pipeline
                                        </h3>

                                        {[
                                            { icon: FileAudio, title: 'File Upload', desc: 'User uploads audio/video file or triggers demo sample', color: 'text-primary-blue' },
                                            { icon: Layers, title: 'Signal Preprocessing', desc: 'Split signal into 1-second windows for consistent analysis', color: 'text-purple-400' },
                                            { icon: Cpu, title: 'Feature Extraction', desc: 'Extract 13 statistical & spectral features per window', color: 'text-yellow-400' },
                                            { icon: Search, title: 'Anomaly Detection', desc: 'Autoencoder computes reconstruction error vs threshold', color: 'text-orange-400' },
                                            { icon: Target, title: 'Fault Classification', desc: 'Random Forest predicts fault type with confidence score', color: 'text-primary-red' },
                                            { icon: MessageSquare, title: 'Explanation Generation', desc: 'AI reasoning explains the decision in plain language', color: 'text-primary-green' },
                                        ].map((step, index) => (
                                            <motion.div
                                                key={step.title}
                                                className="flex items-start gap-4 glass-card p-4 rounded-xl"
                                                initial={{ opacity: 0, x: -20 }}
                                                animate={{ opacity: 1, x: 0 }}
                                                transition={{ delay: index * 0.1 }}
                                            >
                                                <div className={`p-3 rounded-xl bg-white/5 ${step.color}`}>
                                                    <step.icon className="w-5 h-5" />
                                                </div>
                                                <div className="flex-1">
                                                    <div className="flex items-center gap-2">
                                                        <span className="text-gray-500 text-sm font-mono">0{index + 1}</span>
                                                        <h4 className="font-medium text-white">{step.title}</h4>
                                                    </div>
                                                    <p className="text-gray-400 text-sm mt-1">{step.desc}</p>
                                                </div>
                                            </motion.div>
                                        ))}
                                    </div>
                                )}

                                {activeTab === 'history' && (
                                    <div className="space-y-4">
                                        <div className="flex items-center justify-between">
                                            <h3 className="text-lg font-heading font-semibold text-white">
                                                📊 Recent Analyses
                                            </h3>
                                            {history.length > 0 && (
                                                <button
                                                    onClick={handleClearHistory}
                                                    className="flex items-center gap-2 px-3 py-2 text-sm text-gray-400 hover:text-primary-red hover:bg-primary-red/10 rounded-lg transition-all"
                                                >
                                                    <Trash2 className="w-4 h-4" />
                                                    Clear
                                                </button>
                                            )}
                                        </div>

                                        {onLoadResult && history.length > 0 && (
                                            <p className="text-gray-500 text-sm">
                                                💡 Click any item to load its full analysis results
                                            </p>
                                        )}

                                        {history.length === 0 ? (
                                            <div className="text-center py-12 text-gray-500">
                                                <Clock className="w-12 h-12 mx-auto mb-3 opacity-50" />
                                                <p>No analyses yet</p>
                                                <p className="text-sm mt-1">Run a demo or upload a file to see history</p>
                                            </div>
                                        ) : (
                                            <div className="space-y-3">
                                                {history.map((item) => {
                                                    // Handle both old format (no result) and new format
                                                    const status = item.result?.status ?? (item as any).status ?? 'normal';
                                                    const healthScore = item.result?.health_score ?? (item as any).health_score ?? 0;
                                                    const failureType = item.result?.failure_type ?? (item as any).failure_type ?? null;
                                                    const hasFullResult = !!item.result;

                                                    return (
                                                        <motion.button
                                                            key={item.id}
                                                            onClick={() => hasFullResult && handleLoadItem(item)}
                                                            className={`w-full glass-card p-4 rounded-xl flex items-center gap-4 transition-all group text-left ${hasFullResult ? 'hover:bg-white/5 cursor-pointer' : 'opacity-60 cursor-not-allowed'
                                                                }`}
                                                            initial={{ opacity: 0, y: 10 }}
                                                            animate={{ opacity: 1, y: 0 }}
                                                        >
                                                            <div className={`p-2 rounded-lg ${status === 'normal'
                                                                ? 'bg-primary-green/20 text-primary-green'
                                                                : 'bg-primary-red/20 text-primary-red'
                                                                }`}>
                                                                {status === 'normal' ? (
                                                                    <CheckCircle className="w-5 h-5" />
                                                                ) : (
                                                                    <AlertTriangle className="w-5 h-5" />
                                                                )}
                                                            </div>

                                                            <div className="flex-1 min-w-0">
                                                                <p className="text-white font-medium truncate">{item.fileName}</p>
                                                                <p className="text-gray-500 text-sm">{formatTimestamp(item.timestamp)}</p>
                                                            </div>

                                                            <div className="text-right">
                                                                <p className={`font-mono font-bold ${healthScore >= 80 ? 'text-primary-green' :
                                                                    healthScore >= 60 ? 'text-yellow-400' : 'text-primary-red'
                                                                    }`}>
                                                                    {healthScore}
                                                                </p>
                                                                <p className="text-gray-500 text-xs uppercase">
                                                                    {failureType?.replace(/_/g, ' ') || 'Healthy'}
                                                                </p>
                                                            </div>

                                                            {hasFullResult && (
                                                                <ExternalLink className="w-4 h-4 text-gray-600 group-hover:text-primary-blue transition-colors" />
                                                            )}
                                                        </motion.button>
                                                    );
                                                })}
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
};

export default DocsModal;
