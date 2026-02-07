import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Toaster } from 'react-hot-toast';
import { ArrowDown, Sparkles, FileText, Layers, Mic2 } from 'lucide-react';

import { useAnalysis } from './hooks/useAnalysis';
import { Navbar } from './components/Navbar';
import { HealthGauge } from './components/HealthGauge';
import { StatusBadge } from './components/StatusBadge';
import { MetricsDashboard } from './components/MetricsDashboard';
import { AIReasoning } from './components/AIReasoning';
import { FileUpload } from './components/FileUpload';
import { DemoButtons } from './components/DemoButtons';
import { DocsModal } from './components/DocsModal';
import { ApiModal } from './components/ApiModal';
import { PhysicsValidation } from './components/PhysicsValidation';
import { FailureFingerprint } from './components/FailureFingerprint';
import { MaintenanceAdvice } from './components/MaintenanceAdvice';
import { WaveformVisualization } from './components/WaveformVisualization';
import { FailurePrediction } from './components/FailurePrediction';
import { LiveRecording } from './components/LiveRecording';
import { BatchAnalysis } from './components/BatchAnalysis';
import { generatePDFReport } from './utils/reportGenerator';
import { AnalysisResult } from './types/analysis';

function App() {
  const { result, loading, analyze, runDemo, reset, setResult } = useAnalysis();
  const [loadingType, setLoadingType] = useState<'normal' | 'faulty' | null>(null);
  const [analysisMode, setAnalysisMode] = useState<'single' | 'batch'>('single');
  const [uploadedFilename, setUploadedFilename] = useState<string>('');

  // Modal states
  const [docsOpen, setDocsOpen] = useState(false);
  const [apiOpen, setApiOpen] = useState(false);

  const handleNormalDemo = async () => {
    setLoadingType('normal');
    setUploadedFilename('demo_normal.wav');
    await runDemo('normal');
    setLoadingType(null);
  };

  const handleFaultyDemo = async () => {
    setLoadingType('faulty');
    setUploadedFilename('demo_faulty.wav');
    await runDemo('faulty');
    setLoadingType(null);
  };

  const handleFileUpload = async (file: File) => {
    setUploadedFilename(file.name);
    await analyze(file);
  };

  const handleLiveResult = (liveResult: AnalysisResult) => {
    setUploadedFilename(`live_recording_${Date.now()}.webm`);
    setResult(liveResult);
  };

  const handleExportPDF = () => {
    if (result) {
      generatePDFReport(result, uploadedFilename);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-dark-1 via-dark-2 to-dark-1 bg-grid">
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: '#1A1F3A',
            color: '#fff',
            border: '1px solid rgba(255, 255, 255, 0.1)',
          },
        }}
      />

      <Navbar
        onDocsClick={() => setDocsOpen(true)}
        onApiClick={() => setApiOpen(true)}
      />

      {/* Modals */}
      <DocsModal
        isOpen={docsOpen}
        onClose={() => setDocsOpen(false)}
        onLoadResult={(loadedResult) => setResult(loadedResult)}
      />
      <ApiModal isOpen={apiOpen} onClose={() => setApiOpen(false)} />

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-24 pb-16">
        {/* Mode Tabs */}
        <motion.div
          className="flex justify-center gap-4 mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <button
            onClick={() => { setAnalysisMode('single'); reset(); }}
            className={`flex items-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all ${analysisMode === 'single'
              ? 'bg-primary-blue/20 text-primary-blue border border-primary-blue'
              : 'glass-card text-gray-400 hover:text-white'
              }`}
          >
            <FileText size={20} />
            Single Analysis
          </button>
          <button
            onClick={() => { setAnalysisMode('batch'); reset(); }}
            className={`flex items-center gap-2 px-6 py-3 rounded-xl font-semibold transition-all ${analysisMode === 'batch'
              ? 'bg-primary-blue/20 text-primary-blue border border-primary-blue'
              : 'glass-card text-gray-400 hover:text-white'
              }`}
          >
            <Layers size={20} />
            Batch Analysis
          </button>
        </motion.div>

        <AnimatePresence mode="wait">
          {analysisMode === 'batch' ? (
            <motion.div
              key="batch"
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -50 }}
            >
              <BatchAnalysis />
            </motion.div>
          ) : (
            <motion.div
              key="single"
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 50 }}
            >
              {/* Hero Section */}
              <motion.section
                className="text-center py-12 md:py-16"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
              >
                <motion.div
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card mb-6"
                  initial={{ scale: 0.9 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.2 }}
                >
                  <Sparkles className="w-4 h-4 text-primary-blue" />
                  <span className="text-sm text-gray-300">AI-Powered Predictive Maintenance</span>
                </motion.div>

                <h1 className="text-4xl md:text-6xl font-heading font-bold mb-4">
                  <span className="text-gradient-blue">Device Health</span>
                  <br />
                  <span className="text-white">Monitoring System</span>
                </h1>

                <p className="text-lg text-gray-400 max-w-2xl mx-auto mb-8">
                  Detect machine faults before they become failures. Analyze audio and vibration
                  signals with our unified AI model trained on 1,400+ samples across 6 fault types.
                </p>

                {/* Demo Buttons */}
                <DemoButtons
                  onNormalClick={handleNormalDemo}
                  onFaultyClick={handleFaultyDemo}
                  loading={loading}
                  loadingType={loadingType}
                />

                {/* Divider */}
                <div className="flex items-center gap-4 my-8 max-w-md mx-auto">
                  <div className="flex-1 h-px bg-gradient-to-r from-transparent to-gray-600" />
                  <span className="text-gray-500 text-sm">or upload your own</span>
                  <div className="flex-1 h-px bg-gradient-to-l from-transparent to-gray-600" />
                </div>

                {/* File Upload */}
                <FileUpload
                  onUpload={handleFileUpload}
                  loading={loading && loadingType === null}
                  disabled={loading}
                />

                {/* Live Recording Section */}
                {!result && !loading && (
                  <div className="mt-12">
                    <div className="flex items-center gap-4 my-8 max-w-md mx-auto">
                      <div className="flex-1 h-px bg-gradient-to-r from-transparent to-gray-600" />
                      <span className="text-gray-500 text-sm flex items-center gap-2">
                        <Mic2 size={16} />
                        or record live
                      </span>
                      <div className="flex-1 h-px bg-gradient-to-l from-transparent to-gray-600" />
                    </div>
                    <LiveRecording onResult={handleLiveResult} />
                  </div>
                )}
              </motion.section>

              {/* Results Section */}
              <AnimatePresence>
                {result && (
                  <motion.section
                    className="space-y-8"
                    initial={{ opacity: 0, y: 40 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -40 }}
                    transition={{ duration: 0.5 }}
                  >
                    {/* Scroll indicator */}
                    <motion.div
                      className="flex justify-center -mt-4"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1, y: [0, 10, 0] }}
                      transition={{
                        opacity: { delay: 0.5 },
                        y: { duration: 1.5, repeat: Infinity },
                      }}
                    >
                      <ArrowDown className="w-6 h-6 text-primary-blue" />
                    </motion.div>

                    {/* Main Results Grid */}
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                      {/* Health Gauge */}
                      <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.1 }}
                      >
                        <HealthGauge score={result.health_score} />
                      </motion.div>

                      {/* Status & Badge */}
                      <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.2 }}
                      >
                        <StatusBadge status={result.status} failureType={result.failure_type} confidence={result.confidence} />
                      </motion.div>

                      {/* Metrics */}
                      <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.3 }}
                      >
                        <MetricsDashboard data={result} />
                      </motion.div>
                    </div>

                    {/* Physics Validation */}
                    {result.physics_validation && (
                      <PhysicsValidation validation={result.physics_validation} />
                    )}

                    {/* Waveform Visualization - NEW FEATURE 1 */}
                    {result.window_results && result.window_results.length > 0 && (
                      <WaveformVisualization
                        windowResults={result.window_results}
                        threshold={result.reasoning_data?.threshold || 0.05}
                      />
                    )}

                    {/* Failure Prediction - NEW FEATURE 2 */}
                    <FailurePrediction
                      currentHealth={result.health_score}
                      anomalyScore={result.anomaly_score}
                      threshold={result.reasoning_data?.threshold || 0.05}
                      failureType={result.failure_type}
                    />

                    {/* Explanation Card */}
                    <motion.div
                      className="glass-card p-6 rounded-xl border border-white/10"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.5 }}
                    >
                      <p className={`text-lg ${result.status === 'normal' ? 'text-primary-green' : 'text-primary-red'
                        }`}>
                        {result.explanation}
                      </p>
                    </motion.div>

                    {/* Fingerprint and Maintenance Grid */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                      <FailureFingerprint fingerprint={result.failure_fingerprint} />
                      <MaintenanceAdvice advice={result.maintenance_advice} />
                    </div>

                    {/* AI Reasoning */}
                    <AIReasoning data={result} />

                    {/* Action Buttons */}
                    <motion.div
                      className="flex flex-col sm:flex-row justify-center items-center gap-4 pt-4"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: 0.7 }}
                    >
                      <button
                        onClick={handleExportPDF}
                        className="flex items-center gap-2 px-6 py-3 rounded-xl bg-primary-blue/20 border border-primary-blue text-primary-blue hover:bg-primary-blue/30 transition-all font-semibold"
                      >
                        <FileText size={18} />
                        📄 Export PDF Report
                      </button>
                      <button
                        onClick={reset}
                        className="px-6 py-3 rounded-xl glass-card text-gray-400 hover:text-white hover:bg-white/10 transition-all font-medium"
                      >
                        ← Analyze Another File
                      </button>
                    </motion.div>
                  </motion.section>
                )}
              </AnimatePresence>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Footer Stats */}
        <motion.footer
          className="text-center pt-16 pb-8 border-t border-white/5 mt-16"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
        >
          <div className="flex flex-wrap justify-center gap-8 text-gray-500 text-sm font-mono">
            <div>
              <span className="text-primary-blue">1,431</span> Training Samples
            </div>
            <div>
              <span className="text-primary-green">93%</span> Model Accuracy
            </div>
            <div>
              <span className="text-yellow-400">6</span> Fault Classes
            </div>
            <div>
              <span className="text-primary-blue">13</span> Features
            </div>
          </div>
          <p className="text-sm mt-4 flex items-center justify-center gap-2">
            <span className="text-gray-500">Made by</span>
            <span className="font-heading font-bold bg-gradient-to-r from-purple-400 via-pink-500 to-red-500 bg-clip-text text-transparent">
              ⚡ Blacklists
            </span>
          </p>
        </motion.footer>
      </main>
    </div>
  );
}

export default App;
