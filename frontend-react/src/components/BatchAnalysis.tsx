import { useState } from 'react';
import { Upload, Download, AlertCircle, CheckCircle } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import toast from 'react-hot-toast';
import { motion, AnimatePresence } from 'framer-motion';
import { AnalysisResult } from '../types/analysis';

interface BatchResult {
    filename: string;
    result: AnalysisResult;
    timestamp: number;
}

export const BatchAnalysis = () => {
    const [results, setResults] = useState<BatchResult[]>([]);
    const [analyzing, setAnalyzing] = useState(false);
    const [progress, setProgress] = useState({ current: 0, total: 0 });
    const [sortBy, setSortBy] = useState<'health' | 'name' | 'status'>('health');

    const handleMultipleFiles = async (files: FileList) => {
        if (files.length === 0) return;

        setAnalyzing(true);
        setProgress({ current: 0, total: files.length });
        setResults([]);

        const batchResults: BatchResult[] = [];

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            setProgress({ current: i + 1, total: files.length });

            try {
                const formData = new FormData();
                formData.append('file', file);

                const response = await fetch('http://127.0.0.1:5000/analyze', {
                    method: 'POST',
                    body: formData,
                });

                if (!response.ok) throw new Error('Analysis failed');

                const result = await response.json();
                batchResults.push({
                    filename: file.name,
                    result,
                    timestamp: Date.now(),
                });
                toast.success(`✓ ${file.name}`, { duration: 1500 });
            } catch (error) {
                console.error(`Failed to analyze ${file.name}:`, error);
                toast.error(`✗ ${file.name}`);
            }
        }

        setResults(batchResults);
        setAnalyzing(false);
        toast.success(`Analyzed ${batchResults.length} files!`);
    };

    const exportCSV = () => {
        const headers = ['Filename', 'Health Score', 'Status', 'Failure Type', 'Confidence (%)', 'Anomaly Score'];
        const rows = results.map(r => [
            r.filename,
            r.result.health_score,
            r.result.status,
            r.result.failure_type || 'N/A',
            ((r.result.confidence || 0) * 100).toFixed(1),
            r.result.anomaly_score.toFixed(4),
        ]);

        const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `fleet_analysis_${Date.now()}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        toast.success('CSV exported!');
    };

    const getSortedResults = () => {
        const sorted = [...results];
        switch (sortBy) {
            case 'health':
                return sorted.sort((a, b) => a.result.health_score - b.result.health_score);
            case 'name':
                return sorted.sort((a, b) => a.filename.localeCompare(b.filename));
            case 'status':
                return sorted.sort((a, b) => a.result.status.localeCompare(b.result.status));
            default:
                return sorted;
        }
    };

    const sortedResults = getSortedResults();

    const chartData = sortedResults.map(r => ({
        name: r.filename.substring(0, 15),
        health: r.result.health_score,
        status: r.result.status,
    }));

    const getBarColor = (health: number) => {
        if (health >= 90) return '#00FF9F';
        if (health >= 70) return '#FFD700';
        if (health >= 50) return '#FF8C00';
        return '#FF0055';
    };

    const stats = {
        total: results.length,
        healthy: results.filter(r => r.result.status === 'normal').length,
        faulty: results.filter(r => r.result.status === 'faulty').length,
        avgHealth: results.reduce((sum, r) => sum + r.result.health_score, 0) / (results.length || 1),
    };

    return (
        <div className="space-y-6">
            {/* Upload Zone */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="glass-card rounded-xl border-2 border-dashed border-gray-600 hover:border-gray-500 transition-colors p-12 text-center"
            >
                <input
                    type="file"
                    multiple
                    accept=".wav,.mat,.mp3,.mp4,.m4a,.flac,.webm"
                    onChange={(e) => e.target.files && handleMultipleFiles(e.target.files)}
                    className="hidden"
                    id="batch-upload"
                    disabled={analyzing}
                />
                <label htmlFor="batch-upload" className="cursor-pointer block">
                    {analyzing ? (
                        <div className="space-y-4">
                            <div className="w-16 h-16 border-4 border-primary-blue border-t-transparent rounded-full animate-spin mx-auto" />
                            <p className="text-white text-lg">
                                Analyzing... {progress.current} of {progress.total}
                            </p>
                            <div className="max-w-md mx-auto h-2 bg-gray-700 rounded-full overflow-hidden">
                                <motion.div
                                    className="h-full bg-primary-blue"
                                    initial={{ width: 0 }}
                                    animate={{ width: `${(progress.current / progress.total) * 100}%` }}
                                />
                            </div>
                        </div>
                    ) : (
                        <>
                            <Upload className="w-16 h-16 text-gray-500 mx-auto mb-4" />
                            <p className="text-white text-xl font-heading font-semibold mb-2">
                                📊 Fleet Analysis - Upload Multiple Files
                            </p>
                            <p className="text-gray-400 text-sm">
                                Drag & drop or click to select multiple audio files
                            </p>
                            <p className="text-gray-500 text-xs mt-2">
                                .wav, .mat, .mp3, .mp4, .m4a, .flac
                            </p>
                        </>
                    )}
                </label>
            </motion.div>

            <AnimatePresence>
                {results.length > 0 && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="space-y-6"
                    >
                        {/* Summary Stats */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="glass-card rounded-lg p-4 border border-white/10">
                                <p className="text-gray-400 text-sm mb-1">TOTAL FILES</p>
                                <p className="text-3xl font-bold text-white font-mono">{stats.total}</p>
                            </div>
                            <div className="glass-card rounded-lg p-4 border border-primary-green/50">
                                <p className="text-gray-400 text-sm mb-1">HEALTHY</p>
                                <p className="text-3xl font-bold text-primary-green font-mono">{stats.healthy}</p>
                            </div>
                            <div className="glass-card rounded-lg p-4 border border-primary-red/50">
                                <p className="text-gray-400 text-sm mb-1">FAULTY</p>
                                <p className="text-3xl font-bold text-primary-red font-mono">{stats.faulty}</p>
                            </div>
                            <div className="glass-card rounded-lg p-4 border border-primary-blue/50">
                                <p className="text-gray-400 text-sm mb-1">AVG HEALTH</p>
                                <p className="text-3xl font-bold text-primary-blue font-mono">
                                    {Math.round(stats.avgHealth)}
                                </p>
                            </div>
                        </div>

                        {/* Chart */}
                        <div className="glass-card rounded-xl p-6 border border-white/10">
                            <div className="flex items-center justify-between mb-6">
                                <h3 className="text-xl font-heading font-semibold text-white">
                                    📊 Fleet Health Overview
                                </h3>
                                <button
                                    onClick={exportCSV}
                                    className="flex items-center gap-2 px-4 py-2 bg-primary-green/20 hover:bg-primary-green/30 rounded-lg transition-colors text-primary-green font-semibold"
                                >
                                    <Download size={18} />
                                    Export CSV
                                </button>
                            </div>

                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={chartData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                    <XAxis
                                        dataKey="name"
                                        stroke="#9CA3AF"
                                        angle={-45}
                                        textAnchor="end"
                                        height={80}
                                        fontSize={11}
                                    />
                                    <YAxis
                                        stroke="#9CA3AF"
                                        domain={[0, 100]}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: '#1A1F3A',
                                            border: '1px solid #374151',
                                            borderRadius: '8px',
                                        }}
                                        labelStyle={{ color: '#fff' }}
                                    />
                                    <Bar dataKey="health" radius={[8, 8, 0, 0]}>
                                        {chartData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={getBarColor(entry.health)} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>

                        {/* Results Table */}
                        <div className="glass-card rounded-xl p-6 border border-white/10">
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-lg font-heading font-semibold text-white">
                                    Detailed Results
                                </h3>
                                <div className="flex items-center gap-2">
                                    <span className="text-sm text-gray-400">Sort by:</span>
                                    <select
                                        value={sortBy}
                                        onChange={(e) => setSortBy(e.target.value as 'health' | 'name' | 'status')}
                                        className="bg-dark-1 text-white px-3 py-1 rounded border border-white/10 text-sm focus:outline-none focus:border-primary-blue"
                                    >
                                        <option value="health">Health Score</option>
                                        <option value="name">Filename</option>
                                        <option value="status">Status</option>
                                    </select>
                                </div>
                            </div>

                            <div className="overflow-x-auto">
                                <table className="w-full text-sm">
                                    <thead>
                                        <tr className="border-b border-white/10">
                                            <th className="text-left py-3 px-4 text-gray-400 font-semibold">#</th>
                                            <th className="text-left py-3 px-4 text-gray-400 font-semibold">Filename</th>
                                            <th className="text-center py-3 px-4 text-gray-400 font-semibold">Health</th>
                                            <th className="text-center py-3 px-4 text-gray-400 font-semibold">Status</th>
                                            <th className="text-center py-3 px-4 text-gray-400 font-semibold">Failure Type</th>
                                            <th className="text-center py-3 px-4 text-gray-400 font-semibold">Confidence</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {sortedResults.map((r, i) => (
                                            <motion.tr
                                                key={i}
                                                initial={{ opacity: 0, x: -20 }}
                                                animate={{ opacity: 1, x: 0 }}
                                                transition={{ delay: i * 0.05 }}
                                                className="border-b border-white/5 hover:bg-white/5 transition-colors"
                                            >
                                                <td className="py-3 px-4 text-gray-400">{i + 1}</td>
                                                <td className="py-3 px-4 text-white font-mono text-xs">
                                                    {r.filename}
                                                </td>
                                                <td className="text-center py-3 px-4">
                                                    <span
                                                        className="font-bold font-mono text-lg"
                                                        style={{ color: getBarColor(r.result.health_score) }}
                                                    >
                                                        {r.result.health_score}
                                                    </span>
                                                </td>
                                                <td className="text-center py-3 px-4">
                                                    <div className="flex items-center justify-center gap-2">
                                                        {r.result.status === 'normal' ? (
                                                            <CheckCircle size={16} className="text-primary-green" />
                                                        ) : (
                                                            <AlertCircle size={16} className="text-primary-red" />
                                                        )}
                                                        <span className={`text-xs font-semibold uppercase ${r.result.status === 'normal' ? 'text-primary-green' : 'text-primary-red'
                                                            }`}>
                                                            {r.result.status}
                                                        </span>
                                                    </div>
                                                </td>
                                                <td className="text-center py-3 px-4 text-gray-300 capitalize text-xs">
                                                    {r.result.failure_type?.replace(/_/g, ' ') || '—'}
                                                </td>
                                                <td className="text-center py-3 px-4 text-gray-300 font-mono">
                                                    {((r.result.confidence || 0) * 100).toFixed(0)}%
                                                </td>
                                            </motion.tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
};

export default BatchAnalysis;
