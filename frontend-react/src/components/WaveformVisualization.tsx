import { useRef, useState, useEffect } from 'react';
import { Play, Pause, ZoomIn, ZoomOut } from 'lucide-react';
import { motion } from 'framer-motion';
import { WindowResult } from '../types/analysis';

interface WaveformVisualizationProps {
    windowResults: WindowResult[];
    threshold: number;
}

export const WaveformVisualization: React.FC<WaveformVisualizationProps> = ({
    windowResults,
    threshold,
}) => {
    const [currentWindow, setCurrentWindow] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [zoom, setZoom] = useState(1);
    const intervalRef = useRef<NodeJS.Timeout | null>(null);

    useEffect(() => {
        if (isPlaying) {
            intervalRef.current = setInterval(() => {
                setCurrentWindow(prev => {
                    if (prev >= windowResults.length - 1) {
                        setIsPlaying(false);
                        return 0;
                    }
                    return prev + 1;
                });
            }, 500);
        }
        return () => {
            if (intervalRef.current) clearInterval(intervalRef.current);
        };
    }, [isPlaying, windowResults.length]);

    const togglePlayback = () => {
        if (currentWindow >= windowResults.length - 1) {
            setCurrentWindow(0);
        }
        setIsPlaying(!isPlaying);
    };

    const current = windowResults[currentWindow];
    const maxError = Math.max(...windowResults.map(w => w.reconstruction_error));

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="glass-card p-6 rounded-xl"
        >
            <div className="flex items-center justify-between mb-4">
                <div>
                    <h3 className="text-lg font-heading font-semibold text-white mb-1">
                        🎵 Waveform Analysis - Time Domain View
                    </h3>
                    <p className="text-sm text-gray-400">
                        Green regions = Normal | Red regions = Anomalous
                    </p>
                </div>

                <div className="flex items-center gap-2">
                    <button
                        onClick={togglePlayback}
                        className="p-2 bg-primary-blue/20 hover:bg-primary-blue/30 rounded-lg transition-colors text-primary-blue"
                        title="Play/Pause"
                    >
                        {isPlaying ? <Pause size={20} /> : <Play size={20} />}
                    </button>

                    <button
                        onClick={() => setZoom(Math.max(0.5, zoom - 0.25))}
                        className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors text-white"
                        title="Zoom Out"
                    >
                        <ZoomOut size={20} />
                    </button>

                    <button
                        onClick={() => setZoom(Math.min(2, zoom + 0.25))}
                        className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors text-white"
                        title="Zoom In"
                    >
                        <ZoomIn size={20} />
                    </button>
                </div>
            </div>

            {/* Waveform Visualization */}
            <div className="relative h-32 bg-dark-1 rounded-lg overflow-hidden mb-4">
                <div
                    className="flex h-full items-end gap-px transition-transform duration-200"
                    style={{ transform: `scaleX(${zoom})`, transformOrigin: 'left' }}
                >
                    {windowResults.map((window, i) => {
                        const height = (window.reconstruction_error / maxError) * 100;
                        const isActive = i === currentWindow;

                        return (
                            <motion.div
                                key={i}
                                className={`flex-1 min-w-[8px] cursor-pointer transition-all relative ${isActive ? 'ring-2 ring-primary-blue' : ''
                                    }`}
                                style={{ height: '100%' }}
                                onClick={() => setCurrentWindow(i)}
                                whileHover={{ scale: 1.1 }}
                            >
                                {/* Bar */}
                                <div
                                    className={`absolute bottom-0 w-full rounded-t transition-colors ${window.is_anomalous
                                            ? 'bg-gradient-to-t from-primary-red to-primary-red/50'
                                            : 'bg-gradient-to-t from-primary-green to-primary-green/50'
                                        }`}
                                    style={{ height: `${Math.max(10, height)}%` }}
                                />

                                {/* Threshold line indicator */}
                                <div
                                    className="absolute w-full h-px bg-yellow-500/50"
                                    style={{ bottom: `${(threshold / maxError) * 100}%` }}
                                />
                            </motion.div>
                        );
                    })}
                </div>

                {/* Threshold reference line */}
                <div
                    className="absolute left-0 right-0 border-t border-dashed border-yellow-500/70"
                    style={{ bottom: `${(threshold / maxError) * 100}%` }}
                >
                    <span className="absolute right-1 -top-4 text-xs text-yellow-500">Threshold</span>
                </div>
            </div>

            {/* Window Grid */}
            <div className="flex gap-1 mb-4 overflow-x-auto pb-2">
                {windowResults.map((window, i) => (
                    <motion.div
                        key={i}
                        className={`flex-shrink-0 h-10 rounded transition-all cursor-pointer flex items-center justify-center ${window.is_anomalous
                                ? 'bg-primary-red/20 border border-primary-red'
                                : 'bg-primary-green/20 border border-primary-green'
                            } ${currentWindow === i ? 'ring-2 ring-primary-blue scale-110' : ''}`}
                        style={{ width: `${Math.max(32, 100 / windowResults.length)}px` }}
                        title={`Window ${i + 1}: Error ${window.reconstruction_error.toFixed(4)}`}
                        onClick={() => setCurrentWindow(i)}
                        whileHover={{ scale: 1.05 }}
                    >
                        <span className="text-xs text-white font-mono">{i + 1}</span>
                    </motion.div>
                ))}
            </div>

            {/* Current Window Info */}
            {current && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                    <div className="bg-dark-1 rounded-lg p-3 border border-white/5">
                        <p className="text-gray-400 text-xs mb-1">WINDOW</p>
                        <p className="text-white font-mono text-lg">{currentWindow + 1} / {windowResults.length}</p>
                    </div>
                    <div className="bg-dark-1 rounded-lg p-3 border border-white/5">
                        <p className="text-gray-400 text-xs mb-1">ERROR</p>
                        <p className="text-white font-mono text-lg">{current.reconstruction_error.toFixed(4)}</p>
                    </div>
                    <div className="bg-dark-1 rounded-lg p-3 border border-white/5">
                        <p className="text-gray-400 text-xs mb-1">STATUS</p>
                        <p className={`font-semibold text-lg ${current.is_anomalous ? 'text-primary-red' : 'text-primary-green'}`}>
                            {current.is_anomalous ? '⚠️ ANOMALY' : '✓ NORMAL'}
                        </p>
                    </div>
                    <div className="bg-dark-1 rounded-lg p-3 border border-white/5">
                        <p className="text-gray-400 text-xs mb-1">CONFIDENCE</p>
                        <p className="text-white font-mono text-lg">{(current.confidence * 100).toFixed(0)}%</p>
                    </div>
                </div>
            )}

            <p className="text-xs text-gray-500 mt-4 italic">
                💡 Each bar represents a 1-second window. Click any bar to inspect that segment. Red bars exceed the anomaly threshold.
            </p>
        </motion.div>
    );
};

export default WaveformVisualization;
