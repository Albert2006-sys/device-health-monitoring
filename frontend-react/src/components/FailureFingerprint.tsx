import { motion } from 'framer-motion';
import { FailureFingerprint as FingerprintType } from '../types/analysis';
import { useState } from 'react';
import { ChevronDown, ChevronUp, Fingerprint } from 'lucide-react';

interface FailureFingerprintProps {
    fingerprint: FingerprintType | undefined | null;
}

// Radar chart configuration
const FEATURES = [
    { key: 'rms_energy', label: 'RMS Energy', short: 'RMS' },
    { key: 'spectral_kurtosis', label: 'Kurtosis', short: 'Kurt' },
    { key: 'dominant_frequency', label: 'Dom Freq', short: 'Freq' },
    { key: 'spectral_centroid', label: 'Centroid', short: 'Cent' },
    { key: 'mfcc_1', label: 'MFCC 1', short: 'M1' },
    { key: 'mfcc_2', label: 'MFCC 2', short: 'M2' },
    { key: 'mfcc_3', label: 'MFCC 3', short: 'M3' },
];

const RadarChart: React.FC<{ current: Record<string, number>; baseline: Record<string, number> }> = ({
    current,
    baseline
}) => {
    const size = 200;
    const center = size / 2;
    const maxRadius = center - 30;
    const numPoints = FEATURES.length;

    // Calculate polygon points
    const getPolygonPoints = (values: Record<string, number>) => {
        return FEATURES.map((feature, i) => {
            const angle = (Math.PI * 2 * i) / numPoints - Math.PI / 2;
            const value = values[feature.key] || 0;
            const radius = value * maxRadius;
            const x = center + radius * Math.cos(angle);
            const y = center + radius * Math.sin(angle);
            return `${x},${y}`;
        }).join(' ');
    };

    // Calculate label positions
    const getLabelPosition = (index: number) => {
        const angle = (Math.PI * 2 * index) / numPoints - Math.PI / 2;
        const labelRadius = maxRadius + 20;
        return {
            x: center + labelRadius * Math.cos(angle),
            y: center + labelRadius * Math.sin(angle)
        };
    };

    // Grid circles
    const gridLevels = [0.25, 0.5, 0.75, 1.0];

    return (
        <svg viewBox={`0 0 ${size} ${size}`} className="w-full h-auto max-w-[220px]">
            {/* Background grid circles */}
            {gridLevels.map((level, i) => (
                <circle
                    key={i}
                    cx={center}
                    cy={center}
                    r={maxRadius * level}
                    fill="none"
                    stroke="rgba(255,255,255,0.1)"
                    strokeWidth="1"
                />
            ))}

            {/* Grid lines from center */}
            {FEATURES.map((_, i) => {
                const angle = (Math.PI * 2 * i) / numPoints - Math.PI / 2;
                const x2 = center + maxRadius * Math.cos(angle);
                const y2 = center + maxRadius * Math.sin(angle);
                return (
                    <line
                        key={i}
                        x1={center}
                        y1={center}
                        x2={x2}
                        y2={y2}
                        stroke="rgba(255,255,255,0.1)"
                        strokeWidth="1"
                    />
                );
            })}

            {/* Baseline polygon (green) */}
            <motion.polygon
                points={getPolygonPoints(baseline)}
                fill="rgba(34, 197, 94, 0.2)"
                stroke="rgb(34, 197, 94)"
                strokeWidth="2"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
            />

            {/* Current polygon (red/orange) */}
            <motion.polygon
                points={getPolygonPoints(current)}
                fill="rgba(239, 68, 68, 0.3)"
                stroke="rgb(239, 68, 68)"
                strokeWidth="2"
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.5, type: 'spring' }}
            />

            {/* Data points for current */}
            {FEATURES.map((feature, i) => {
                const angle = (Math.PI * 2 * i) / numPoints - Math.PI / 2;
                const value = current[feature.key] || 0;
                const x = center + value * maxRadius * Math.cos(angle);
                const y = center + value * maxRadius * Math.sin(angle);
                return (
                    <motion.circle
                        key={`current-${i}`}
                        cx={x}
                        cy={y}
                        r="3"
                        fill="rgb(239, 68, 68)"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: 0.6 + i * 0.05 }}
                    />
                );
            })}

            {/* Labels */}
            {FEATURES.map((feature, i) => {
                const pos = getLabelPosition(i);
                return (
                    <text
                        key={i}
                        x={pos.x}
                        y={pos.y}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        className="text-[9px] fill-gray-400"
                    >
                        {feature.short}
                    </text>
                );
            })}
        </svg>
    );
};

export const FailureFingerprint: React.FC<FailureFingerprintProps> = ({ fingerprint }) => {
    const [expanded, setExpanded] = useState(false);

    if (!fingerprint) return null;

    return (
        <motion.div
            className="glass-card p-4 rounded-xl"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
        >
            {/* Header */}
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <Fingerprint className="w-5 h-5 text-primary-blue" />
                    <h3 className="text-white font-medium">Failure Fingerprint</h3>
                </div>
                <button
                    onClick={() => setExpanded(!expanded)}
                    className="text-gray-400 hover:text-white transition-colors"
                >
                    {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>
            </div>

            {/* Radar Chart */}
            <div className="flex justify-center">
                <RadarChart
                    current={fingerprint.features}
                    baseline={fingerprint.baseline}
                />
            </div>

            {/* Legend */}
            <div className="flex justify-center gap-6 mt-2 text-xs">
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-primary-green/40 border border-primary-green"></div>
                    <span className="text-gray-400">Healthy Baseline</span>
                </div>
                <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded-full bg-primary-red/40 border border-primary-red"></div>
                    <span className="text-gray-400">Current Sample</span>
                </div>
            </div>

            {/* Caption */}
            <p className="text-xs text-gray-500 text-center mt-2 italic">
                Each fault creates a unique vibration signature.
            </p>

            {/* Expanded details */}
            {expanded && (
                <motion.div
                    className="mt-4 pt-4 border-t border-white/10"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                >
                    <p className="text-xs text-gray-300 mb-3">
                        This fingerprint shows how the current machine behavior
                        deviates from the learned healthy baseline across key
                        vibration features.
                    </p>

                    <div className="bg-dark-1 p-3 rounded-lg border border-white/5">
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-xs text-gray-400">Deviation Score:</span>
                            <span className={`text-sm font-mono ${fingerprint.deviation_score > 0.3 ? 'text-primary-red' :
                                    fingerprint.deviation_score > 0.15 ? 'text-yellow-400' : 'text-primary-green'
                                }`}>
                                {(fingerprint.deviation_score * 100).toFixed(0)}%
                            </span>
                        </div>
                        <p className="text-xs text-gray-500">{fingerprint.interpretation}</p>
                    </div>
                </motion.div>
            )}
        </motion.div>
    );
};

export default FailureFingerprint;
