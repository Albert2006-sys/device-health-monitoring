import { motion } from 'framer-motion';
import { CheckCircle, AlertTriangle, Info, HelpCircle } from 'lucide-react';
import { ConfidenceLevel } from '../types/analysis';
import { useState } from 'react';

interface StatusBadgeProps {
    status: 'normal' | 'warning' | 'faulty';
    confidence: number | null;
    failureType?: string | null;
    outOfDistribution?: boolean;
}

const getConfidenceLevel = (confidence: number | null): ConfidenceLevel => {
    if (!confidence) return 'medium';
    if (confidence >= 0.85) return 'high';
    if (confidence >= 0.60) return 'medium';
    return 'low';
};

const confidenceConfig = {
    high: { icon: '🟢', text: 'High Confidence', color: 'text-primary-green' },
    medium: { icon: '🟡', text: 'Medium Confidence', color: 'text-yellow-400' },
    low: { icon: '🔴', text: 'Low Confidence', color: 'text-primary-red' },
};

const formatFailureType = (type: string | null | undefined): string => {
    if (!type) return '';
    return type
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
};

export const StatusBadge: React.FC<StatusBadgeProps> = ({
    status,
    confidence,
    failureType,
    outOfDistribution = false
}) => {
    const [showTooltip, setShowTooltip] = useState(false);
    const isNormal = status === 'normal';
    const isWarning = status === 'warning';
    const isFaulty = status === 'faulty';
    const confidenceLevel = getConfidenceLevel(confidence);
    const conf = confidenceConfig[confidenceLevel];

    const glowClass = isNormal ? 'glow-green' : isWarning ? 'glow-yellow' : 'glow-red';
    const textColor = isNormal
        ? 'text-primary-green'
        : isWarning
            ? 'text-yellow-400'
            : 'text-primary-red';

    return (
        <motion.div
            className="flex flex-col gap-4"
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
        >
            {/* Status Badge */}
            <motion.div
                className={`flex items-center gap-3 px-6 py-4 rounded-2xl glass-card ${glowClass}`}
                whileHover={{ scale: 1.02 }}
                transition={{ type: 'spring', stiffness: 300 }}
            >
                {isNormal ? (
                    <CheckCircle className="w-8 h-8 text-primary-green" />
                ) : isWarning ? (
                    <AlertTriangle className="w-8 h-8 text-yellow-400" />
                ) : (
                    <AlertTriangle className="w-8 h-8 text-primary-red" />
                )}
                <div>
                    <h3 className={`text-2xl font-heading font-bold ${textColor}`}>
                        {status.toUpperCase()}
                    </h3>
                    {failureType && (
                        <p className="text-sm text-gray-400 font-mono">
                            {formatFailureType(failureType)}
                        </p>
                    )}
                    {outOfDistribution && !failureType && (
                        <p className="text-sm text-yellow-400 font-mono">
                            Unknown Pattern
                        </p>
                    )}
                </div>
            </motion.div>

            {/* Out-of-Distribution Warning */}
            {outOfDistribution && (
                <motion.div
                    className="flex items-start gap-3 px-4 py-3 rounded-xl bg-yellow-500/10 border border-yellow-500/30"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                >
                    <AlertTriangle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                    <div className="flex-1">
                        <p className="text-yellow-400 font-medium text-sm">
                            Unseen audio pattern — diagnosis intentionally limited
                        </p>
                        <div className="relative inline-block mt-2">
                            <button
                                onMouseEnter={() => setShowTooltip(true)}
                                onMouseLeave={() => setShowTooltip(false)}
                                className="flex items-center gap-1 text-xs text-gray-400 hover:text-gray-300 transition-colors"
                            >
                                <HelpCircle className="w-3 h-3" />
                                Why is this limited?
                            </button>
                            {showTooltip && (
                                <div className="absolute bottom-full left-0 mb-2 p-3 bg-dark-1 border border-white/10 rounded-lg shadow-xl z-50 w-72">
                                    <p className="text-xs text-gray-300 leading-relaxed">
                                        This system is optimized for mechanical sounds (engines, bearings, brakes).
                                        Non-machine audio may be flagged as anomalous without a specific diagnosis.
                                        This is intentional safety behavior.
                                    </p>
                                </div>
                            )}
                        </div>
                    </div>
                </motion.div>
            )}

            {/* Confidence Badge */}
            <motion.div
                className="flex items-center gap-3 px-6 py-3 rounded-xl glass-card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
                title="Confidence reflects model certainty based on reconstruction error and classifier probability"
            >
                <span className="text-xl">{conf.icon}</span>
                <div className="flex items-center gap-2">
                    <span className={`font-medium ${conf.color}`}>{conf.text}</span>
                    {confidence && (
                        <span className="text-gray-400 font-mono text-sm">
                            ({(confidence * 100).toFixed(0)}%)
                        </span>
                    )}
                </div>
                <Info className="w-4 h-4 text-gray-500 ml-auto cursor-help" />
            </motion.div>
        </motion.div>
    );
};

export default StatusBadge;
