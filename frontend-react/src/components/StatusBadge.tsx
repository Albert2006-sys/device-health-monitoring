import { motion } from 'framer-motion';
import { CheckCircle, AlertTriangle, Info } from 'lucide-react';
import { ConfidenceLevel } from '../types/analysis';

interface StatusBadgeProps {
    status: 'normal' | 'faulty';
    confidence: number | null;
    failureType?: string | null;
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
    failureType
}) => {
    const isNormal = status === 'normal';
    const confidenceLevel = getConfidenceLevel(confidence);
    const conf = confidenceConfig[confidenceLevel];

    return (
        <motion.div
            className="flex flex-col gap-4"
            initial={{ opacity: 0, x: 50 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
        >
            {/* Status Badge */}
            <motion.div
                className={`flex items-center gap-3 px-6 py-4 rounded-2xl glass-card ${isNormal ? 'glow-green' : 'glow-red'
                    }`}
                whileHover={{ scale: 1.02 }}
                transition={{ type: 'spring', stiffness: 300 }}
            >
                {isNormal ? (
                    <CheckCircle className="w-8 h-8 text-primary-green" />
                ) : (
                    <AlertTriangle className="w-8 h-8 text-primary-red" />
                )}
                <div>
                    <h3 className={`text-2xl font-heading font-bold ${isNormal ? 'text-primary-green' : 'text-primary-red'
                        }`}>
                        {status.toUpperCase()}
                    </h3>
                    {failureType && (
                        <p className="text-sm text-gray-400 font-mono">
                            {formatFailureType(failureType)}
                        </p>
                    )}
                </div>
            </motion.div>

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
