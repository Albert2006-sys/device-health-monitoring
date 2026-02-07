import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronDown } from 'lucide-react';
import { AnalysisResult, ReasoningPoint, ReasoningData } from '../types/analysis';

interface AIReasoningProps {
    data: AnalysisResult;
}

const generateReasoningPoints = (data: AnalysisResult): ReasoningPoint[] => {
    const points: ReasoningPoint[] = [];
    // Safely access reasoning_data with empty object default
    const rd: Partial<ReasoningData> = data.reasoning_data || {};
    const threshold = rd.threshold ?? 0.043;
    const anomalyScore = data.anomaly_score ?? 0;

    if (data.status === 'faulty') {
        // Faulty reasoning
        if (rd.windows_analyzed && rd.anomalous_windows !== undefined) {
            const ratio = ((rd.anomalous_windows / rd.windows_analyzed) * 100).toFixed(0);
            points.push({
                icon: '📊',
                text: `Anomaly detected in ${rd.anomalous_windows} of ${rd.windows_analyzed} windows (${ratio}% exceeded threshold)`
            });
        }

        if (anomalyScore > threshold * 3) {
            points.push({
                icon: '⚠️',
                text: `Reconstruction error (${anomalyScore.toFixed(4)}) is ${(anomalyScore / threshold).toFixed(1)}x above threshold — severe deviation`
            });
        } else if (anomalyScore > threshold * 1.5) {
            points.push({
                icon: '⚠️',
                text: `Reconstruction error (${anomalyScore.toFixed(4)}) significantly exceeds threshold (${threshold.toFixed(4)})`
            });
        } else {
            points.push({
                icon: '⚠️',
                text: `Reconstruction error (${anomalyScore.toFixed(4)}) above threshold (${threshold.toFixed(4)})`
            });
        }

        // Fault-specific explanations
        const faultExplanations: Record<string, { icon: string; text: string }[]> = {
            bearing_fault: [
                { icon: '🔬', text: 'High spectral kurtosis indicates impulsive vibration patterns' },
                { icon: '📈', text: 'Dominant frequency aligns with bearing defect characteristics' }
            ],
            worn_brakes: [
                { icon: '🔬', text: 'Spectral patterns indicate abnormal friction characteristics' },
                { icon: '📈', text: 'Frequency profile matches worn brake component signatures' }
            ],
            bad_ignition: [
                { icon: '🔬', text: 'Irregular timing patterns suggest ignition system issues' },
                { icon: '📈', text: 'Engine vibration profile deviates from normal startup' }
            ],
            dead_battery: [
                { icon: '🔬', text: 'Low energy patterns indicate insufficient power supply' },
                { icon: '📈', text: 'Startup audio shows characteristic weak cranking signature' }
            ],
            mixed_faults: [
                { icon: '🔬', text: 'Multiple fault indicators detected across signal' },
                { icon: '📈', text: 'Pattern matches combination of known failure modes' }
            ]
        };

        if (data.failure_type && faultExplanations[data.failure_type]) {
            points.push(...faultExplanations[data.failure_type]);
        }

        if (rd.rf_confidence) {
            points.push({
                icon: '🎯',
                text: `Random Forest classifier: ${(rd.rf_confidence * 100).toFixed(0)}% certainty in fault identification`
            });
        }

    } else {
        // Normal reasoning
        if (rd.windows_analyzed) {
            points.push({
                icon: '📊',
                text: `Signal analyzed across ${rd.windows_analyzed} time windows`
            });
        }

        const errorPercent = ((anomalyScore / threshold) * 100).toFixed(0);
        if (anomalyScore < threshold * 0.2) {
            points.push({
                icon: '✅',
                text: `Reconstruction error (${anomalyScore.toFixed(4)}) is only ${errorPercent}% of threshold — excellent condition`
            });
        } else if (anomalyScore < threshold * 0.5) {
            points.push({
                icon: '✅',
                text: `Reconstruction error (${anomalyScore.toFixed(4)}) is ${errorPercent}% of threshold — well within normal range`
            });
        } else {
            points.push({
                icon: '✅',
                text: `Reconstruction error (${anomalyScore.toFixed(4)}) is ${errorPercent}% of threshold — within acceptable limits`
            });
        }

        if (rd.distance_from_threshold) {
            points.push({
                icon: '🛡️',
                text: `Safety margin: ${rd.distance_from_threshold.toFixed(4)} below anomaly threshold`
            });
        }

        if (rd.anomalous_windows !== undefined && rd.windows_analyzed) {
            if (rd.anomalous_windows === 0) {
                points.push({
                    icon: '🎯',
                    text: 'All windows matched learned healthy baseline patterns'
                });
            }
        }

        points.push({
            icon: '📈',
            text: 'No impulsive or irregular vibration patterns detected'
        });
    }

    // Add fallback if no points were generated
    if (points.length === 0) {
        points.push({
            icon: '🔍',
            text: `Analysis completed with ${data.status} status and ${(data.confidence ?? 0) * 100}% confidence`
        });
    }

    return points;
};

export const AIReasoning: React.FC<AIReasoningProps> = ({ data }) => {
    const [isExpanded, setIsExpanded] = useState(false);
    const points = generateReasoningPoints(data);

    return (
        <motion.div
            className="glass-card rounded-2xl overflow-hidden"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.6 }}
        >
            {/* Header - Always visible */}
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="w-full flex items-center justify-between p-6 hover:bg-white/5 transition-colors"
            >
                <div className="flex items-center gap-3">
                    <span className="text-2xl">🔍</span>
                    <span className="text-lg font-heading font-semibold text-white">
                        AI Reasoning
                    </span>
                </div>
                <motion.div
                    animate={{ rotate: isExpanded ? 180 : 0 }}
                    transition={{ duration: 0.2 }}
                >
                    <ChevronDown className="w-5 h-5 text-gray-400" />
                </motion.div>
            </button>

            {/* Expandable content */}
            <AnimatePresence>
                {isExpanded && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.3 }}
                        className="overflow-hidden"
                    >
                        <div className="px-6 pb-6 space-y-3">
                            {points.map((point, index) => (
                                <motion.div
                                    key={index}
                                    className="flex items-start gap-3 text-gray-300"
                                    initial={{ opacity: 0, x: -20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: index * 0.1 }}
                                >
                                    <span className="text-lg flex-shrink-0 mt-0.5">{point.icon}</span>
                                    <span className="text-sm leading-relaxed">{point.text}</span>
                                </motion.div>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
};

export default AIReasoning;
