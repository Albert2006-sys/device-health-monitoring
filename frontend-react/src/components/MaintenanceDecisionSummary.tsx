import { Shield, Activity, FlaskConical, Wrench, Info } from 'lucide-react';
import { motion } from 'framer-motion';
import type { PhysicsValidation } from '../types/analysis';

interface MaintenanceDecisionSummaryProps {
    status: 'normal' | 'warning' | 'faulty';
    confidence: number | null;
    anomalyScore: number;
    failureType: string | null;
    physicsValidation?: PhysicsValidation;
    explanation: string;
}

/** Confidence tier derived from the ML output */
const getConfidenceTier = (c: number) => {
    if (c >= 0.85) return { label: 'High', color: 'text-primary-green', bg: 'bg-primary-green/20' };
    if (c >= 0.6) return { label: 'Medium', color: 'text-yellow-400', bg: 'bg-yellow-400/20' };
    return { label: 'Low', color: 'text-orange-400', bg: 'bg-orange-400/20' };
};

/** Risk level derived purely from the 3-state status */
const riskConfig = {
    normal: {
        label: 'Low Risk',
        color: 'text-primary-green',
        bg: 'bg-primary-green/10',
        border: 'border-primary-green',
        icon: '●',
    },
    warning: {
        label: 'Medium Risk',
        color: 'text-yellow-400',
        bg: 'bg-yellow-500/10',
        border: 'border-yellow-500',
        icon: '●',
    },
    faulty: {
        label: 'High Risk',
        color: 'text-primary-red',
        bg: 'bg-primary-red/10',
        border: 'border-primary-red',
        icon: '●',
    },
};

/** Physics badge text mapping */
const physicsBadge = (consistent: boolean | null | undefined) => {
    if (consistent === true) return { label: 'Physics Consistent', color: 'text-primary-green', bg: 'bg-primary-green/15' };
    if (consistent === false) return { label: 'Physics Conflict', color: 'text-yellow-400', bg: 'bg-yellow-400/15' };
    return { label: 'Physics Not Applicable', color: 'text-gray-400', bg: 'bg-gray-500/15' };
};

/** Rule-based recommended action -- no predictions, no forecasts */
const getRecommendedAction = (status: string) => {
    if (status === 'normal') {
        return 'No immediate maintenance required. Continue routine monitoring.';
    }
    if (status === 'warning') {
        return (
            'Early signs of abnormal behaviour detected. ' +
            'Schedule inspection or increase monitoring frequency.'
        );
    }
    return (
        'Consistent anomaly patterns detected. ' +
        'Maintenance intervention recommended.'
    );
};

/** Human-readable fault label */
const formatFault = (f: string) =>
    f.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

export const MaintenanceDecisionSummary: React.FC<MaintenanceDecisionSummaryProps> = ({
    status,
    confidence,
    anomalyScore,
    failureType,
    physicsValidation,
    explanation,
}) => {
    const risk = riskConfig[status] ?? riskConfig.normal;
    const conf = getConfidenceTier(confidence ?? 0);
    const physics = physicsBadge(physicsValidation?.consistent);
    const action = getRecommendedAction(status);

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={`rounded-xl border-2 p-6 ${risk.bg} ${risk.border}`}
        >
            {/* ── Header ── */}
            <div className="flex items-start justify-between mb-5">
                <div>
                    <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-1">
                        Decision Support (Condition-Based)
                    </p>
                    <h3 className="text-xl font-heading font-bold text-white">
                        Maintenance Decision Summary
                    </h3>
                </div>
                <Shield className={risk.color} size={36} />
            </div>

            {/* ── Indicator Row ── */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-6">
                {/* Risk Level */}
                <div className="bg-dark-1 rounded-lg p-4 border border-white/5">
                    <div className="flex items-center gap-2 mb-2">
                        <Shield size={16} className="text-gray-400" />
                        <span className="text-xs text-gray-400 font-semibold uppercase">
                            Risk Level
                        </span>
                    </div>
                    <p className={`text-2xl font-bold ${risk.color}`}>{risk.label}</p>
                    {failureType && (
                        <p className="text-xs text-gray-400 mt-1">
                            Detected: {formatFault(failureType)}
                        </p>
                    )}
                </div>

                {/* Confidence */}
                <div className="bg-dark-1 rounded-lg p-4 border border-white/5">
                    <div className="flex items-center gap-2 mb-2">
                        <Activity size={16} className="text-gray-400" />
                        <span className="text-xs text-gray-400 font-semibold uppercase">
                            Confidence
                        </span>
                    </div>
                    <div className="flex items-baseline gap-2">
                        <p className="text-2xl font-bold font-mono text-white">
                            {confidence !== null ? `${(confidence * 100).toFixed(0)}%` : '--'}
                        </p>
                        <span
                            className={`text-xs font-semibold px-2 py-0.5 rounded-full ${conf.bg} ${conf.color}`}
                        >
                            {conf.label}
                        </span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                        Anomaly score: {anomalyScore.toFixed(4)}
                    </p>
                </div>

                {/* Physics Validation Badge */}
                <div className="bg-dark-1 rounded-lg p-4 border border-white/5">
                    <div className="flex items-center gap-2 mb-2">
                        <FlaskConical size={16} className="text-gray-400" />
                        <span className="text-xs text-gray-400 font-semibold uppercase">
                            Physics Check
                        </span>
                    </div>
                    <p className={`text-lg font-bold ${physics.color}`}>{physics.label}</p>
                    {physicsValidation?.reason && (
                        <p className="text-xs text-gray-500 mt-1 line-clamp-2">
                            {physicsValidation.reason}
                        </p>
                    )}
                </div>
            </div>

            {/* ── Recommended Action ── */}
            <div className="bg-dark-1 rounded-lg p-5 border-l-4 border-primary-blue mb-4">
                <div className="flex items-start gap-3">
                    <div className="p-2 bg-primary-blue/20 rounded-lg shrink-0">
                        <Wrench size={20} className="text-primary-blue" />
                    </div>
                    <div className="flex-1">
                        <p className="text-sm font-semibold text-white mb-1">
                            Recommended Action
                        </p>
                        <p className="text-sm text-gray-300 leading-relaxed">{action}</p>
                    </div>
                </div>
            </div>

            {/* ── Explanation Panel (reuses backend AI reasoning text) ── */}
            {explanation && (
                <div className="bg-dark-1/60 rounded-lg p-4 border border-white/5">
                    <div className="flex items-center gap-2 mb-2">
                        <Info size={14} className="text-gray-400" />
                        <span className="text-xs text-gray-400 font-semibold uppercase">
                            Analysis Explanation
                        </span>
                    </div>
                    <p className="text-sm text-gray-300 leading-relaxed">{explanation}</p>
                </div>
            )}

            <p className="text-xs text-gray-600 mt-4 italic">
                This assessment reflects current machine condition only. It does not
                predict remaining useful life or future failure timelines.
            </p>
        </motion.div>
    );
};

export default MaintenanceDecisionSummary;
