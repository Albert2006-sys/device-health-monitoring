import { motion } from 'framer-motion';
import { FlaskConical, AlertTriangle, Check, HelpCircle } from 'lucide-react';
import { PhysicsValidation as PhysicsValidationType } from '../types/analysis';
import { useState } from 'react';

interface PhysicsValidationProps {
    validation: PhysicsValidationType | undefined;
}

export const PhysicsValidation: React.FC<PhysicsValidationProps> = ({ validation }) => {
    const [showDetails, setShowDetails] = useState(false);

    if (!validation) return null;

    const isConsistent = validation.consistent === true;
    const isInconsistent = validation.consistent === false;
    const isUnknown = validation.consistent === null;

    const getBadgeConfig = () => {
        if (isConsistent) {
            return {
                icon: <Check className="w-4 h-4" />,
                label: 'Physics-Consistent',
                bgColor: 'bg-primary-green/10',
                borderColor: 'border-primary-green/30',
                textColor: 'text-primary-green',
            };
        } else if (isInconsistent) {
            return {
                icon: <AlertTriangle className="w-4 h-4" />,
                label: 'Physics-Inconsistent',
                bgColor: 'bg-yellow-500/10',
                borderColor: 'border-yellow-500/30',
                textColor: 'text-yellow-400',
            };
        } else {
            return {
                icon: <HelpCircle className="w-4 h-4" />,
                label: 'Physics N/A',
                bgColor: 'bg-gray-500/10',
                borderColor: 'border-gray-500/30',
                textColor: 'text-gray-400',
            };
        }
    };

    const config = getBadgeConfig();

    return (
        <motion.div
            className="space-y-3"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
        >
            {/* Badge */}
            <motion.div
                className={`flex items-center gap-3 px-4 py-3 rounded-xl ${config.bgColor} border ${config.borderColor}`}
                whileHover={{ scale: 1.02 }}
            >
                <div className={`p-2 rounded-lg bg-white/5 ${config.textColor}`}>
                    <FlaskConical className="w-5 h-5" />
                </div>
                <div className="flex-1">
                    <div className="flex items-center gap-2">
                        {config.icon}
                        <span className={`font-medium ${config.textColor}`}>{config.label}</span>
                    </div>
                    <p className="text-xs text-gray-400 mt-1">Physics-Informed ML Validation</p>
                </div>
                <button
                    onClick={() => setShowDetails(!showDetails)}
                    className="text-gray-500 hover:text-gray-300 text-xs underline"
                >
                    {showDetails ? 'Hide' : 'Details'}
                </button>
            </motion.div>

            {/* Details Panel */}
            {showDetails && validation.reason && (
                <motion.div
                    className="glass-card p-4 rounded-xl space-y-3"
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                >
                    <div className="flex items-start gap-2">
                        <FlaskConical className="w-4 h-4 text-primary-blue mt-0.5" />
                        <div>
                            <p className="text-sm text-white font-medium">🔬 Physics Reasoning</p>
                            <p className="text-sm text-gray-300 mt-1">{validation.reason}</p>
                        </div>
                    </div>

                    {/* Observed Values */}
                    {validation.observed && (
                        <div className="bg-dark-1 p-3 rounded-lg border border-white/5">
                            <p className="text-xs text-gray-400 mb-2">Observed Signal Properties</p>
                            <div className="grid grid-cols-2 gap-2 text-xs">
                                {validation.observed.dominant_frequency_hz !== undefined && (
                                    <div>
                                        <span className="text-gray-500">Peak Frequency:</span>
                                        <span className="text-primary-blue ml-1 font-mono">
                                            {validation.observed.dominant_frequency_hz.toFixed(0)} Hz
                                        </span>
                                    </div>
                                )}
                                {validation.observed.spectral_kurtosis !== undefined && (
                                    <div>
                                        <span className="text-gray-500">Impulsiveness:</span>
                                        <span className="text-yellow-400 ml-1 font-mono">
                                            {validation.observed.spectral_kurtosis.toFixed(2)}
                                        </span>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Expected Values */}
                    {validation.expected && validation.expected.frequency_range_hz && (
                        <div className="bg-dark-1 p-3 rounded-lg border border-white/5">
                            <p className="text-xs text-gray-400 mb-2">Expected Mechanical Behavior</p>
                            <div className="text-xs">
                                <span className="text-gray-500">Frequency Range:</span>
                                <span className="text-primary-green ml-1 font-mono">
                                    {validation.expected.frequency_range_hz} Hz
                                </span>
                            </div>
                            {validation.expected.description && (
                                <p className="text-xs text-gray-400 mt-1 italic">
                                    {validation.expected.description}
                                </p>
                            )}
                        </div>
                    )}

                    {/* PIML Note */}
                    <p className="text-xs text-gray-500 italic">
                        This system uses Physics-Informed ML (PIML), combining data-driven models
                        with mechanical validation to improve trust and explainability.
                    </p>
                </motion.div>
            )}
        </motion.div>
    );
};

export default PhysicsValidation;
