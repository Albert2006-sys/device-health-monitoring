import { motion } from 'framer-motion';
import { Wrench, AlertTriangle, Clock, CheckCircle } from 'lucide-react';
import { MaintenanceAdvice as MaintenanceAdviceType } from '../types/analysis';

interface MaintenanceAdviceProps {
    advice: MaintenanceAdviceType | undefined;
}

const urgencyConfig = {
    high: {
        icon: <AlertTriangle className="w-4 h-4" />,
        label: 'HIGH',
        bgColor: 'bg-primary-red/10',
        borderColor: 'border-primary-red/30',
        textColor: 'text-primary-red',
        badgeColor: 'bg-primary-red'
    },
    medium: {
        icon: <Clock className="w-4 h-4" />,
        label: 'MEDIUM',
        bgColor: 'bg-yellow-500/10',
        borderColor: 'border-yellow-500/30',
        textColor: 'text-yellow-400',
        badgeColor: 'bg-yellow-500'
    },
    low: {
        icon: <CheckCircle className="w-4 h-4" />,
        label: 'LOW',
        bgColor: 'bg-primary-green/10',
        borderColor: 'border-primary-green/30',
        textColor: 'text-primary-green',
        badgeColor: 'bg-primary-green'
    }
};

export const MaintenanceAdvice: React.FC<MaintenanceAdviceProps> = ({ advice }) => {
    if (!advice) return null;

    const urgency = advice.urgency || 'medium';
    const config = urgencyConfig[urgency] || urgencyConfig.medium;
    const actions = advice.recommended_actions || advice.actions || [];

    return (
        <motion.div
            className={`glass-card p-5 rounded-xl ${config.bgColor} border ${config.borderColor}`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
        >
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                    <div className="p-2 rounded-lg bg-white/5">
                        <Wrench className="w-5 h-5 text-primary-blue" />
                    </div>
                    <h3 className="text-white font-heading font-semibold">
                        🛠️ Maintenance Recommendation
                    </h3>
                </div>

                {/* Urgency Badge */}
                <motion.div
                    className={`flex items-center gap-1 px-3 py-1 rounded-full ${config.badgeColor}`}
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.9, type: 'spring' }}
                >
                    {config.icon}
                    <span className="text-white text-xs font-bold">{config.label}</span>
                </motion.div>
            </div>

            {/* Action Items */}
            {actions.length > 0 && (
                <ul className="space-y-2 mb-4">
                    {actions.map((action, index) => (
                        <motion.li
                            key={index}
                            className="flex items-start gap-2"
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: 0.8 + index * 0.1 }}
                        >
                            <span className={`mt-1.5 w-1.5 h-1.5 rounded-full ${config.badgeColor} flex-shrink-0`} />
                            <span className="text-sm text-gray-300">{action}</span>
                        </motion.li>
                    ))}
                </ul>
            )}

            {/* Note */}
            {advice.note && (
                <div className="bg-dark-1 p-3 rounded-lg border border-white/5">
                    <p className="text-xs text-gray-400 italic">
                        {advice.note}
                    </p>
                </div>
            )}

            {/* Footer */}
            <p className="text-xs text-gray-500 mt-3">
                Generated using fault severity and physics validation.
            </p>
        </motion.div>
    );
};

export default MaintenanceAdvice;
