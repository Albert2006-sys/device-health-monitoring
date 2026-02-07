import { Calendar, TrendingDown, DollarSign, AlertCircle, Wrench } from 'lucide-react';
import { motion } from 'framer-motion';

interface FailurePredictionProps {
    currentHealth: number;
    anomalyScore: number;
    threshold: number;
    failureType: string | null;
}

export const FailurePrediction: React.FC<FailurePredictionProps> = ({
    currentHealth,
    anomalyScore,
    threshold,
    failureType,
}) => {
    const calculatePrediction = () => {
        // Excellent condition - no prediction needed
        if (currentHealth >= 90) {
            return {
                days_until_failure: null,
                urgency: 'low' as const,
                message: '✅ Equipment in excellent condition',
                recommendation: 'Continue routine monitoring. No immediate maintenance required.',
                cost_impact: 0,
                degradation_rate: 0,
                predicted_date: null,
            };
        }

        // Calculate degradation rate
        const excessScore = Math.max(0, anomalyScore - threshold);
        const baseRate = 0.5; // Base degradation %/day
        const anomalyMultiplier = excessScore > 0 ? excessScore * 10 : 1;
        const degradationRate = baseRate * anomalyMultiplier;

        // Days until health reaches critical threshold (30)
        const daysUntilCritical = Math.max(1, (currentHealth - 30) / degradationRate);

        // Determine urgency level
        const getUrgency = (days: number) => {
            if (days < 7) return 'critical';
            if (days < 30) return 'high';
            if (days < 90) return 'medium';
            return 'low';
        };

        const urgency = getUrgency(daysUntilCritical);

        // Estimated costs (industry averages)
        const costMultipliers = {
            critical: 75000,
            high: 35000,
            medium: 15000,
            low: 5000,
        };

        // Fault-specific cost adjustments
        const faultCostMultipliers: Record<string, number> = {
            bearing_fault: 1.5,
            worn_brakes: 1.0,
            bad_ignition: 1.2,
            dead_battery: 0.8,
            mixed_faults: 2.0,
        };

        const baseCost = costMultipliers[urgency];
        const faultMultiplier = failureType ? (faultCostMultipliers[failureType] || 1.0) : 1.0;
        const estimatedCost = baseCost * faultMultiplier;

        const messages: Record<string, string> = {
            critical: `🔴 CRITICAL: Predicted failure in ${Math.round(daysUntilCritical)} days`,
            high: `🟠 HIGH PRIORITY: Predicted failure in ${Math.round(daysUntilCritical)} days`,
            medium: `🟡 MEDIUM: Predicted failure in ${Math.round(daysUntilCritical)} days`,
            low: `🟢 LOW RISK: Predicted failure in ${Math.round(daysUntilCritical)} days`,
        };

        const recommendations: Record<string, string> = {
            critical: 'IMMEDIATE ACTION: Shutdown equipment and inspect within 48 hours.',
            high: 'Schedule emergency maintenance within 7 days. Reduce operating load.',
            medium: 'Plan maintenance window within 30 days. Order replacement parts.',
            low: 'Schedule routine maintenance. Continue health monitoring.',
        };

        return {
            days_until_failure: Math.round(daysUntilCritical),
            urgency,
            message: messages[urgency],
            recommendation: recommendations[urgency],
            cost_impact: estimatedCost,
            degradation_rate: degradationRate,
            predicted_date: new Date(Date.now() + daysUntilCritical * 86400000),
        };
    };

    const prediction = calculatePrediction();

    const urgencyConfig = {
        critical: {
            bg: 'bg-primary-red/10',
            border: 'border-primary-red',
            text: 'text-primary-red',
            icon: '🔴',
        },
        high: {
            bg: 'bg-orange-500/10',
            border: 'border-orange-500',
            text: 'text-orange-500',
            icon: '🟠',
        },
        medium: {
            bg: 'bg-yellow-500/10',
            border: 'border-yellow-500',
            text: 'text-yellow-400',
            icon: '🟡',
        },
        low: {
            bg: 'bg-primary-green/10',
            border: 'border-primary-green',
            text: 'text-primary-green',
            icon: '🟢',
        },
    };

    const config = urgencyConfig[prediction.urgency];

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={`rounded-xl border-2 p-6 ${config.bg} ${config.border}`}
        >
            {/* Header */}
            <div className="flex items-start justify-between mb-6">
                <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                        <h3 className="text-xl font-heading font-bold text-white">
                            📈 Predictive Maintenance Forecast
                        </h3>
                        <span className="text-2xl">{config.icon}</span>
                    </div>
                    <p className={`font-semibold ${config.text}`}>
                        {prediction.message}
                    </p>
                </div>
                <AlertCircle className={config.text} size={40} />
            </div>

            {/* Metrics Grid */}
            {prediction.days_until_failure !== null && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <div className="bg-dark-1 rounded-lg p-4 border border-white/5">
                        <div className="flex items-center gap-2 mb-2">
                            <Calendar size={18} className="text-gray-400" />
                            <span className="text-xs text-gray-400 font-semibold">TIME TO FAILURE</span>
                        </div>
                        <p className="text-3xl font-bold font-mono text-white mb-1">
                            {prediction.days_until_failure}
                        </p>
                        <p className="text-sm text-gray-400">days</p>
                        <p className="text-xs text-gray-500 mt-2">
                            ~{prediction.predicted_date?.toLocaleDateString()}
                        </p>
                    </div>

                    <div className="bg-dark-1 rounded-lg p-4 border border-white/5">
                        <div className="flex items-center gap-2 mb-2">
                            <TrendingDown size={18} className="text-gray-400" />
                            <span className="text-xs text-gray-400 font-semibold">DEGRADATION</span>
                        </div>
                        <p className="text-3xl font-bold font-mono text-white mb-1">
                            {prediction.degradation_rate.toFixed(1)}
                        </p>
                        <p className="text-sm text-gray-400">% per day</p>
                    </div>

                    <div className="bg-dark-1 rounded-lg p-4 border border-white/5">
                        <div className="flex items-center gap-2 mb-2">
                            <DollarSign size={18} className="text-gray-400" />
                            <span className="text-xs text-gray-400 font-semibold">FAILURE COST</span>
                        </div>
                        <p className="text-3xl font-bold font-mono text-white mb-1">
                            ${(prediction.cost_impact / 1000).toFixed(0)}K
                        </p>
                        <p className="text-sm text-gray-400">if unaddressed</p>
                    </div>

                    <div className="bg-dark-1 rounded-lg p-4 border border-white/5">
                        <div className="flex items-center gap-2 mb-2">
                            <Wrench size={18} className="text-gray-400" />
                            <span className="text-xs text-gray-400 font-semibold">URGENCY</span>
                        </div>
                        <p className={`text-2xl font-bold uppercase mb-1 ${config.text}`}>
                            {prediction.urgency}
                        </p>
                        <p className="text-sm text-gray-400">priority</p>
                    </div>
                </div>
            )}

            {/* Recommendation */}
            <div className="bg-dark-1 rounded-lg p-5 border-l-4 border-primary-blue">
                <div className="flex items-start gap-3">
                    <div className="p-2 bg-primary-blue/20 rounded-lg">
                        <Wrench size={20} className="text-primary-blue" />
                    </div>
                    <div className="flex-1">
                        <p className="text-sm font-semibold text-white mb-2">
                            📋 Recommended Action:
                        </p>
                        <p className="text-sm text-gray-300 leading-relaxed">
                            {prediction.recommendation}
                        </p>
                    </div>
                </div>
            </div>

            {/* Timeline Visual */}
            {prediction.days_until_failure !== null && (
                <div className="mt-6">
                    <p className="text-xs text-gray-400 mb-2">Degradation Timeline:</p>
                    <div className="relative h-2 bg-gray-700 rounded-full overflow-hidden">
                        <motion.div
                            initial={{ width: 0 }}
                            animate={{ width: `${Math.min(100, (prediction.days_until_failure / 90) * 100)}%` }}
                            transition={{ duration: 1, delay: 0.3 }}
                            className={`h-full ${config.border.replace('border', 'bg')}`}
                        />
                    </div>
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>Now</span>
                        <span>30 days</span>
                        <span>60 days</span>
                        <span>90 days</span>
                    </div>
                </div>
            )}

            <p className="text-xs text-gray-500 mt-6 italic">
                * Prediction based on current degradation rate. Actual failure time may vary with operating conditions.
            </p>
        </motion.div>
    );
};

export default FailurePrediction;
