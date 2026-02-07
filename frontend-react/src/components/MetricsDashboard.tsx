import { motion } from 'framer-motion';
import { Activity, Clock, Layers, Target } from 'lucide-react';
import { MetricsCard } from './MetricsCard';
import { AnalysisResult, ReasoningData } from '../types/analysis';

interface MetricsDashboardProps {
    data: AnalysisResult;
}

export const MetricsDashboard: React.FC<MetricsDashboardProps> = ({ data }) => {
    // Safely access reasoning_data with defaults
    const rd: Partial<ReasoningData> = data.reasoning_data || {};
    const isNormal = data.status === 'normal';

    return (
        <motion.div
            className="grid grid-cols-1 sm:grid-cols-2 gap-4"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.3 }}
        >
            <MetricsCard
                title="Anomaly Score"
                value={(data.anomaly_score ?? 0).toFixed(4)}
                subtitle={`Threshold: ${rd.threshold?.toFixed(4) ?? 'N/A'}`}
                icon={Activity}
                color={isNormal ? 'green' : 'red'}
                delay={0.1}
            />

            <MetricsCard
                title="Processing Time"
                value={`${data.processing_ms ?? 0}ms`}
                subtitle={(data.processing_ms ?? 0) > 1000
                    ? `${((data.processing_ms ?? 0) / 1000).toFixed(1)}s total`
                    : 'Fast analysis'
                }
                icon={Clock}
                color="blue"
                delay={0.2}
            />

            <MetricsCard
                title="Windows Analyzed"
                value={rd.windows_analyzed ?? 1}
                subtitle={`${rd.anomalous_windows ?? 0} anomalous`}
                icon={Layers}
                color={(rd.anomalous_windows ?? 0) > 0 ? 'yellow' : 'green'}
                delay={0.3}
            />

            <MetricsCard
                title="Confidence"
                value={rd.rf_confidence
                    ? `${(rd.rf_confidence * 100).toFixed(0)}%`
                    : (data.confidence ? `${(data.confidence * 100).toFixed(0)}%` : 'N/A')
                }
                subtitle={data.failure_type
                    ? data.failure_type.replace(/_/g, ' ')
                    : 'Healthy pattern'
                }
                icon={Target}
                color={(rd.rf_confidence ?? data.confidence ?? 0) >= 0.8 ? 'green' : 'yellow'}
                delay={0.4}
            />
        </motion.div>
    );
};

export default MetricsDashboard;
