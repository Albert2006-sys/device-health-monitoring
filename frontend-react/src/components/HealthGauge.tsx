import { motion } from 'framer-motion';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

interface HealthGaugeProps {
    score: number;
    animated?: boolean;
}

const getColor = (value: number) => {
    if (value >= 90) return '#00FF9F'; // Green
    if (value >= 70) return '#FFD700'; // Yellow
    if (value >= 50) return '#FF8C00'; // Orange
    return '#FF0055'; // Red
};

const getGradient = (value: number) => {
    if (value >= 90) return 'radial-gradient(circle, rgba(0, 255, 159, 0.4) 0%, transparent 70%)';
    if (value >= 70) return 'radial-gradient(circle, rgba(255, 215, 0, 0.4) 0%, transparent 70%)';
    if (value >= 50) return 'radial-gradient(circle, rgba(255, 140, 0, 0.4) 0%, transparent 70%)';
    return 'radial-gradient(circle, rgba(255, 0, 85, 0.4) 0%, transparent 70%)';
};

export const HealthGauge: React.FC<HealthGaugeProps> = ({
    score,
    animated = true
}) => {
    const color = getColor(score);

    return (
        <motion.div
            className="relative w-64 h-64 mx-auto"
            initial={animated ? { scale: 0.5, opacity: 0, rotate: -180 } : {}}
            animate={{ scale: 1, opacity: 1, rotate: 0 }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
        >
            {/* Glow effect behind */}
            <motion.div
                className="absolute inset-0 rounded-full blur-2xl opacity-60"
                style={{ background: getGradient(score), zIndex: 0 }}
                animate={{
                    scale: [1, 1.1, 1],
                    opacity: [0.4, 0.6, 0.4]
                }}
                transition={{
                    duration: 2,
                    repeat: Infinity,
                    ease: 'easeInOut'
                }}
            />

            {/* Main gauge */}
            <div className="relative z-10">
                <CircularProgressbar
                    value={score}
                    text={`${score}`}
                    styles={buildStyles({
                        pathColor: color,
                        textColor: color,
                        trailColor: 'rgba(255, 255, 255, 0.1)',
                        pathTransitionDuration: 1,
                        textSize: '28px',
                    })}
                />
            </div>

            {/* Score label */}
            <motion.div
                className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 text-center"
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 }}
            >
                <p className="text-sm text-gray-400 font-mono tracking-widest uppercase">
                    Health Score
                </p>
            </motion.div>
        </motion.div>
    );
};

export default HealthGauge;
