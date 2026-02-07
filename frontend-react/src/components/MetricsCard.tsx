import { motion } from 'framer-motion';
import { LucideIcon } from 'lucide-react';

interface MetricsCardProps {
    title: string;
    value: string | number;
    subtitle?: string;
    icon: LucideIcon;
    color?: 'blue' | 'green' | 'red' | 'yellow';
    delay?: number;
}

const colorClasses = {
    blue: 'text-primary-blue glow-blue',
    green: 'text-primary-green glow-green',
    red: 'text-primary-red glow-red',
    yellow: 'text-yellow-400',
};

export const MetricsCard: React.FC<MetricsCardProps> = ({
    title,
    value,
    subtitle,
    icon: Icon,
    color = 'blue',
    delay = 0,
}) => {
    return (
        <motion.div
            className="glass-card p-6 rounded-2xl hover:bg-white/10 transition-all duration-300 cursor-default"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay }}
            whileHover={{ y: -4, scale: 1.02 }}
        >
            <div className="flex items-start gap-4">
                <div className={`p-3 rounded-xl bg-dark-3 ${colorClasses[color].split(' ')[0]}`}>
                    <Icon className="w-6 h-6" />
                </div>
                <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-400 uppercase tracking-wider font-mono">
                        {title}
                    </p>
                    <motion.p
                        className={`text-3xl font-heading font-bold mt-1 ${colorClasses[color].split(' ')[0]}`}
                        initial={{ scale: 0.5 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: delay + 0.2, type: 'spring' }}
                    >
                        {value}
                    </motion.p>
                    {subtitle && (
                        <p className="text-sm text-gray-500 mt-1 font-mono truncate">
                            {subtitle}
                        </p>
                    )}
                </div>
            </div>
        </motion.div>
    );
};

export default MetricsCard;
