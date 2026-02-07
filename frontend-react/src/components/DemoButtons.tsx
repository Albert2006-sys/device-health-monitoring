import { motion } from 'framer-motion';
import { CheckCircle, AlertTriangle, Loader2 } from 'lucide-react';

interface DemoButtonsProps {
    onNormalClick: () => Promise<void>;
    onFaultyClick: () => Promise<void>;
    loading?: boolean;
    loadingType?: 'normal' | 'faulty' | null;
}

export const DemoButtons: React.FC<DemoButtonsProps> = ({
    onNormalClick,
    onFaultyClick,
    loading = false,
    loadingType = null,
}) => {
    return (
        <motion.div
            className="flex flex-wrap justify-center gap-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
        >
            {/* Normal Sample Button */}
            <motion.button
                onClick={onNormalClick}
                disabled={loading}
                className={`
          flex items-center gap-3 px-8 py-4 rounded-2xl font-heading font-semibold text-lg
          bg-gradient-to-r from-primary-green to-emerald-500
          text-dark-1 shadow-lg
          hover:shadow-primary-green/30 hover:shadow-xl
          disabled:opacity-50 disabled:cursor-not-allowed
          transition-all duration-300
        `}
                whileHover={!loading ? { scale: 1.05, y: -2 } : {}}
                whileTap={!loading ? { scale: 0.98 } : {}}
            >
                {loadingType === 'normal' ? (
                    <Loader2 className="w-6 h-6 animate-spin" />
                ) : (
                    <CheckCircle className="w-6 h-6" />
                )}
                Test Normal Sample
            </motion.button>

            {/* Faulty Sample Button */}
            <motion.button
                onClick={onFaultyClick}
                disabled={loading}
                className={`
          flex items-center gap-3 px-8 py-4 rounded-2xl font-heading font-semibold text-lg
          bg-gradient-to-r from-primary-red to-orange-500
          text-white shadow-lg
          hover:shadow-primary-red/30 hover:shadow-xl
          disabled:opacity-50 disabled:cursor-not-allowed
          transition-all duration-300
        `}
                whileHover={!loading ? { scale: 1.05, y: -2 } : {}}
                whileTap={!loading ? { scale: 0.98 } : {}}
            >
                {loadingType === 'faulty' ? (
                    <Loader2 className="w-6 h-6 animate-spin" />
                ) : (
                    <AlertTriangle className="w-6 h-6" />
                )}
                Test Faulty Sample
            </motion.button>
        </motion.div>
    );
};

export default DemoButtons;
