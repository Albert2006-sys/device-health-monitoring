import { motion } from 'framer-motion';
import { Activity, Github, FileText, Zap, Cpu, Radio, Layers } from 'lucide-react';

interface NavbarProps {
    onDocsClick: () => void;
    onApiClick: () => void;
    onBatchClick?: () => void;
    onDeployClick?: () => void;
    onLiveClick?: () => void;
    onLogoClick?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ onDocsClick, onApiClick, onBatchClick, onDeployClick, onLiveClick, onLogoClick }) => {
    return (
        <motion.nav
            className="fixed top-0 left-0 right-0 z-50 glass-card border-0 border-b border-white/10"
            initial={{ y: -100 }}
            animate={{ y: 0 }}
            transition={{ duration: 0.5 }}
        >
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                    {/* Logo */}
                    <button
                        onClick={onLogoClick}
                        className="flex items-center gap-3 hover:opacity-80 transition-opacity"
                    >
                        <div className="p-2 rounded-xl bg-gradient-to-br from-primary-blue to-primary-green">
                            <Activity className="w-6 h-6 text-dark-1" />
                        </div>
                        <div className="text-left">
                            <h1 className="font-heading font-bold text-lg text-white">
                                Device Health Monitor
                            </h1>
                            <p className="text-xs text-gray-500 font-mono">v2.0 Unified</p>
                        </div>
                    </button>

                    {/* Links */}
                    <div className="flex items-center gap-4">
                        {onBatchClick && (
                            <button
                                onClick={onBatchClick}
                                className="flex items-center gap-2 px-3 py-2 rounded-lg text-gray-400 hover:text-primary-blue hover:bg-white/5 transition-all"
                                title="Batch analysis"
                            >
                                <Layers className="w-4 h-4" />
                                <span className="text-sm hidden sm:inline">Batch</span>
                            </button>
                        )}
                        {onDeployClick && (
                            <button
                                onClick={onDeployClick}
                                className="flex items-center gap-2 px-3 py-2 rounded-lg text-gray-400 hover:text-primary-green hover:bg-white/5 transition-all"
                                title="Deploy to device"
                            >
                                <Cpu className="w-4 h-4" />
                                <span className="text-sm hidden sm:inline">Deploy</span>
                            </button>
                        )}
                        {onLiveClick && (
                            <button
                                onClick={onLiveClick}
                                className="flex items-center gap-2 px-3 py-2 rounded-lg text-gray-400 hover:text-primary-blue hover:bg-white/5 transition-all"
                                title="Live device monitor"
                            >
                                <Radio className="w-4 h-4" />
                                <span className="text-sm hidden sm:inline">Live</span>
                            </button>
                        )}
                        <div className="w-px h-5 bg-white/10 hidden sm:block" />
                        <button
                            onClick={onDocsClick}
                            className="flex items-center gap-2 px-3 py-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-all"
                            title="View documentation"
                        >
                            <FileText className="w-4 h-4" />
                            <span className="text-sm hidden sm:inline">Docs</span>
                        </button>
                        <button
                            onClick={onApiClick}
                            className="flex items-center gap-2 px-3 py-2 rounded-lg text-gray-400 hover:text-white hover:bg-white/5 transition-all"
                            title="API reference"
                        >
                            <Zap className="w-4 h-4" />
                            <span className="text-sm hidden sm:inline">API</span>
                        </button>
                        <a
                            href="https://github.com/Albert2006-sys/device-health-monitoring.git"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-2 px-4 py-2 rounded-lg glass-card text-gray-300 hover:text-white hover:bg-white/10 transition-all"
                            title="View source code"
                        >
                            <Github className="w-4 h-4" />
                            <span className="text-sm hidden sm:inline">GitHub</span>
                        </a>
                    </div>
                </div>
            </div>
        </motion.nav>
    );
};

export default Navbar;
