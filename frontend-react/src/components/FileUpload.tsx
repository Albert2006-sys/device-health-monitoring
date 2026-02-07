import { useCallback, useState } from 'react';
import { motion } from 'framer-motion';
import { Upload, FileAudio, Loader2, X } from 'lucide-react';

interface FileUploadProps {
    onUpload: (file: File) => Promise<void>;
    loading?: boolean;
    disabled?: boolean;
}

const ACCEPTED_FORMATS = ['.wav', '.mat', '.mp3', '.mp4', '.m4a', '.flac'];

export const FileUpload: React.FC<FileUploadProps> = ({
    onUpload,
    loading = false,
    disabled = false,
}) => {
    const [isDragging, setIsDragging] = useState(false);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);

    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        if (!disabled && !loading) setIsDragging(true);
    }, [disabled, loading]);

    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    }, []);

    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        if (disabled || loading) return;

        const file = e.dataTransfer.files[0];
        if (file) {
            setSelectedFile(file);
        }
    }, [disabled, loading]);

    const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setSelectedFile(file);
        }
    }, []);

    const handleAnalyze = async () => {
        if (selectedFile) {
            await onUpload(selectedFile);
        }
    };

    const clearFile = () => {
        setSelectedFile(null);
    };

    return (
        <motion.div
            className="w-full max-w-xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
        >
            <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`
          relative p-8 rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer
          ${isDragging
                        ? 'border-primary-blue bg-primary-blue/10 scale-105'
                        : 'border-gray-600 hover:border-gray-400 glass-card'
                    }
          ${disabled || loading ? 'opacity-50 cursor-not-allowed' : ''}
        `}
            >
                <input
                    type="file"
                    accept={ACCEPTED_FORMATS.join(',')}
                    onChange={handleFileSelect}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    disabled={disabled || loading}
                />

                <div className="flex flex-col items-center gap-4 text-center">
                    {loading ? (
                        <>
                            <Loader2 className="w-12 h-12 text-primary-blue animate-spin" />
                            <p className="text-lg font-medium text-primary-blue">Analyzing...</p>
                        </>
                    ) : selectedFile ? (
                        <>
                            <FileAudio className="w-12 h-12 text-primary-green" />
                            <div>
                                <p className="text-lg font-medium text-white">{selectedFile.name}</p>
                                <p className="text-sm text-gray-400">
                                    {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                                </p>
                            </div>
                        </>
                    ) : (
                        <>
                            <Upload className={`w-12 h-12 ${isDragging ? 'text-primary-blue' : 'text-gray-400'}`} />
                            <div>
                                <p className="text-lg font-medium text-white">
                                    Drop file here or click to browse
                                </p>
                                <p className="text-sm text-gray-500 mt-1 font-mono">
                                    {ACCEPTED_FORMATS.join(', ')}
                                </p>
                            </div>
                        </>
                    )}
                </div>
            </div>

            {/* Action buttons */}
            {selectedFile && !loading && (
                <motion.div
                    className="flex gap-3 mt-4 justify-center"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                >
                    <button
                        onClick={handleAnalyze}
                        className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-primary-blue to-primary-green rounded-xl font-medium text-dark-1 hover:opacity-90 transition-opacity"
                    >
                        <FileAudio className="w-5 h-5" />
                        Analyze File
                    </button>
                    <button
                        onClick={clearFile}
                        className="flex items-center gap-2 px-4 py-3 glass-card rounded-xl text-gray-400 hover:text-white hover:bg-white/10 transition-all"
                    >
                        <X className="w-5 h-5" />
                        Clear
                    </button>
                </motion.div>
            )}
        </motion.div>
    );
};

export default FileUpload;
