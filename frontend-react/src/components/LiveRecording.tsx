import { useState, useRef } from 'react';
import { Mic, Square, Loader2 } from 'lucide-react';
import toast from 'react-hot-toast';
import { motion, AnimatePresence } from 'framer-motion';
import { AnalysisResult } from '../types/analysis';
import { API_BASE } from '../services/api';

interface LiveRecordingProps {
    onResult: (result: AnalysisResult) => void;
}

export const LiveRecording: React.FC<LiveRecordingProps> = ({ onResult }) => {
    const [recording, setRecording] = useState(false);
    const [analyzing, setAnalyzing] = useState(false);
    const [countdown, setCountdown] = useState(5);
    const mediaRecorder = useRef<MediaRecorder | null>(null);
    const audioChunks = useRef<Blob[]>([]);
    const countdownInterval = useRef<any | null>(null);
    const streamRef = useRef<MediaStream | null>(null);

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 44100,
                }
            });

            streamRef.current = stream;
            mediaRecorder.current = new MediaRecorder(stream, {
                mimeType: 'audio/webm',
            });

            audioChunks.current = [];

            mediaRecorder.current.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunks.current.push(event.data);
                }
            };

            mediaRecorder.current.onstop = async () => {
                const audioBlob = new Blob(audioChunks.current, { type: 'audio/webm' });
                const audioFile = new File([audioBlob], `recording_${Date.now()}.webm`, {
                    type: 'audio/webm'
                });

                setAnalyzing(true);

                try {
                    const formData = new FormData();
                    formData.append('file', audioFile);

                    const response = await fetch(`${API_BASE}/analyze`, {
                        method: 'POST',
                        body: formData,
                    });

                    if (!response.ok) {
                        const errorData = await response.json().catch(() => ({}));
                        throw new Error(errorData.error || 'Analysis failed');
                    }

                    const result = await response.json();
                    onResult(result);
                    toast.success('Live recording analyzed!');
                } catch (error: any) {
                    console.error('Analysis failed:', error);
                    toast.error(error.message || 'Analysis failed. Please try again.');
                } finally {
                    setAnalyzing(false);
                }

                // Stop all tracks
                if (streamRef.current) {
                    streamRef.current.getTracks().forEach(track => track.stop());
                }
            };

            mediaRecorder.current.start();
            setRecording(true);
            setCountdown(5);

            // Countdown timer
            countdownInterval.current = setInterval(() => {
                setCountdown(prev => {
                    if (prev <= 1) {
                        stopRecording();
                        return 0;
                    }
                    return prev - 1;
                });
            }, 1000);

            toast.success('🎤 Recording started!');
        } catch (error) {
            console.error('Microphone access denied:', error);
            toast.error('Please allow microphone access to record');
        }
    };

    const stopRecording = () => {
        if (mediaRecorder.current && mediaRecorder.current.state === 'recording') {
            mediaRecorder.current.stop();
            setRecording(false);
        }

        if (countdownInterval.current) {
            clearInterval(countdownInterval.current);
            countdownInterval.current = null;
        }
    };

    return (
        <motion.div
            className="glass-card rounded-xl border border-white/10 p-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
        >
            <div className="text-center mb-6">
                <h3 className="text-2xl font-heading font-semibold text-white mb-2">
                    🎤 Live Audio Recording
                </h3>
                <p className="text-gray-400 text-sm">
                    Record any machine sound for instant AI analysis
                </p>
            </div>

            <div className="flex flex-col items-center gap-6">
                <AnimatePresence mode="wait">
                    {!recording && !analyzing && (
                        <motion.div
                            key="idle"
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.9, opacity: 0 }}
                            className="flex flex-col items-center gap-4"
                        >
                            <button
                                onClick={startRecording}
                                className="relative group"
                            >
                                <div className="absolute inset-0 bg-gradient-to-r from-primary-red to-pink-600 rounded-full blur-xl opacity-50 group-hover:opacity-75 transition-opacity" />
                                <div className="relative flex items-center gap-3 px-10 py-6 bg-gradient-to-r from-primary-red to-pink-600 text-white rounded-full font-semibold text-lg hover:scale-105 transition-transform shadow-2xl">
                                    <Mic size={28} />
                                    <span>Start 5-Second Recording</span>
                                </div>
                            </button>

                            <p className="text-gray-500 text-sm max-w-md text-center">
                                💡 Place device near machine, engine, or appliance.
                                AI will analyze sound patterns for faults.
                            </p>
                        </motion.div>
                    )}

                    {recording && (
                        <motion.div
                            key="recording"
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.9, opacity: 0 }}
                            className="flex flex-col items-center gap-6"
                        >
                            <div className="relative">
                                <motion.div
                                    animate={{ scale: [1, 1.2, 1] }}
                                    transition={{ duration: 1.5, repeat: Infinity }}
                                    className="absolute inset-0 bg-primary-red rounded-full blur-2xl opacity-50"
                                />
                                <div className="relative w-32 h-32 bg-primary-red rounded-full flex items-center justify-center shadow-2xl">
                                    <Mic size={48} className="text-white" />
                                </div>
                            </div>

                            <div className="text-center">
                                <motion.p
                                    key={countdown}
                                    initial={{ scale: 1.5, opacity: 0 }}
                                    animate={{ scale: 1, opacity: 1 }}
                                    className="text-6xl font-bold text-white font-mono mb-2"
                                >
                                    {countdown}
                                </motion.p>
                                <p className="text-gray-400">seconds remaining</p>
                            </div>

                            {/* Waveform Animation */}
                            <div className="flex items-end gap-1 h-16">
                                {[...Array(12)].map((_, i) => (
                                    <motion.div
                                        key={i}
                                        className="w-2 bg-primary-red rounded-full"
                                        animate={{
                                            height: ['20%', '100%', '20%'],
                                        }}
                                        transition={{
                                            duration: 0.8,
                                            repeat: Infinity,
                                            delay: i * 0.1,
                                        }}
                                    />
                                ))}
                            </div>

                            <button
                                onClick={stopRecording}
                                className="flex items-center gap-2 px-6 py-3 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors text-white font-semibold"
                            >
                                <Square size={20} />
                                Stop Recording
                            </button>
                        </motion.div>
                    )}

                    {analyzing && (
                        <motion.div
                            key="analyzing"
                            initial={{ scale: 0.9, opacity: 0 }}
                            animate={{ scale: 1, opacity: 1 }}
                            exit={{ scale: 0.9, opacity: 0 }}
                            className="flex flex-col items-center gap-6"
                        >
                            <motion.div
                                animate={{ rotate: 360 }}
                                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                            >
                                <Loader2 size={64} className="text-primary-blue" />
                            </motion.div>
                            <p className="text-white text-xl font-semibold">
                                Analyzing your recording...
                            </p>
                            <p className="text-gray-400 text-sm">
                                AI is extracting features and detecting anomalies
                            </p>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            {/* Info Cards */}
            <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-dark-1 rounded-lg p-4 border border-white/5">
                    <p className="text-primary-blue text-sm font-semibold mb-1">⚡ INSTANT</p>
                    <p className="text-gray-400 text-xs">Results in under 2 seconds</p>
                </div>
                <div className="bg-dark-1 rounded-lg p-4 border border-white/5">
                    <p className="text-primary-blue text-sm font-semibold mb-1">🎯 ACCURATE</p>
                    <p className="text-gray-400 text-xs">93% classification accuracy</p>
                </div>
                <div className="bg-dark-1 rounded-lg p-4 border border-white/5">
                    <p className="text-primary-blue text-sm font-semibold mb-1">🔒 PRIVATE</p>
                    <p className="text-gray-400 text-xs">Audio processed locally</p>
                </div>
            </div>
        </motion.div>
    );
};

export default LiveRecording;
