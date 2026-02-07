import { useState, useCallback } from 'react';
import { analyzeFile, analyzeDemo } from '../services/api';
import { AnalysisResult } from '../types/analysis';
import { saveToHistory } from '../utils/analysisHistory';
import toast from 'react-hot-toast';

export const useAnalysis = () => {
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const analyze = useCallback(async (file: File) => {
        setLoading(true);
        setError(null);

        try {
            const data = await analyzeFile(file);
            setResult(data);

            // Save to history with file name
            saveToHistory(data, file.name);

            toast.success('Analysis complete!', {
                icon: data.status === 'normal' ? '✅' : '⚠️',
            });
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Analysis failed';
            setError(message);
            toast.error(message);
        } finally {
            setLoading(false);
        }
    }, []);

    const runDemo = useCallback(async (type: 'normal' | 'faulty') => {
        setLoading(true);
        setError(null);

        try {
            const data = await analyzeDemo(type);
            setResult(data);

            // Save to history with demo type
            saveToHistory(data, `Demo (${type})`);

            toast.success(`Demo ${type} sample analyzed!`, {
                icon: type === 'normal' ? '✅' : '⚠️',
            });
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Demo failed';
            setError(message);
            toast.error(message);
        } finally {
            setLoading(false);
        }
    }, []);

    const reset = useCallback(() => {
        setResult(null);
        setError(null);
    }, []);

    return { result, loading, error, analyze, runDemo, reset, setResult };
};
