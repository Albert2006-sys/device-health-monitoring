import { AnalysisResult } from '../types/analysis';

export const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5000';
console.log('[API] Base URL:', API_BASE);

export const analyzeFile = async (file: File): Promise<AnalysisResult> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        body: formData
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Analysis failed' }));
        throw new Error(error.error || 'Analysis failed');
    }

    return response.json();
};

export const analyzeDemo = async (type: 'normal' | 'faulty'): Promise<AnalysisResult> => {
    const response = await fetch(`${API_BASE}/analyze/demo?type=${type}`);

    if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Demo analysis failed' }));
        throw new Error(error.error || 'Demo analysis failed');
    }

    return response.json();
};

export const checkHealth = async (): Promise<{ status: string }> => {
    const response = await fetch(`${API_BASE}/health`);
    return response.json();
};
