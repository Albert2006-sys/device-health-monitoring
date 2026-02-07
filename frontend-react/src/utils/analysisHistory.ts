import { AnalysisResult } from '../types/analysis';

const STORAGE_KEY = 'device_health_analysis_history';
const MAX_HISTORY = 10;

export interface AnalysisHistoryItem {
    id: string;
    timestamp: string;
    fileName: string;
    // Store the full result for restoration
    result: AnalysisResult;
}

/**
 * Save an analysis result to localStorage history
 */
export const saveToHistory = (
    result: AnalysisResult,
    fileName: string = 'Demo Sample'
): void => {
    try {
        const history = getHistory();

        const newItem: AnalysisHistoryItem = {
            id: crypto.randomUUID(),
            timestamp: new Date().toISOString(),
            fileName,
            result, // Store full result
        };

        // Add to beginning and limit to MAX_HISTORY items
        const updatedHistory = [newItem, ...history].slice(0, MAX_HISTORY);

        localStorage.setItem(STORAGE_KEY, JSON.stringify(updatedHistory));
    } catch (error) {
        console.warn('Failed to save to history:', error);
    }
};

/**
 * Get analysis history from localStorage
 */
export const getHistory = (): AnalysisHistoryItem[] => {
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (!stored) return [];
        return JSON.parse(stored);
    } catch (error) {
        console.warn('Failed to read history:', error);
        return [];
    }
};

/**
 * Get a specific history item by ID
 */
export const getHistoryItem = (id: string): AnalysisHistoryItem | null => {
    const history = getHistory();
    return history.find(item => item.id === id) || null;
};

/**
 * Clear all history
 */
export const clearHistory = (): void => {
    localStorage.removeItem(STORAGE_KEY);
};

/**
 * Format timestamp for display
 */
export const formatTimestamp = (isoString: string): string => {
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
};
