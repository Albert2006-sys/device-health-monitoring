// API Response Types

export interface ReasoningData {
    windows_analyzed: number;
    anomalous_windows: number;
    threshold: number;
    rf_confidence: number;
    min_error?: number;
    max_error?: number;
    distance_from_threshold?: number;
}

export interface AnalysisResult {
    status: 'normal' | 'faulty';
    health_score: number;
    anomaly_score: number;
    failure_type: string | null;
    confidence: number | null;
    explanation: string;
    reasoning_data?: ReasoningData;  // Made optional
    processing_ms: number;
}

export type ConfidenceLevel = 'high' | 'medium' | 'low';

export interface ReasoningPoint {
    icon: string;
    text: string;
}
