// API Response Types

export interface PhysicsValidation {
    consistent: boolean | null;
    reason: string;
    observed?: {
        dominant_frequency_hz?: number;
        spectral_kurtosis?: number;
        anomaly_score?: number;
    };
    expected?: {
        frequency_range_hz?: string;
        description?: string;
        kurtosis_threshold?: number;
    };
    agree_windows?: number;
    disagree_windows?: number;
    na_windows?: number;
    total_windows?: number;
    agree_ratio?: number;
    summary?: string;
    modulation_applied?: string | null;
}

export interface FailureFingerprint {
    features: {
        rms_energy: number;
        spectral_kurtosis: number;
        dominant_frequency: number;
        spectral_centroid: number;
        mfcc_1: number;
        mfcc_2: number;
        mfcc_3: number;
    };
    baseline: {
        rms_energy: number;
        spectral_kurtosis: number;
        dominant_frequency: number;
        spectral_centroid: number;
        mfcc_1: number;
        mfcc_2: number;
        mfcc_3: number;
    };
    deviation_score: number;
    interpretation: string;
}

export interface MaintenanceAdvice {
    urgency: 'low' | 'medium' | 'high';
    recommended_actions?: string[];
    actions?: string[];
    severity?: string;
    note?: string;
    generated_from?: string;
}

export interface WindowResult {
    start_time: number;
    end_time: number;
    reconstruction_error: number;
    is_anomalous: boolean;
    predicted_class: string;
    confidence: number;
}

export interface ReasoningData {
    windows_analyzed: number;
    anomalous_windows: number;
    threshold: number;
    rf_confidence: number;
    min_error?: number;
    max_error?: number;
    distance_from_threshold?: number;
    dominant_frequency_hz?: number;
    spectral_kurtosis?: number;
    anomaly_ratio?: number;
}

export interface WindowSummary {
    total: number;
    anomalous: number;
    ratio: number;
    overlap: number;
    window_duration: number;
    hop_duration: number;
}

export interface WindowStats {
    total_windows: number;
    anomalous_windows: number;
    anomaly_ratio: number;
    overlap: number;
}

export interface AnalysisResult {
    status: 'normal' | 'warning' | 'faulty';
    health_score: number;
    anomaly_score: number;
    failure_type: string | null;
    confidence: number | null;
    explanation: string;
    reasoning_data?: ReasoningData;
    processing_ms: number;
    out_of_distribution?: boolean;
    physics_validation?: PhysicsValidation;
    failure_fingerprint?: FailureFingerprint;
    maintenance_advice?: MaintenanceAdvice;
    window_results?: WindowResult[];
    window_summary?: WindowSummary;
    window_stats?: WindowStats;
    audio_duration?: number;
}

export type ConfidenceLevel = 'high' | 'medium' | 'low';

export interface ReasoningPoint {
    icon: string;
    text: string;
}
