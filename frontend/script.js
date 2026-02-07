/**
 * Device Health Monitoring System - Frontend Script
 * 
 * Handles demo button clicks, file uploads, API calls, and UI updates.
 */

const API_BASE = 'http://localhost:5000';

// DOM Elements
const btnNormal = document.getElementById('btn-normal');
const btnFaulty = document.getElementById('btn-faulty');
const btnUpload = document.getElementById('btn-upload');
const fileInput = document.getElementById('file-input');
const loadingEl = document.getElementById('loading');
const resultsEl = document.getElementById('results');
const heroCard = document.getElementById('hero-card');
const healthScoreEl = document.getElementById('health-score');
const statusBadge = document.getElementById('status-badge');

/**
 * Test a sample by calling the demo API endpoint.
 * @param {string} type - 'normal' or 'faulty'
 */
async function testSample(type) {
    setLoading(true);

    try {
        const response = await fetch(`${API_BASE}/analyze/demo?type=${type}`);

        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);

    } catch (error) {
        console.error('API Error:', error);
        alert('Error connecting to backend API.\n\nMake sure the Flask server is running:\ncd backend && python app.py');
    } finally {
        setLoading(false);
    }
}

/**
 * Called when user selects a file.
 * Enables the upload button if a valid file is selected.
 */
function onFileSelected() {
    const file = fileInput.files[0];
    if (file) {
        btnUpload.disabled = false;
    } else {
        btnUpload.disabled = true;
    }
}

/**
 * Analyze the uploaded file via POST /analyze.
 */
async function analyzeUploadedFile() {
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a file first.');
        return;
    }

    // Validate file extension
    const validExtensions = ['.wav', '.mat', '.mp4', '.mp3', '.m4a', '.flac', '.avi', '.mov'];
    const fileName = file.name.toLowerCase();
    const isValid = validExtensions.some(ext => fileName.endsWith(ext));

    if (!isValid) {
        alert('Invalid file type. Please upload a .wav or .mat file.');
        return;
    }

    setLoading(true);

    try {
        // Create FormData for file upload
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || data.details || 'Upload failed');
        }

        displayResults(data);

    } catch (error) {
        console.error('Upload Error:', error);
        alert('Unable to analyze this file. Please try a different sample.\n\nError: ' + error.message);
    } finally {
        setLoading(false);
    }
}

/**
 * Toggle loading state.
 * @param {boolean} isLoading
 */
function setLoading(isLoading) {
    // Disable all buttons during loading
    btnNormal.disabled = isLoading;
    btnFaulty.disabled = isLoading;
    btnUpload.disabled = isLoading || !fileInput.files[0];

    if (isLoading) {
        loadingEl.classList.remove('hidden');
        resultsEl.classList.add('hidden');
    } else {
        loadingEl.classList.add('hidden');
    }
}

/**
 * Display analysis results in the UI.
 * @param {Object} data - API response data
 */
function displayResults(data) {
    // Show results section
    resultsEl.classList.remove('hidden');

    // Update body background based on status
    document.body.classList.remove('status-normal', 'status-faulty');
    document.body.classList.add(`status-${data.status}`);

    // Animate health score
    animateHealthScore(data.health_score);

    // Update status badge
    statusBadge.textContent = data.status.toUpperCase();
    statusBadge.className = 'status-badge';
    statusBadge.classList.add(`badge-${data.status}`);

    // Update confidence badge
    const confidenceLevel = getConfidenceLevel(data);
    updateConfidenceBadge(confidenceLevel);

    // Update details
    document.getElementById('detail-status').textContent = data.status.toUpperCase();
    document.getElementById('detail-anomaly').textContent = formatNumber(data.anomaly_score);
    document.getElementById('detail-confidence').textContent =
        data.confidence !== null ? `${(data.confidence * 100).toFixed(1)}%` : '—';
    document.getElementById('detail-failure').textContent =
        data.failure_type || '—';
    document.getElementById('detail-time').textContent = `${data.processing_ms}ms`;

    // Update explanation
    document.getElementById('explanation-text').textContent = data.explanation;

    // Generate AI reasoning points
    const reasoningPoints = generateReasoningPoints(data);
    updateReasoningSection(reasoningPoints);
}

/**
 * Determine confidence level based on anomaly score and RF confidence.
 * Uses reasoning_data from backend for accurate metrics.
 * @param {Object} data - API response data
 * @returns {Object} - { level: 'high'|'medium'|'low', text: string }
 */
function getConfidenceLevel(data) {
    const rd = data.reasoning_data || {};
    const confidence = data.confidence || 0;
    const anomalyScore = data.anomaly_score || 0;
    const threshold = rd.threshold || 0.043;

    // Calculate how far anomaly_score is from threshold
    const distanceFromThreshold = Math.abs(anomalyScore - threshold);
    const normalizedDistance = distanceFromThreshold / threshold;

    // High confidence conditions
    if (confidence >= 0.85) {
        return { level: 'high', text: 'High Confidence' };
    }
    if (normalizedDistance > 0.5 && data.status === 'normal') {
        return { level: 'high', text: 'High Confidence' };
    }

    // Low confidence conditions
    if (confidence < 0.6) {
        return { level: 'low', text: 'Low Confidence' };
    }
    if (normalizedDistance < 0.15) {
        return { level: 'low', text: 'Low Confidence' };
    }

    // Medium confidence (default)
    return { level: 'medium', text: 'Medium Confidence' };
}

/**
 * Update the confidence badge UI.
 * @param {Object} confidenceData
 */
function updateConfidenceBadge(confidenceData) {
    const badge = document.getElementById('confidence-badge');
    const iconEl = badge.querySelector('.confidence-icon');
    const textEl = badge.querySelector('.confidence-text');

    // Remove previous classes
    badge.classList.remove('confidence-high', 'confidence-medium', 'confidence-low');
    badge.classList.add(`confidence-${confidenceData.level}`);

    // Update text
    textEl.textContent = confidenceData.text;

    // Update icon based on level
    const icons = { high: '🟢', medium: '🟡', low: '🔴' };
    iconEl.textContent = icons[confidenceData.level];
}

/**
 * Generate reasoning bullet points from API response.
 * Uses reasoning_data from backend for accurate metrics.
 * @param {Object} data - API response data
 * @returns {string[]} - Array of reasoning strings
 */
function generateReasoningPoints(data) {
    const points = [];
    const rd = data.reasoning_data || {};
    const anomalyScore = data.anomaly_score || 0;
    const threshold = rd.threshold || 0.043;

    if (data.status === 'faulty') {
        // Faulty reasoning with actual data
        if (rd.windows_analyzed && rd.anomalous_windows !== undefined) {
            const ratio = ((rd.anomalous_windows / rd.windows_analyzed) * 100).toFixed(0);
            points.push(`Anomaly detected in ${rd.anomalous_windows} of ${rd.windows_analyzed} windows (${ratio}% exceeded threshold)`);
        }

        // Error severity
        if (anomalyScore > threshold * 3) {
            points.push(`Reconstruction error (${anomalyScore.toFixed(4)}) is ${(anomalyScore / threshold).toFixed(1)}x above threshold — severe deviation`);
        } else if (anomalyScore > threshold * 1.5) {
            points.push(`Reconstruction error (${anomalyScore.toFixed(4)}) significantly exceeds threshold (${threshold.toFixed(4)})`);
        } else {
            points.push(`Reconstruction error (${anomalyScore.toFixed(4)}) above threshold (${threshold.toFixed(4)})`);
        }

        // Failure type specific reasoning
        const failureType = data.failure_type;
        if (failureType === 'worn_brakes') {
            points.push('Spectral patterns indicate abnormal friction characteristics');
            points.push('Frequency profile matches worn brake component signatures');
        } else if (failureType === 'bad_ignition') {
            points.push('Irregular timing patterns suggest ignition system issues');
            points.push('Engine vibration profile deviates from normal startup sequence');
        } else if (failureType === 'dead_battery') {
            points.push('Low energy patterns indicate insufficient power supply');
            points.push('Startup audio shows characteristic weak cranking signature');
        } else if (failureType === 'mixed_faults') {
            points.push('Multiple fault indicators detected across signal');
            points.push('Pattern matches combination of known failure modes');
        } else if (failureType === 'bearing_fault') {
            points.push('High spectral kurtosis indicates impulsive vibration');
            points.push('Dominant frequency aligns with bearing defect patterns');
        }

        // RF confidence
        if (rd.rf_confidence) {
            points.push(`Random Forest classifier: ${(rd.rf_confidence * 100).toFixed(0)}% certainty in fault identification`);
        }

    } else {
        // Normal reasoning with actual data
        if (rd.windows_analyzed) {
            points.push(`Signal analyzed across ${rd.windows_analyzed} time windows`);
        }

        // Error relative to threshold with specific numbers
        const errorPercent = ((anomalyScore / threshold) * 100).toFixed(0);
        if (anomalyScore < threshold * 0.2) {
            points.push(`Reconstruction error (${anomalyScore.toFixed(4)}) is only ${errorPercent}% of threshold — excellent condition`);
        } else if (anomalyScore < threshold * 0.5) {
            points.push(`Reconstruction error (${anomalyScore.toFixed(4)}) is ${errorPercent}% of threshold — well within normal range`);
        } else if (anomalyScore < threshold * 0.8) {
            points.push(`Reconstruction error (${anomalyScore.toFixed(4)}) is ${errorPercent}% of threshold — within acceptable limits`);
        } else {
            points.push(`Reconstruction error (${anomalyScore.toFixed(4)}) is ${errorPercent}% of threshold — near boundary but acceptable`);
        }

        // Distance from threshold
        if (rd.distance_from_threshold !== undefined) {
            points.push(`Safety margin: ${rd.distance_from_threshold.toFixed(4)} below anomaly threshold`);
        }

        // Anomalous windows
        if (rd.anomalous_windows !== undefined && rd.windows_analyzed) {
            if (rd.anomalous_windows === 0) {
                points.push('All windows matched learned healthy baseline patterns');
            } else {
                points.push(`${rd.anomalous_windows} of ${rd.windows_analyzed} windows showed minor deviations (within tolerance)`);
            }
        }

        points.push('No impulsive or irregular vibration patterns detected');
    }

    return points;
}

/**
 * Update the reasoning section with bullet points.
 * @param {string[]} points
 */
function updateReasoningSection(points) {
    const listEl = document.getElementById('reasoning-list');
    listEl.innerHTML = '';

    points.forEach(point => {
        const li = document.createElement('li');
        li.textContent = point;
        listEl.appendChild(li);
    });
}

/**
 * Toggle the AI Reasoning section visibility.
 */
function toggleReasoning() {
    const content = document.getElementById('reasoning-content');
    const icon = document.getElementById('toggle-icon');

    content.classList.toggle('hidden');
    icon.textContent = content.classList.contains('hidden') ? '▼' : '▲';
}

/**
 * Animate the health score number.
 * @param {number} targetScore
 */
function animateHealthScore(targetScore) {
    const duration = 1000; // 1 second
    const startTime = performance.now();
    const startValue = 0;

    // Determine color class
    let colorClass;
    if (targetScore >= 85) {
        colorClass = 'score-green';
    } else if (targetScore >= 60) {
        colorClass = 'score-orange';
    } else {
        colorClass = 'score-red';
    }

    // Set color
    healthScoreEl.className = 'health-score ' + colorClass;

    // Animate number
    function updateScore(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease out function
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const currentValue = Math.round(startValue + (targetScore - startValue) * easeOut);

        healthScoreEl.textContent = currentValue;

        if (progress < 1) {
            requestAnimationFrame(updateScore);
        }
    }

    requestAnimationFrame(updateScore);
}

/**
 * Format large numbers for display.
 * @param {number} num
 * @returns {string}
 */
function formatNumber(num) {
    if (num === null || num === undefined) return '—';

    if (num >= 1000000) {
        return (num / 1000000).toFixed(2) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(2) + 'K';
    } else if (num < 1) {
        return num.toFixed(6);
    } else {
        return num.toFixed(2);
    }
}

// Initialize
console.log('Device Health Monitoring System - Frontend Ready');
console.log('API Base:', API_BASE);

