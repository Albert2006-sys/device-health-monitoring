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
