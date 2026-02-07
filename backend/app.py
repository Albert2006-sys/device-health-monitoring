"""
Device Health Monitoring System - Flask API

Endpoints:
- GET  /health   - Health check
- POST /analyze  - Analyze uploaded audio/mat file for machine health

Safe for hackathon demo: never crashes, always returns JSON.
"""

import os
import sys
import time
import tempfile
from flask import Flask, jsonify, request
from flask_cors import CORS

# Add utils directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from analyze import MachineHealthAnalyzer

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}},
     supports_credentials=False,
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     methods=["GET", "POST", "OPTIONS"])

# Initialize analyzer once on startup
analyzer = None


def get_analyzer():
    """Lazy initialization of analyzer."""
    global analyzer
    if analyzer is None:
        analyzer = MachineHealthAnalyzer()
    return analyzer


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route('/analyze', methods=['POST'])
def analyze_file():
    """
    Analyze an uploaded file for machine health.
    
    Accepts:
        - File upload (multipart/form-data) with key 'file'
        - JSON with 'file_path' key for local file path
    
    Returns:
        JSON with status, health_score, anomaly_score, etc.
    """
    start_time = time.perf_counter()
    
    try:
        file_path = None
        temp_file = None
        
        # Option 1: File upload
        if 'file' in request.files:
            uploaded_file = request.files['file']
            
            if uploaded_file.filename == '':
                return jsonify({
                    "error": "Invalid or missing input file"
                }), 400
            
            # Save to temp file (close it immediately for Windows compatibility)
            suffix = os.path.splitext(uploaded_file.filename)[1]
            fd, file_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)  # Close file descriptor immediately
            uploaded_file.save(file_path)
            temp_file = file_path  # Store path for cleanup
        
        # Option 2: JSON with file path
        elif request.is_json and 'file_path' in request.json:
            file_path = request.json['file_path']
            
            if not os.path.exists(file_path):
                return jsonify({
                    "error": "Invalid or missing input file"
                }), 400
        
        else:
            return jsonify({
                "error": "Invalid or missing input file"
            }), 400
        
        # Validate file extension
        valid_extensions = ['.mat', '.wav', '.mp3', '.flac', '.mp4', '.avi', '.mov', '.mkv', '.m4a', '.m4v', '.webm']
        if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
            if temp_file:
                try:
                    os.unlink(temp_file)
                except:
                    pass
            return jsonify({
                "error": "Invalid or missing input file"
            }), 400
        
        # Run inference
        analysis = get_analyzer()
        try:
            result = analysis.analyze(file_path)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                "error": f"Analysis failed: {str(e)}"
            }), 500
        
        # Clean up temp file
        if temp_file:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        # Calculate processing time
        end_time = time.perf_counter()
        processing_ms = int((end_time - start_time) * 1000)
        
        # Build response - pass through all analysis fields + processing time
        response = {**result, "processing_ms": processing_ms}
        
        return jsonify(response), 200

    except Exception as e:
        # Clean up temp file if exists
        if 'temp_file' in locals() and temp_file:
            try:
                os.unlink(temp_file)
            except:
                pass
        
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500


@app.route('/analyze/demo', methods=['GET'])
def analyze_demo():
    """
    Demo endpoint that analyzes pre-loaded sample files.
    
    Query params:
        type: 'normal' or 'faulty'
    
    Returns:
        JSON analysis result
    """
    start_time = time.perf_counter()
    
    try:
        sample_type = request.args.get('type', 'normal')
        
        # Get sample file paths
        base_dir = os.path.dirname(__file__)
        if sample_type == 'faulty':
            file_path = os.path.join(base_dir, 'data', 'test', 'faulty', 'IR007_0_1797.mat')
        else:
            file_path = os.path.join(base_dir, 'data', 'train', 'normal', 'Normal_0.mat')
        
        if not os.path.exists(file_path):
            return jsonify({
                "error": "Demo sample file not found"
            }), 404
        
        # Run inference
        analysis = get_analyzer()
        result = analysis.analyze(file_path)
        
        # Calculate processing time
        end_time = time.perf_counter()
        processing_ms = int((end_time - start_time) * 1000)
        
        # Build response - pass through all analysis fields + processing time
        response = {**result, "processing_ms": processing_ms}
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("Starting Device Health Monitoring API...")
    print(f"  Listening on port {port}")
    print("Endpoints:")
    print("  GET  /health        - Health check")
    print("  POST /analyze       - Analyze uploaded file")
    print("  GET  /analyze/demo  - Demo with sample files (?type=normal|faulty)")
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true')
