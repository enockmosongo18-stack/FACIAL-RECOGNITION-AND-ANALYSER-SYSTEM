"""
Flask Web Application

Web interface for the Face Recognition and Analysis System.
Provides REST API and user-friendly web interface.
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import logging
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from io import BytesIO
import base64

from src.face_detection import FaceDetector
from src.face_recognition import FaceRecognizer
from src.face_analysis import FaceAnalyzer
from src.database import FaceDatabase
from src.utils import ImageUtils, FileUtils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize modules
detector = FaceDetector(model_type="haarcascade")
recognizer = FaceRecognizer()
analyzer = FaceAnalyzer()
database = FaceDatabase()


def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'services': {
            'face_detection': 'active',
            'face_recognition': 'active',
            'face_analysis': 'active',
            'database': 'active'
        }
    })


@app.route('/api/detect', methods=['POST'])
def detect_faces():
    """Detect faces in uploaded image"""
    
    try:
        # Check if file is provided
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Detect faces
        image = cv2.imread(filepath)
        faces = detector.detect_faces(image=image)
        
        # Prepare response
        response_faces = []
        for i, face in enumerate(faces):
            response_faces.append({
                'id': i,
                'x': face['x'],
                'y': face['y'],
                'width': face['width'],
                'height': face['height'],
                'confidence': face['confidence']
            })
        
        # Draw faces on image
        image_with_faces = detector.draw_faces(image, faces)
        _, buffer = cv2.imencode('.jpg', image_with_faces)
        image_base64 = base64.b64encode(buffer).decode()
        
        return jsonify({
            'status': 'success',
            'detected_faces': len(faces),
            'faces': response_faces,
            'image': f'data:image/jpeg;base64,{image_base64}'
        })
    
    except Exception as e:
        logger.error(f"Error in detect_faces: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/recognize', methods=['POST'])
def recognize_faces():
    """Recognize faces in uploaded image"""
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Recognize faces
        image = cv2.imread(filepath)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        recognized = recognizer.recognize_faces(image=rgb_image)
        
        # Record recognitions
        for face in recognized:
            if face['name'] != 'Unknown':
                database.add_recognition_record(face['name'], face['confidence'])
        
        return jsonify({
            'status': 'success',
            'recognized_faces': len(recognized),
            'faces': recognized
        })
    
    except Exception as e:
        logger.error(f"Error in recognize_faces: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_faces():
    """Analyze faces in uploaded image"""
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Detect and analyze faces
        image = cv2.imread(filepath)
        faces = detector.detect_faces(image=image)
        
        analyses = []
        for face in faces:
            face_box = (face['x'], face['y'], face['width'], face['height'])
            analysis = analyzer.analyze(image=image, face_box=face_box)
            analyses.append(analysis)
        
        return jsonify({
            'status': 'success',
            'analyzed_faces': len(analyses),
            'analyses': analyses
        })
    
    except Exception as e:
        logger.error(f"Error in analyze_faces: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/known-faces', methods=['GET'])
def get_known_faces():
    """Get list of known faces"""
    
    try:
        faces = recognizer.get_known_faces()
        return jsonify({
            'status': 'success',
            'total': len(faces),
            'faces': faces
        })
    
    except Exception as e:
        logger.error(f"Error in get_known_faces: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/add-known-face', methods=['POST'])
def add_known_face():
    """Add a known face to database"""
    
    try:
        if 'file' not in request.files or 'name' not in request.form:
            return jsonify({'error': 'File and name required'}), 400
        
        file = request.files['file']
        name = request.form['name']
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Add to recognizer and database
        success = recognizer.add_known_face(filepath, name)
        
        if success:
            database.add_known_face(name, [0] * 128)  # Placeholder encoding
            return jsonify({
                'status': 'success',
                'message': f'Added {name} to known faces'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Failed to add face'
            }), 400
    
    except Exception as e:
        logger.error(f"Error in add_known_face: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get system statistics"""
    
    try:
        stats = database.get_statistics()
        stats['status'] = 'success'
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"Error in get_statistics: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    logger.info("Starting Face Recognition Application...")
    app.run(host='0.0.0.0', port=5000, debug=True)
