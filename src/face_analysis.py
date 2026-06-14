"""
Face Analysis Module

This module provides facial analysis capabilities including:
- Age estimation
- Gender classification
- Emotion detection
- Facial landmarks detection
"""

import cv2
import numpy as np
import logging
from typing import Dict, List
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceAnalyzer:
    """
    Face Analysis Engine
    
    Analyzes faces to extract demographic and emotional information.
    Uses pre-trained deep learning models for age, gender, and emotion.
    """
    
    def __init__(self):
        """Initialize Face Analyzer with pre-trained models"""
        
        self.age_model = None
        self.gender_model = None
        self.emotion_model = None
        self.landmark_detector = None
        
        self.age_range = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', 
                         '(38-43)', '(48-53)', '(60-100)']
        self.gender_list = ['Male', 'Female']
        self.emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
        
        self._load_models()
        logger.info("Face Analyzer initialized")
    
    def _load_models(self):
        """Load pre-trained models"""
        
        try:
            # Age and Gender models (Caffe)
            age_proto = "models/age_deploy.prototxt"
            age_model = "models/age_net.caffemodel"
            gender_proto = "models/gender_deploy.prototxt"
            gender_model = "models/gender_net.caffemodel"
            
            if os.path.exists(age_proto) and os.path.exists(age_model):
                self.age_model = cv2.dnn.readNetFromCaffe(age_proto, age_model)
                logger.info("Age model loaded successfully")
            else:
                logger.warning("Age model files not found")
            
            if os.path.exists(gender_proto) and os.path.exists(gender_model):
                self.gender_model = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)
                logger.info("Gender model loaded successfully")
            else:
                logger.warning("Gender model files not found")
            
            # Facial landmarks detector
            self._init_landmark_detector()
        
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    def _init_landmark_detector(self):
        """Initialize facial landmarks detector using OpenCV"""
        
        try:
            # Using LBF Face Landmark Detector
            landmark_model = "models/lbfmodel.yaml"
            
            if os.path.exists(landmark_model):
                self.landmark_detector = cv2.face.createFacemarkLBF()
                self.landmark_detector.loadModel(landmark_model)
                logger.info("Landmark detector loaded successfully")
            else:
                logger.warning("Landmark model not found")
        
        except Exception as e:
            logger.warning(f"Could not initialize landmark detector: {str(e)}")
    
    def analyze(self, image_path: str = None, image: np.ndarray = None, 
                face_box: tuple = None) -> Dict:
        """
        Analyze a face and extract information
        
        Args:
            image_path (str): Path to image file
            image (np.ndarray): Image array (BGR format from OpenCV)
            face_box (tuple): (x, y, width, height) of face region
        
        Returns:
            Dict: Analysis results
                {
                    'age_group': str,
                    'gender': str,
                    'emotions': list,
                    'dominant_emotion': str,
                    'landmarks': list,
                    'confidence': float
                }
        """
        
        # Load image
        if image_path is not None:
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return {}
        
        elif image is None:
            logger.error("Either image_path or image array must be provided")
            return {}
        
        try:
            result = {
                'age_group': 'Unknown',
                'gender': 'Unknown',
                'emotions': [],
                'dominant_emotion': 'Unknown',
                'landmarks': [],
                'confidence': 0.0
            }
            
            # Extract face region if not provided
            if face_box is not None:
                x, y, w, h = face_box
                face_image = image[y:y+h, x:x+w]
            else:
                face_image = image
            
            # Age and Gender detection
            if self.age_model is not None and self.gender_model is not None:
                age_gender_result = self._predict_age_gender(face_image)
                result.update(age_gender_result)
            
            # Emotion detection
            emotion_result = self._predict_emotion(face_image)
            result.update(emotion_result)
            
            # Landmarks detection
            if self.landmark_detector is not None:
                landmarks = self._detect_landmarks(image, face_box)
                result['landmarks'] = landmarks
            
            logger.info(f"Analysis complete: {result}")
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing face: {str(e)}")
            return {}
    
    def _predict_age_gender(self, face_image: np.ndarray) -> Dict:
        """Predict age and gender from face image"""
        
        result = {
            'age_group': 'Unknown',
            'gender': 'Unknown',
            'confidence': 0.0
        }
        
        try:
            # Prepare input blob
            blob = cv2.dnn.blobFromImage(
                face_image, 1.0, (227, 227),
                [104.0, 117.0, 123.0], swapRB=False
            )
            
            # Predict gender
            if self.gender_model is not None:
                self.gender_model.setInput(blob)
                gender_predictions = self.gender_model.forward()
                gender_index = np.argmax(gender_predictions[0])
                result['gender'] = self.gender_list[gender_index]
            
            # Predict age
            if self.age_model is not None:
                self.age_model.setInput(blob)
                age_predictions = self.age_model.forward()
                age_index = np.argmax(age_predictions[0])
                result['age_group'] = self.age_range[age_index]
                result['confidence'] = float(age_predictions[0][age_index])
        
        except Exception as e:
            logger.warning(f"Error in age/gender prediction: {str(e)}")
        
        return result
    
    def _predict_emotion(self, face_image: np.ndarray) -> Dict:
        """Predict emotion from face image"""
        
        result = {
            'emotions': [],
            'dominant_emotion': 'Unknown'
        }
        
        try:
            # Simple emotion detection based on facial features
            # This is a placeholder - for production, use a trained emotion model
            
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Detect face features (smile, etc.)
            smile_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_smile.xml'
            )
            smiles = smile_cascade.detectMultiScale(gray, 1.1, 5)
            
            # Simple heuristic
            if len(smiles) > 0:
                dominant_emotion = 'Happy'
                emotions_scores = {
                    'Happy': 0.8,
                    'Neutral': 0.2,
                    'Angry': 0.0,
                    'Sad': 0.0,
                    'Fearful': 0.0,
                    'Surprised': 0.0,
                    'Disgusted': 0.0
                }
            else:
                dominant_emotion = 'Neutral'
                emotions_scores = {
                    'Neutral': 0.7,
                    'Happy': 0.1,
                    'Angry': 0.1,
                    'Sad': 0.1,
                    'Fearful': 0.0,
                    'Surprised': 0.0,
                    'Disgusted': 0.0
                }
            
            result['emotions'] = emotions_scores
            result['dominant_emotion'] = dominant_emotion
        
        except Exception as e:
            logger.warning(f"Error in emotion prediction: {str(e)}")
        
        return result
    
    def _detect_landmarks(self, image: np.ndarray, face_box: tuple = None) -> List:
        """Detect facial landmarks"""
        
        landmarks = []
        
        try:
            if self.landmark_detector is None:
                return landmarks
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # If face box is provided, create face rectangle
            if face_box is not None:
                x, y, w, h = face_box
                faces = [cv2.Rect(x, y, w, h)]
            else:
                # Detect faces first
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                faces_rects = face_cascade.detectMultiScale(gray, 1.1, 5)
                faces = [cv2.Rect(x, y, w, h) for (x, y, w, h) in faces_rects]
            
            # Detect landmarks
            for face in faces:
                _, landmarks_list = self.landmark_detector.fit(gray, faces)
                if landmarks_list is not None:
                    landmarks = landmarks_list[0].tolist()
        
        except Exception as e:
            logger.warning(f"Error detecting landmarks: {str(e)}")
        
        return landmarks
    
    def draw_analysis(self, image: np.ndarray, analysis: Dict, 
                     face_box: tuple = None) -> np.ndarray:
        """
        Draw analysis results on image
        
        Args:
            image (np.ndarray): Image to draw on
            analysis (Dict): Analysis results
            face_box (tuple): (x, y, width, height) of face region
        
        Returns:
            np.ndarray: Image with drawn analysis
        """
        
        image_copy = image.copy()
        
        if face_box is not None:
            x, y, w, h = face_box
        else:
            x, y, w, h = 0, 0, image.shape[1], image.shape[0]
        
        # Draw rectangle around face
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Put text
        y_offset = y - 10
        
        # Age and Gender
        age_gender_text = f"{analysis.get('gender', 'Unknown')}, {analysis.get('age_group', 'Unknown')}"
        cv2.putText(image_copy, age_gender_text, (x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset -= 25
        
        # Emotion
        emotion_text = f"Emotion: {analysis.get('dominant_emotion', 'Unknown')}"
        cv2.putText(image_copy, emotion_text, (x, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return image_copy
