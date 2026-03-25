# data_collection_app.py
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import pandas as pd

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
UPLOAD_FOLDER = Path('collected_data')
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Emotions and prompts
EMOTIONS = {
    'happiness': {
        'prompt': "Write about a happy memory or something that makes you joyful",
        'text': "Today I feel happy and grateful for all the wonderful things in my life",
        'color': "#FFD700",
        'icon': "😊"
    },
    'sadness': {
        'prompt': "Write about a sad experience or something that makes you feel melancholy",
        'text': "Sometimes life feels heavy and I can't help but feel sad",
        'color': "#4682B4",
        'icon': "😢"
    },
    'anger': {
        'prompt': "Write about something that frustrates or angers you",
        'text': "I feel frustrated when things don't go as planned",
        'color': "#DC143C",
        'icon': "😠"
    },
    'fear': {
        'prompt': "Write about something that makes you feel anxious or fearful",
        'text': "There are moments when uncertainty makes me feel uneasy",
        'color': "#800080",
        'icon': "😨"
    },
    'surprise': {
        'prompt': "Write about a surprising or unexpected event",
        'text': "I was completely surprised when something unexpected happened",
        'color': "#FFA500",
        'icon': "😲"
    },
    'disgust': {
        'prompt': "Write about something that disgusts or repulses you",
        'text': "I find certain things absolutely repulsive",
        'color': "#228B22",
        'icon': "🤢"
    },
    'contempt': {
        'prompt': "Write about something you feel superior or dismissive about",
        'text': "I have little respect for things that seem trivial",
        'color': "#A0522D",
        'icon': "🙄"
    },
    'neutral': {
        'prompt': "Write a neutral statement about your day",
        'text': "Today is an ordinary day like any other",
        'color': "#808080",
        'icon': "😐"
    }
}

class DataCollector:
    def __init__(self):
        self.sessions_file = UPLOAD_FOLDER / 'sessions.json'
        self.annotations_file = UPLOAD_FOLDER / 'annotations.csv'
        self.images_folder = UPLOAD_FOLDER / 'images'
        self.images_folder.mkdir(exist_ok=True)
        
        # Load existing sessions
        if self.sessions_file.exists():
            with open(self.sessions_file, 'r') as f:
                self.sessions = json.load(f)
        else:
            self.sessions = {}
    
    def start_session(self, participant_id, age, gender, handedness, occupation):
        """Start a new data collection session"""
        session_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().isoformat()
        
        self.sessions[session_id] = {
            'session_id': session_id,
            'participant_id': participant_id,
            'age': age,
            'gender': gender,
            'handedness': handedness,
            'occupation': occupation,
            'timestamp': timestamp,
            'samples': []
        }
        
        # Save to file
        with open(self.sessions_file, 'w') as f:
            json.dump(self.sessions, f, indent=2)
        
        return session_id
    
    def add_sample(self, session_id, emotion, image_file):
        """Add a handwriting sample to the session"""
        if session_id not in self.sessions:
            return False, "Session not found"
        
        # Save image
        filename = f"{self.sessions[session_id]['participant_id']}_{emotion}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        image_path = self.images_folder / filename
        image_file.save(image_path)
        
        # Add to session
        sample = {
            'sample_id': f"{session_id}_{emotion}",
            'emotion': emotion,
            'filename': filename,
            'timestamp': datetime.now().isoformat(),
            'image_path': str(image_path)
        }
        
        self.sessions[session_id]['samples'].append(sample)
        
        # Update annotations file
        self.update_annotations()
        
        # Save sessions
        with open(self.sessions_file, 'w') as f:
            json.dump(self.sessions, f, indent=2)
        
        return True, filename
    
    def update_annotations(self):
        """Update CSV annotations file"""
        all_annotations = []
        
        for session_id, session in self.sessions.items():
            for sample in session['samples']:
                annotation = {
                    'filename': sample['filename'],
                    'participant_id': session['participant_id'],
                    'session_id': session_id,
                    'emotion': sample['emotion'],
                    'age': session['age'],
                    'gender': session['gender'],
                    'handedness': session['handedness'],
                    'occupation': session['occupation'],
                    'timestamp': sample['timestamp']
                }
                all_annotations.append(annotation)
        
        if all_annotations:
            df = pd.DataFrame(all_annotations)
            df.to_csv(self.annotations_file, index=False)
        
        return len(all_annotations)
    
    def get_stats(self):
        """Get collection statistics"""
        stats = {
            'total_sessions': len(self.sessions),
            'total_samples': 0,
            'samples_by_emotion': {e: 0 for e in EMOTIONS.keys()}
        }
        
        for session in self.sessions.values():
            stats['total_samples'] += len(session['samples'])
            for sample in session['samples']:
                stats['samples_by_emotion'][sample['emotion']] += 1
        
        return stats

# Initialize collector
collector = DataCollector()

# ============================================
# Flask Routes
# ============================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', emotions=EMOTIONS)

@app.route('/start_session', methods=['POST'])
def start_session():
    """Start a new data collection session"""
    data = request.json
    session_id = collector.start_session(
        participant_id=data['participant_id'],
        age=data['age'],
        gender=data['gender'],
        handedness=data['handedness'],
        occupation=data['occupation']
    )
    return jsonify({'session_id': session_id})

@app.route('/collect/<emotion>')
def collect_emotion(emotion):
    """Collect handwriting for specific emotion"""
    if emotion not in EMOTIONS:
        return "Emotion not found", 404
    return render_template('collect.html', emotion=emotion, 
                          emotion_data=EMOTIONS[emotion])

@app.route('/upload', methods=['POST'])
def upload():
    """Upload handwriting image"""
    session_id = request.form.get('session_id')
    emotion = request.form.get('emotion')
    image_file = request.files.get('image')
    
    if not all([session_id, emotion, image_file]):
        return jsonify({'success': False, 'error': 'Missing data'}), 400
    
    success, result = collector.add_sample(session_id, emotion, image_file)
    
    if success:
        stats = collector.get_stats()
        return jsonify({
            'success': True,
            'filename': result,
            'stats': stats
        })
    else:
        return jsonify({'success': False, 'error': result}), 400

@app.route('/stats')
def stats():
    """Get collection statistics"""
    return jsonify(collector.get_stats())

@app.route('/export')
def export():
    """Export all collected data"""
    zip_path = UPLOAD_FOLDER / 'export.zip'
    import zipfile
    with zipfile.ZipFile(zip_path, 'w') as zf:
        # Add images
        for img_path in collector.images_folder.glob('*.png'):
            zf.write(img_path, f'images/{img_path.name}')
        
        # Add annotations
        if collector.annotations_file.exists():
            zf.write(collector.annotations_file, 'annotations.csv')
        
        # Add sessions
        if collector.sessions_file.exists():
            zf.write(collector.sessions_file, 'sessions.json')
    
    return send_file(zip_path, as_attachment=True)

@app.route('/dashboard')
def dashboard():
    """Admin dashboard"""
    stats = collector.get_stats()
    return render_template('dashboard.html', stats=stats, emotions=EMOTIONS)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)