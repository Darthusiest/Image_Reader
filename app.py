from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import traceback
from NN_Update_Beta import NeuralNetwork

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
        
        # Get language parameter
        lang = request.form.get('lang', '').strip()
        
        # Save file securely
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        try:
            file.save(file_path)
            print(f"[INFO] File saved: {file_path}")
        except Exception as e:
            return jsonify({'error': f'Failed to save file: {str(e)}'}), 500
        
        # Process the file
        try:
            nn = NeuralNetwork()
            extracted_text = nn.extract_image(file_path)
            
            if not extracted_text:
                return jsonify({'error': 'No text could be extracted from the file'}), 400
            
            # Translate if language is specified
            if lang:
                try:
                    translated_text = nn.translate_text(extracted_text, lang)
                    result = {
                        'original_text': extracted_text,
                        'translated_text': translated_text,
                        'language': lang
                    }
                except Exception as e:
                    print(f"[ERROR] Translation failed: {e}")
                    result = {
                        'original_text': extracted_text,
                        'error': f'Translation failed: {str(e)}',
                        'language': lang
                    }
            else:
                result = {
                    'original_text': extracted_text,
                    'translated_text': None,
                    'language': None
                }
            
            return jsonify(result)
            
        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
        
        finally:
            # Clean up uploaded file
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"[INFO] Cleaned up: {file_path}")
            except Exception as e:
                print(f"[WARNING] Failed to clean up file: {e}")
    
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'OCR Translator is running'})

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting OCR Translator Web Application...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    app.run(debug=True, host='0.0.0.0', port=5000)
