from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import tempfile
import traceback
from werkzeug.utils import secure_filename

# Import the NeuralNetwork class
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from NN_Update_Beta import NeuralNetwork

app = Flask(__name__, 
            static_folder='../static',
            template_folder='../templates')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract():
    """Handle file upload and processing"""
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
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            file.save(tmp_file.name)
            file_path = tmp_file.name
        
        try:
            # Process the file
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
            # Clean up temporary file
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
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

# Vercel requires this for serverless deployment
app.debug = True

# For local development
if __name__ == '__main__':
    app.run(debug=True) 