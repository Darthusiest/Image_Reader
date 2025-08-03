# OCR Translator Web Application

A web-based OCR (Optical Character Recognition) and translation tool that can extract text from images and PDFs, then translate it to various languages.

## Features

- **OCR Processing**: Extract text from images (PNG, JPG, JPEG, GIF, BMP, TIFF) and PDF files
- **Translation**: Translate extracted text to multiple languages using Google Translate
- **Web Interface**: User-friendly drag-and-drop interface
- **Error Handling**: Robust error handling and user feedback
- **File Cleanup**: Automatic cleanup of uploaded files for security

## Prerequisites

### Required Software

1. **Python 3.7+**
2. **Tesseract OCR** - Download and install from: https://github.com/UB-Mannheim/tesseract/wiki
   - Default installation path: `C:\Tesseract-OCR\tesseract.exe`
3. **Poppler** (for PDF processing) - Download from: https://github.com/oschwartz10612/poppler-windows/releases
   - Extract to `C:\poppler\` or another location

### Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install flask pytesseract numpy Pillow opencv-python deep-translator pdf2image Werkzeug
```

## Installation

1. **Clone or download** this project to your local machine
2. **Install Tesseract OCR**:
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Install to default location: `C:\Tesseract-OCR\`
   - Add to PATH environment variable

3. **Install Poppler** (for PDF support):
   - Download from: https://github.com/oschwartz10612/poppler-windows/releases
   - Extract to `C:\poppler\`
   - Add `C:\poppler\bin` to PATH environment variable

4. **Install Python dependencies**:
   ```bash
   cd "Neural Networks"
   pip install -r requirements.txt
   ```

## Running the Application

### Method 1: Using run.py (Recommended)
```bash
cd "Neural Networks"
python run.py
```

### Method 2: Direct Flask execution
```bash
cd "Neural Networks"
python app.py
```

### Method 3: Using Flask CLI
```bash
cd "Neural Networks"
set FLASK_APP=app.py
set FLASK_ENV=development
flask run
```

## Usage

1. **Start the application** using one of the methods above
2. **Open your web browser** and go to: `http://localhost:5000`
3. **Upload a file**:
   - Drag and drop an image or PDF file onto the upload area
   - Or click to select a file from your computer
4. **Choose translation language** (optional):
   - Select a language from the dropdown to translate the extracted text
   - Or leave as "No Translation" to only extract text
5. **Click Submit** to process the file
6. **View results**:
   - Extracted text will appear in the first text area
   - Translated text (if requested) will appear in the second text area

## Supported File Types

- **Images**: PNG, JPG, JPEG, GIF, BMP, TIFF
- **Documents**: PDF
- **Maximum file size**: 16MB

## Supported Languages

- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Russian (ru)
- Japanese (ja)
- Korean (ko)
- Chinese (zh)
- Arabic (ar)
- Hindi (hi)
- Auto-detect (auto)

## Troubleshooting

### Common Issues

1. **"Tesseract not found" error**:
   - Ensure Tesseract is installed at `C:\Tesseract-OCR\tesseract.exe`
   - Or update the path in `NN_Update_Beta.py` line 12

2. **PDF processing fails**:
   - Ensure Poppler is installed and in PATH
   - Check that `C:\poppler\bin` is in your system PATH

3. **Translation fails**:
   - Check your internet connection
   - The Google Translate API may have rate limits

4. **File upload fails**:
   - Check file size (max 16MB)
   - Ensure file type is supported
   - Check that the `uploads` folder exists and is writable

### Debug Mode

The application runs in debug mode by default. Check the console output for detailed error messages.

## File Structure

```
Neural Networks/
├── app.py                 # Flask web application
├── run.py                 # Application launcher
├── NN_Update_Beta.py     # OCR and translation logic
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   └── index.html        # Web interface
└── uploads/              # Temporary file storage (auto-created)
```

## API Endpoints

- `GET /` - Main web interface
- `POST /extract` - Process uploaded file
- `GET /health` - Health check endpoint

## Security Features

- File type validation
- File size limits (16MB)
- Secure filename handling
- Automatic file cleanup
- Input sanitization

## Development

To modify the application:

1. **Frontend**: Edit `templates/index.html`
2. **Backend Logic**: Edit `NN_Update_Beta.py`
3. **Web Routes**: Edit `app.py`
4. **Dependencies**: Update `requirements.txt`

## License

This project is for educational purposes. Please respect the terms of service for any third-party APIs used (Google Translate, etc.). 
