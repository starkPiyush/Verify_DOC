from flask import Flask, request, jsonify, render_template
import os
import uuid
import re
from PIL import Image
from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
from rapidfuzz import fuzz, process as rapid_process
import base64

app = Flask(__name__, template_folder='templates', static_folder='static')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

document_keywords = {
    "Aadhar Card": ["uidai", "unique identification authority of india", "government of india", "aadhaar"],
    "PAN Card": ["income tax department", "permanent account number", "pan", "govt of india"],
    "Handicap smart card": ["smart card", "disability", "handicap card", "govt issued"],
    "Birth Certificate": ["birth certificate", "date of birth", "place of birth"],
    "Bonafide Certificate": ["bonafide", "student", "institution", "studying", "enrolled"],
    "Caste certificate": ["caste", "category", "scheduled caste", "scheduled tribe", "other backward class"],
    "Current Month Salary Slip": ["salary", "monthly pay", "employee code", "basic pay"],
    "Passport and VISA": ["passport", "visa", "republic of india", "expiry date"],
    "Marksheet": ["marks", "subject", "grade", "exam", "percentage", "semester"],
    "Transgender Certificate": ["transgender", "gender identity", "third gender"]
}

DOC_MODEL_PATHS = {
    "Aadhar Card": "models/aadhaar.pt",
    "PAN Card": "models/pan_best.pt",
    "Handicap smart card": "models/handicap_smart_card.pt",
    "Birth Certificate": "models/Birth_certificatebest.pt",
    "Bonafide Certificate": None,
    "Caste certificate": "models/caste_certificate.pt",
    "Current Month Salary Slip": "models/salaryslipbest.pt",
    "Passport and VISA": "models/passport.pt",
    "Marksheet": None,
    "Transgender Certificate": "models/trans_best.pt"
}

def classify_document(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image, lang='eng')
    text = text.lower()
    scores = {}
    for doc_type, keywords in document_keywords.items():
        score = sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw.lower()) + r"\b", text))
        scores[doc_type] = score
    best_match = max(scores, key=scores.get)
    best_score = scores[best_match]
    if best_score > 0:
        return best_match, best_score
    else:
        return "Unknown Document", 0

def load_image(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == 'pdf':
        images = convert_from_path(file_path)
        return cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
    else:
        return cv2.cvtColor(np.array(Image.open(file_path).convert("RGB")), cv2.COLOR_RGB2BGR)

def extract_name_from_text(text):
    # Simple regex for name extraction, can be improved per document type
    match = re.search(r'name[:\s]+([A-Z][a-zA-Z\s]+)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # fallback: first line with more than 2 words
    for line in text.splitlines():
        if len(line.split()) > 2:
            return line.strip()
    return ""

def process_document(file_path, doc_type):
    image = load_image(file_path)
    model_path = DOC_MODEL_PATHS.get(doc_type)
    if model_path and os.path.exists(model_path):
        fields, annotated_image = run_yolo_ocr(image, model_path)
        # Try to extract name from YOLO fields
        extracted_name = ""
        for k, v in fields.items():
            if "name" in k.lower():
                extracted_name = v
                break
        if not extracted_name:
            # fallback to OCR
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')
            extracted_name = extract_name_from_text(text)
        # Format all fields as key-value pairs
        raw_text = "\n".join([f"{k}: {v}" for k, v in fields.items()])
        return {
            "extracted_name": extracted_name,
            "raw_text": raw_text,
            "fields": fields,
            "annotated_image": image_to_base64(annotated_image)
        }
    else:
        # Regex + OCR fallback
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6')
        name = extract_name_from_text(text)
        return {
            "extracted_name": name,
            "raw_text": text,
            "fields": {},
            "annotated_image": None
        }

def normalize_name(name):
    if not name:
        return ""
    # Lowercase, remove punctuation, collapse spaces
    name = name.lower()
    name = re.sub(r'[^a-z0-9 ]+', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def fuzzy_match_name(extracted_name, user_name):
    if not extracted_name or not user_name:
        return 0
    name1 = normalize_name(extracted_name)
    name2 = normalize_name(user_name)
    # Use multiple matchers for robustness
    scores = [
        fuzz.token_set_ratio(name1, name2),
        fuzz.partial_ratio(name1, name2),
        fuzz.ratio(name1, name2)
    ]
    return max(scores)

def image_to_base64(image):
    if image is None:
        return None
    _, buffer = cv2.imencode('.jpg', image)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

def run_yolo_ocr(image, model_path):
    from ultralytics import YOLO
    model = YOLO(model_path)
    results = model(image)[0]
    fields = {}
    image_drawn = image.copy()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        class_name = results.names[cls_id]
        text = pytesseract.image_to_string(Image.fromarray(image_rgb[y1:y2, x1:x2]), config='--psm 6').strip()
        fields[class_name] = text if text else "not_verified"
        cv2.rectangle(image_drawn, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_drawn, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return fields, image_drawn

@app.route('/')
def index():
    return render_template('index.html', doc_types=list(document_keywords.keys()))

@app.route('/classify_document', methods=['POST'])
def classify_document_api():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        file_id = str(uuid.uuid4())
        file_extension = file.filename.split('.')[-1].lower()
        file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.{file_extension}")
        file.save(file_path)
        doc_type, confidence = classify_document(file_path)
        os.remove(file_path)
        return jsonify({'document_type': doc_type, 'confidence': confidence})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process_documents', methods=['POST'])
def process_documents_api():
    try:
        files = request.files.getlist('files')
        user_name = request.form.get('user_name', '').strip()
        confirmed_types = request.form.getlist('confirmed_types')
        results = []
        for idx, file in enumerate(files):
            file_id = str(uuid.uuid4())
            file_extension = file.filename.split('.')[-1].lower()
            file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.{file_extension}")
            file.save(file_path)
            doc_type = confirmed_types[idx] if idx < len(confirmed_types) and confirmed_types[idx] else None
            if not doc_type:
                doc_type, confidence = classify_document(file_path)
            else:
                confidence = None
            doc_info = process_document(file_path, doc_type)
            extracted_name = doc_info["extracted_name"]
            match_score = fuzzy_match_name(extracted_name, user_name)
            match_result = "pass" if match_score >= 85 else "fail"
            results.append({
                "filename": file.filename,
                "doc_type": doc_type,
                "confidence": confidence,
                "extracted_name": extracted_name,
                "user_name": user_name,
                "match_score": match_score,
                "match_result": match_result,
                "raw_text": doc_info["raw_text"],
                "fields": doc_info.get("fields", {}),
                "annotated_image": doc_info.get("annotated_image", None)
            })
            os.remove(file_path)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)