# AI Document Verification System

A full-stack solution for secure, AI-powered government document verification.  
This project consists of a Flask-based backend (server-side) and two frontends:
- **index.html** (server-side, rendered by Flask)
- **enterprise.html** (client-side, standalone, can run on any machine in the same LAN)

## Features

- Document classification and field extraction using AI/ML and OCR
- API key generation and management
- Fuzzy matching and validation of user-provided fields
- RESTful API with interactive Swagger documentation
- Two usage modes:
  - **Server-side (index.html):** For direct interaction with the backend
  - **Client-side (enterprise.html):** For enterprise users, interacts with the backend API over the network

---

## Project Structure

```
.
├── app.py                # Flask backend (API, document processing, API key management)
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Server-side frontend (rendered by Flask)
├── enterprise.html       # Client-side frontend (standalone, uses API over LAN)
├── static/               # (Optional) Static assets
└── ...                   # Other files/models
```

---

## System Dependencies (install manually before running the app)

**For Tesseract OCR:**
```sh
sudo apt update && sudo apt install -y tesseract-ocr
```

**For PDF to image conversion (pdf2image):**
```sh
sudo apt install -y poppler-utils
```

---

## Python Dependencies

Install all Python requirements:
```sh
pip install -r requirements.txt
```

**requirements.txt:**
```
Flask
Flask-Cors
pymongo
pytesseract
opencv-python
pdf2image
Pillow
numpy
rapidfuzz
ultralytics
uuid
```

---

## Running the Server

1. **Start the Flask backend:**
   ```sh
   python app.py
   ```
   By default, the server runs at [http://127.0.0.1:5000](http://127.0.0.1:5000).

2. **Access the server-side frontend:**
   - Open [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

3. **Access the API documentation (Swagger UI):**
   - Open [http://127.0.0.1:5000/apidocs/](http://127.0.0.1:5000/apidocs/) in your browser.
   - Here you can view all API endpoints, request/response formats, and try the API interactively.

---

## Using the Client-Side Frontend (enterprise.html)

- `enterprise.html` is a standalone client and **not served by Flask**.
- You can open it directly in your browser from any machine on the same LAN as the server.
- It communicates with the Flask backend via API calls (make sure to set the correct server IP in the JS if needed).
- The "API Documentation" link in `enterprise.html` should point to the full URL of the server's Swagger UI, e.g.:
  ```
  http://<flask-server-ip>:5000/apidocs/
  ```
  Replace `<flask-server-ip>` with the actual IP address of the machine running Flask.

---

## API Key Management

- Generate an API key using the server-side frontend (`index.html`) or via the `/generate_api_key` API endpoint.
- Use this API key in the client-side frontend (`enterprise.html`) to access protected endpoints.

---

## Notes

- **index.html** is rendered by Flask and is the main server-side interface.
- **enterprise.html** is a client-side app and can be distributed to enterprise users; it does not require a Flask route.
- Both frontends interact with the same backend API, and all API documentation is available via Swagger UI.

---

## License

To be decided.
