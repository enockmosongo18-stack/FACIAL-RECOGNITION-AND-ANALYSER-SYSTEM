"""
API Documentation

Complete REST API documentation for the Facial Recognition System.
"""

# FACIAL RECOGNITION SYSTEM - API DOCUMENTATION

## Base URL
```
http://localhost:5000
```

## Authentication
Currently no authentication is required (for development).
For production, implement proper authentication.

---

## Endpoints

### 1. Health Check

**GET** `/api/health`

Check if the system is running and all services are active.

**Response:**
```json
{
    "status": "healthy",
    "version": "1.0.0",
    "services": {
        "face_detection": "active",
        "face_recognition": "active",
        "face_analysis": "active",
        "database": "active"
    }
}
```

---

### 2. Detect Faces

**POST** `/api/detect`

Detect all faces in an image.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Parameter: `file` (image file, required)

**cURL Example:**
```bash
curl -X POST -F "file=@photo.jpg" http://localhost:5000/api/detect
```

**Response:**
```json
{
    "status": "success",
    "detected_faces": 2,
    "faces": [
        {
            "id": 0,
            "x": 100,
            "y": 150,
            "width": 120,
            "height": 150,
            "confidence": 0.98
        },
        {
            "id": 1,
            "x": 350,
            "y": 120,
            "width": 130,
            "height": 160,
            "confidence": 0.96
        }
    ],
    "image": "data:image/jpeg;base64,..."
}
```

**Error Response:**
```json
{
    "error": "File type not allowed"
}
```

---

### 3. Recognize Faces

**POST** `/api/recognize`

Recognize faces in an image against known faces database.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Parameter: `file` (image file, required)

**cURL Example:**
```bash
curl -X POST -F "file=@photo.jpg" http://localhost:5000/api/recognize
```

**Response:**
```json
{
    "status": "success",
    "recognized_faces": 2,
    "faces": [
        {
            "name": "John Doe",
            "confidence": 0.92,
            "x": 100,
            "y": 150,
            "width": 120,
            "height": 150
        },
        {
            "name": "Unknown",
            "confidence": 0.0,
            "x": 350,
            "y": 120,
            "width": 130,
            "height": 160
        }
    ]
}
```

---

### 4. Analyze Faces

**POST** `/api/analyze`

Perform demographic and emotional analysis on faces.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Parameter: `file` (image file, required)

**cURL Example:**
```bash
curl -X POST -F "file=@photo.jpg" http://localhost:5000/api/analyze
```

**Response:**
```json
{
    "status": "success",
    "analyzed_faces": 1,
    "analyses": [
        {
            "age_group": "(25-32)",
            "gender": "Male",
            "emotions": {
                "Happy": 0.8,
                "Neutral": 0.1,
                "Angry": 0.05,
                "Sad": 0.05,
                "Fearful": 0.0,
                "Surprised": 0.0,
                "Disgusted": 0.0
            },
            "dominant_emotion": "Happy",
            "confidence": 0.85,
            "landmarks": [[100, 120], [200, 125], ...]
        }
    ]
}
```

---

### 5. Get Known Faces

**GET** `/api/known-faces`

Get list of all known faces in the database.

**cURL Example:**
```bash
curl http://localhost:5000/api/known-faces
```

**Response:**
```json
{
    "status": "success",
    "total": 5,
    "faces": [
        "John Doe",
        "Jane Smith",
        "Bob Johnson",
        "Alice Brown",
        "Charlie Davis"
    ]
}
```

---

### 6. Add Known Face

**POST** `/api/add-known-face`

Add a new face to the known faces database.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Parameter: `file` (face image, required)
- Parameter: `name` (person's name, required)

**cURL Example:**
```bash
curl -X POST -F "file=@john.jpg" -F "name=John Doe" http://localhost:5000/api/add-known-face
```

**Response:**
```json
{
    "status": "success",
    "message": "Added John Doe to known faces"
}
```

**Error Response:**
```json
{
    "status": "error",
    "message": "Failed to add face"
}
```

---

### 7. Get Statistics

**GET** `/api/statistics`

Get system statistics.

**cURL Example:**
```bash
curl http://localhost:5000/api/statistics
```

**Response:**
```json
{
    "status": "success",
    "total_known_faces": 5,
    "total_recognitions": 42,
    "total_analyses": 128
}
```

---

## Error Handling

### Common Error Codes

| Code | Error | Solution |
|------|-------|----------|
| 400 | No file provided | Include `file` parameter in request |
| 400 | File type not allowed | Use jpg, jpeg, png, or bmp format |
| 404 | Not found | Check endpoint URL |
| 500 | Internal server error | Check server logs |

### Error Response Format

```json
{
    "error": "Error message describing what went wrong"
}
```

---

## Rate Limiting

Currently not implemented. For production, implement:
- Rate limiting per IP
- API key authentication
- Request throttling

---

## Data Formats

### Image Formats
- Supported: JPEG, PNG, BMP, TIFF
- Max size: 16MB
- Recommended: 640x480 or larger

### Confidence Scores
- Range: 0.0 - 1.0
- 0.0 = No match
- 1.0 = Perfect match
- Threshold: 0.6 (configurable)

### Age Groups
```
(0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)
```

### Emotions
```
Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised
```

---

## Example Workflows

### Workflow 1: Detect and Analyze

```bash
# 1. Upload image and detect faces
curl -X POST -F "file=@photo.jpg" http://localhost:5000/api/detect

# 2. Get analysis for same image
curl -X POST -F "file=@photo.jpg" http://localhost:5000/api/analyze
```

### Workflow 2: Register and Recognize

```bash
# 1. Register known person
curl -X POST -F "file=@john.jpg" -F "name=John Doe" \
  http://localhost:5000/api/add-known-face

# 2. Get known faces list
curl http://localhost:5000/api/known-faces

# 3. Recognize faces in new image
curl -X POST -F "file=@group_photo.jpg" http://localhost:5000/api/recognize
```

### Workflow 3: Complete Analysis

```bash
# 1. Detect faces
curl -X POST -F "file=@photo.jpg" http://localhost:5000/api/detect

# 2. Recognize who they are
curl -X POST -F "file=@photo.jpg" http://localhost:5000/api/recognize

# 3. Analyze their features
curl -X POST -F "file=@photo.jpg" http://localhost:5000/api/analyze

# 4. Check statistics
curl http://localhost:5000/api/statistics
```

---

## Best Practices

1. **Error Handling**: Always check `status` field in response
2. **Image Quality**: Use well-lit images for better accuracy
3. **File Optimization**: Compress images before uploading
4. **Caching**: Cache responses when possible
5. **Logging**: Log all API calls for audit trail
6. **Privacy**: Never store unnecessary face data

---

## Troubleshooting

### Issue: 500 Internal Server Error
- Check server logs
- Verify image format
- Ensure sufficient disk space

### Issue: No faces detected
- Improve image lighting
- Use higher resolution images
- Try different angles

### Issue: Low confidence scores
- Use better quality images
- Ensure proper face orientation
- Adjust distance threshold in config

---

## Future Enhancements

- [ ] WebSocket support for real-time processing
- [ ] Batch API for processing multiple images
- [ ] API key authentication
- [ ] Rate limiting
- [ ] Response caching
- [ ] Async processing
- [ ] Download results as JSON/CSV
