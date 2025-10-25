# Exercise 4: Model Deployment with FastAPI

## Objective
Understand how ML models are deployed in production via REST APIs.

## Red Team Context
Production ML models are deployed via APIs. Understanding this attack surface is critical for:
- Input validation bypasses
- Rate limiting evasion  
- Model endpoint enumeration
- API parameter manipulation

## Exercise Steps

### Step 1: Review the Provided serve_model.py

The course provides a FastAPI template at `../serve_model.py`. Review it to understand:
- How the model is loaded
- API endpoint structure
- Input/output formats
- Error handling

### Step 2: Start the Model Server

```bash
# Navigate to course root directory
cd /path/to/ai-red-team-course

# Start the FastAPI server
python serve_model.py
```

Expected output:
```
Model loaded.
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 3: Test the API

Open a new terminal and test the API:

```python
# Create a test script: test_api.py
import requests
import io
from PIL import Image
import numpy as np

# API endpoint
url = "http://localhost:8000/predict"

# Create a dummy MNIST image (28x28 grayscale)
# In real scenario, you'd load an actual image
dummy_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
img = Image.fromarray(dummy_image)

# Convert to bytes
img_bytes = io.BytesIO()
img.save(img_bytes, format='PNG')
img_bytes = img_bytes.getvalue()

# Send request
response = requests.post(url, data={'image_bytes': img_bytes})
print(response.json())
```

### Step 4: Document API Behavior

Document your findings:

```markdown
# Model Deployment Notes

## API Endpoint Analysis

### Endpoint: POST /predict
- **URL**: http://localhost:8000/predict
- **Purpose**: Predict digit from image
- **Input**: image_bytes (PNG/JPEG encoded image)
- **Output**: JSON with prediction

### Input Requirements
- Image format: PNG or JPEG
- Size: 28x28 (MNIST standard)
- Color: Grayscale
- Data type: Binary encoded

### Output Format
```json
{
  "prediction": 5,
  "confidence": 0.95
}
```

### Error Handling
- Missing image: Returns error message
- Invalid format: Returns error message
- Model not loaded: Returns error

## Attack Surface Analysis

### Potential Vulnerabilities
1. **Input Validation Bypass**
   - What if we send non-28x28 images?
   - What if we send colored images?
   - What happens with malformed data?

2. **Resource Exhaustion**
   - Can we send very large images?
   - Rate limiting implemented?
   - Batch processing possible?

3. **Information Disclosure**
   - Error messages reveal model architecture?
   - Stack traces exposed?
   - Model version information?

### Testing Commands
```

### Step 5: Test Attack Scenarios

Try these tests:

```bash
# Test 1: Send invalid image
curl -X POST http://localhost:8000/predict \
  -F "image_bytes=@invalid_file.txt"

# Test 2: Send oversized image  
# Create large image and send

# Test 3: Check for rate limiting
for i in {1..100}; do
  curl -X POST http://localhost:8000/predict \
    -F "image_bytes=@test_image.png"
done

# Test 4: Check error messages
curl -X POST http://localhost:8000/predict
```

### Step 6: Document Findings

Create `model_deployment_notes.md` with:

1. **API Documentation**
   - Endpoints discovered
   - Input/output formats
   - Authentication requirements (if any)

2. **Security Observations**
   - Input validation strengths/weaknesses
   - Error message handling
   - Rate limiting presence
   - Authentication mechanisms

3. **Attack Scenarios**
   - Potential evasion techniques
   - Input manipulation vectors
   - Denial of service possibilities

## Deliverable

Create `week-1/model_deployment_notes.md` with your findings.

Include:
- API documentation
- Security assessment
- Test results
- Recommendations for securing the deployment

## Key Takeaways

1. **Model Deployment Architecture**
   - Models deployed as REST APIs
   - Input validation is critical
   - Error handling must not leak info

2. **Attack Surface**
   - API endpoint enumeration
   - Input validation testing
   - Fuzzing model inputs
   - Rate limiting assessment

3. **Connection to Red Team Work**
   - Production models = API endpoints
   - Input manipulation = adversarial samples
   - API security + ML security = full assessment

## Next Steps

This deployment knowledge enables:
- Week 3: Testing evasion attacks via API
- Week 5: Prompt injection through LLM APIs
- Professional engagements: Real model deployments in production

## References

- FastAPI Documentation: https://fastapi.tiangolo.com/
- REST API Security: OWASP API Security Top 10
- API Penetration Testing methodology
