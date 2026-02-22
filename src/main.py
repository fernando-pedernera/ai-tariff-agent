import os
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv

# --- ENV LOADING / CARGA DE ENTORNO ---
load_dotenv()

app = FastAPI(title="AI Tariff Agent - Week 2", version="0.3.2")

class ProductDescription(BaseModel):
    description: str

# --- CONFIGURATION / CONFIGURACIÓN ---
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

# --- CLIENT INITIALIZATION / INICIALIZACIÓN DEL CLIENTE ---
if not OPENAI_API_KEY or not OPENAI_ENDPOINT:
    ai_client = None
    print("⚠️ Warning: Missing AI credentials.")
else:
    try:
        ai_client = AzureOpenAI(
            api_key=OPENAI_API_KEY,
            api_version="2024-12-01-preview",
            azure_endpoint=OPENAI_ENDPOINT
        )
        print("✅ Azure OpenAI connected successfully.")
    except Exception as e:
        ai_client = None
        print(f"❌ Initialization error: {e}")

# --- MOCK DATA / DATOS SIMULADOS ---
mock_regulations = {
    "6109.10": {"duty": "35%", "restriction": "Textile certificate required"},
    "8471.30": {"duty": "0%", "restriction": "Free circulation"},
    "default": {"duty": "Manual check required", "restriction": "Requires HS position analysis"}
}

# --- ENDPOINTS / PUNTOS DE CONEXIÓN ---

@app.post("/classify")
async def classify_product(product: ProductDescription):
    """Classify product using Azure OpenAI + mock regulations."""
    if not ai_client:
        raise HTTPException(status_code=500, detail="AI service not configured.")

    try:
        response = ai_client.chat.completions.create(
            model=OPENAI_DEPLOYMENT,
            messages=[{
                "role": "user", 
                "content": f"Classify this: {product.description}. Return ONLY the 6-digit HS Code. No text."
            }],
            max_tokens=10,
            temperature=0
        )
        
        suggested_code = response.choices[0].message.content.strip()
        clean_code = suggested_code.replace(".", "")[:6]
        reg_info = mock_regulations.get(clean_code, mock_regulations["default"])

        return {
            "description": product.description,
            "hs_code": suggested_code,
            "regulations": reg_info,
            "status": "AI Classified + Mock Verified"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """Service status check."""
    return {
        "status": "online",
        "ai_ready": ai_client is not None,
        "mode": "Direct Environment Variables"
    }

# --- HTML DEMO ENDPOINT / PUNTO DE DEMO HTML ---
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <body>
            <h2>Clasificador de productos</h2>
            <form action="/classify_form" method="post">
                <input name="description" type="text" placeholder="Descripción del producto"/>
                <button type="submit">Clasificar</button>
            </form>
        </body>
    </html>
    """

@app.post("/classify_form", response_class=HTMLResponse)
async def classify_form(description: str = Form(...)):
    """Form-based classification for demo purposes."""
    if not ai_client:
        return "<p>Error: AI service not configured.</p>"

    try:
        response = ai_client.chat.completions.create(
            model=OPENAI_DEPLOYMENT,
            messages=[{
                "role": "user", 
                "content": f"Classify this: {description}. Return ONLY the 6-digit HS Code. No text."
            }],
            max_tokens=10,
            temperature=0
        )
        
        suggested_code = response.choices[0].message.content.strip()
        clean_code = suggested_code.replace(".", "")[:6]
        reg_info = mock_regulations.get(clean_code, mock_regulations["default"])

        return f"""
        <html>
            <body>
                <h3>Resultado</h3>
                <p><b>Descripción:</b> {description}</p>
                <p><b>HS Code:</b> {suggested_code}</p>
                <p><b>Regulación:</b> {reg_info}</p>
                <a href="/">Volver</a>
            </body>
        </html>
        """
    except Exception as e:
        return f"<p>Error: {e}</p>"
