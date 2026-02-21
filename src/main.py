import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AzureOpenAI
from dotenv import load_dotenv

# --- ENV LOADING / CARGA DE ENTORNO ---
# Loads .env locally; uses Azure App Service Configuration in the cloud.
# Carga .env en local; usa la Configuración de App Service en la nube.
load_dotenv()

app = FastAPI(title="AI Tariff Agent - Week 2", version="0.3.2")

class ProductDescription(BaseModel):
    description: str

# --- CONFIGURATION / CONFIGURACIÓN ---
# Fetching direct environment variables for Azure OpenAI.
# Obteniendo variables de entorno directas para Azure OpenAI.
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
# Mock customs database for Week 2 logic validation.
# Base de datos simulada para validar la lógica en la Semana 2.
mock_regulations = {
    "6109.10": {"duty": "35%", "restriction": "Textile certificate required"},
    "8471.30": {"duty": "0%", "restriction": "Free circulation"},
    "default": {"duty": "Manual check required", "restriction": "Requires HS position analysis"}
}

# --- ENDPOINTS / PUNTOS DE CONEXIÓN ---

@app.post("/classify")
async def classify_product(product: ProductDescription):
    """
    Uses GPT-4o-mini to suggest an HS Code and filters it via mock regulations.
    Usa GPT-4o-mini para sugerir un código arancelario y lo filtra vía regulaciones mock.
    """
    if not ai_client:
        raise HTTPException(status_code=500, detail="AI service not configured.")

    try:
        # Prompt optimized for deterministic 6-digit output.
        # Prompt optimizado para una salida determinista de 6 dígitos.
        response = ai_client.chat.completions.create(
            model=OPENAI_DEPLOYMENT,
            messages=[{
                "role": "user", 
                "content": f"Classify this: {product.description}. Return ONLY the 6-digit HS Code. No text."
            }],
            max_tokens=10,
            temperature=0  # Zero temperature ensures consistent expert reasoning.
        )
        
        suggested_code = response.choices[0].message.content.strip()
        
        # Cleanup and Mock Lookup / Limpieza y Búsqueda en Mock
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
    """Service status check / Verificación de estado del servicio."""
    return {
        "status": "online",
        "ai_ready": ai_client is not None,
        "mode": "Direct Environment Variables"
    }