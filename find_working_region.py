import os
import google.auth
from google.auth.transport.requests import Request
from vertexai.preview.generative_models import GenerativeModel
import vertexai

# --- AUTHENTICATION ---
try:
    credentials, project_id = google.auth.default()
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    print(f"✅ Authenticated as Project: {project_id}")
except Exception as e:
    print(f"❌ Auth Failed: {e}")
    exit()

# --- CONFIGURATION ---
# We will test the STABLE 2.0 model first, then the NEW 3.0 model
MODELS_TO_TEST = [
    "gemini-2.0-flash-001",  # The current standard (Fast, Stable)
    "gemini-3.0-pro-preview" # The newest release (Nov 18, 2025)
]

# Regions to check (Google rolls out new models to these specific spots first)
REGIONS_TO_TRY = [
    "us-central1",  # Iowa
    "us-east4",     # Virginia
    "us-west1",     # Oregon
    "us-west4",     # Las Vegas
    "northamerica-northeast1" # Montreal
]

print(f"🔍 Scanning {len(REGIONS_TO_TRY)} regions for {len(MODELS_TO_TEST)} models...\n")

found_working_config = False

for model_name in MODELS_TO_TEST:
    print(f"--- Testing Model: {model_name} ---")
    for region in REGIONS_TO_TRY:
        print(f"  Checking {region}...", end=" ")
        try:
            # Initialize Vertex AI for this specific region
            vertexai.init(project=project_id, location=region, credentials=credentials)
            
            # Try a simple ping
            model = GenerativeModel(model_name)
            response = model.generate_content("Hello")
            
            print("✅ ALIVE!")
            print(f"\n🎉 SUCCESS! Found a working configuration.")
            print(f"---------------------------------------------------")
            print(f"Update your .env file:")
            print(f"GOOGLE_CLOUD_LOCATION={region}")
            print(f"\nUpdate your app.py (Line 46):")
            print(f'model="{model_name}"')
            print(f"---------------------------------------------------")
            found_working_config = True
            break # Stop searching regions for this model
            
        except Exception as e:
            # If it fails, print a tiny X and move on
            print("❌")
    
    if found_working_config:
        break # Stop searching models if we found one

if not found_working_config:
    print("\n❌ Could not find ANY working model/region combo. Check your Project API quotas.")