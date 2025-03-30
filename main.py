from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models.efficientnet import EfficientNet_B0_Weights
from PIL import Image
import numpy as np
import tensorflow as tf
import base64
import io
import logging
import time

# Constants
IMG_SIZE = 224
DISEASES = ["eczema", "seborrheic_keratosis"]
SKIN_TYPES = ["dry", "normal", "oily"]

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the HybridModel for disease prediction
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = self.backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1280, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 1),
                nn.Sigmoid()
            ) for _ in range(len(DISEASES))
        ])

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).view(x.size(0), -1)
        return torch.cat([head(x) for head in self.heads], dim=1)

# Define the AcneModel for acne severity prediction
class AcneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = self.backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.regressor = nn.Sequential(
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x).view(x.size(0), -1)
        x = self.regressor(x)
        return x

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

logger.info("Loading models...")
try:
    disease_model = HybridModel().to(device)
    disease_model.load_state_dict(torch.load("./final_models/disease.pth", map_location=device))
    disease_model.eval()
    logger.info("Disease model loaded.")

    acne_model = AcneModel().to(device)
    acne_model.load_state_dict(torch.load("./final_models/acne.pth", map_location=device))
    acne_model.eval()
    logger.info("Acne model loaded.")

    skin_type_model = tf.keras.models.load_model("./final_models/skin_type.h5")
    logger.info("Skin type model loaded.")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise RuntimeError("Failed to load models.")

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0).to(device)

@app.post("/predict")
async def predict(image_base64: str, background_tasks: BackgroundTasks):
    start_time = time.time()

    try:
        logger.info("Decoding image...")
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Resize image early to avoid memory issues
        image = image.resize((IMG_SIZE, IMG_SIZE))

        logger.info("Preprocessing image...")
        image_tensor = preprocess_image(image)

        predictions = {}

        # Predict disease
        logger.info("Predicting disease...")
        with torch.no_grad():
            disease_output = disease_model(image_tensor)
        disease_probabilities = disease_output.squeeze().cpu().tolist()
        predictions["disease"] = {disease: float(prob) for disease, prob in zip(DISEASES, disease_probabilities)}

        # Predict acne severity
        logger.info("Predicting acne severity...")
        with torch.no_grad():
            acne_output = acne_model(image_tensor)
        predictions["acne"] = float(acne_output.squeeze().cpu().tolist())

        # Predict skin type (TensorFlow model)
        logger.info("Predicting skin type...")
        skin_type_image = np.array(image) / 255.0
        skin_type_image = np.expand_dims(skin_type_image, axis=0)
        
        # Run prediction asynchronously to avoid timeout
        def skin_type_task():
            skin_type_output = skin_type_model.predict_on_batch(skin_type_image)
            predictions["skin_type"] = {skin_type: float(prob) for skin_type, prob in zip(SKIN_TYPES, skin_type_output.squeeze().tolist())}
        
        background_tasks.add_task(skin_type_task)

        end_time = time.time()
        logger.info(f"Prediction completed in {end_time - start_time:.2f} seconds.")

        return {"predictions": predictions}
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

