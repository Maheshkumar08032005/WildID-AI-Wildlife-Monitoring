# WildID: AI-Assisted Wildlife Monitoring System

## Start the web app
1. Extract this ZIP.
2. Install Python 3.10+ and MongoDB Community Server (MongoDB is used for saving records).
3. Double-click `run_wildid.bat`.
4. Open `http://127.0.0.1:8000`.

## Enable local image AI
1. Close the server window with Ctrl+C.
2. Double-click `setup_ai.bat` and wait. It installs CPU-only PyTorch and torchvision.
3. Start `run_wildid.bat` again.
4. On first startup, MobileNetV3 ImageNet weights may be downloaded once. After caching, classification runs locally.

## Current AI capabilities — read before demonstration
- The included MobileNetV3 model is a general ImageNet classifier. It can provide a broad animal-category estimate, but it is not a specialist wildlife classifier and can misidentify unusual species.
- It is not a validated animal-vs-human safety classifier for every image.
- **This build does not include a validated injury detection or body-part localization model.** Therefore, it does not claim an injury is present, does not show a fake red highlight, and does not prescribe a medicine.
- The result page no longer shows the permanent “Injury location: Not determined…” placeholder. It uses a transparent injury-assessment status and safe treatment/referral guidance.
- Different images are classified from their own pixel content. Identical or similar images can legitimately produce similar classifications. The app must never randomize diagnoses or treatment text to force different answers.

## MongoDB
Database: `wildid_ai`
Collection: `animal_records`

In mongosh:
```javascript
use wildid_ai
db.animal_records.countDocuments()
db.animal_records.find().sort({created_at:-1}).pretty()
```

## Pages
Dashboard, Upload & Analyse, Analysis History, Reports & Export, MongoDB Records.

## Screenshots
- `screenshots/01_dashboard.png`
- `screenshots/02_upload.png`
- `screenshots/03_analysis.png`
- `screenshots/04_history.png`
- `screenshots/05_mongodb.png`

## Safety
Do not handle wild animals or administer medicines based on an AI result. The application is a student prototype, not a veterinary diagnostic tool. Consult a qualified veterinarian or wildlife rescue professional.


## Image-specific result update
Each result now includes this upload's top classifier labels, confidence, image dimensions, and image fingerprint. This makes the explanation traceable to the actual classifier response for that image. This is not injury recognition: the included model does not detect wounds, locate injured body parts, or select treatments. A separately trained and validated veterinary injury detector/localizer is still required.


## Updated animal verification
The animal-verification step now checks the top 10 ImageNet predictions and includes common dog-breed labels (retrievers, terriers, spaniels, hounds, shepherds, etc.) and cat-breed labels. This reduces false rejection of animal photos when the model predicts a breed name rather than the generic word "dog" or "cat". A general ImageNet classifier is still not a specialist animal-vs-human model, so mistakes remain possible.

## Important injury-detection limitation
The GitHub project supplied for reference used ImageNet keyword guesses and red-pixel colour thresholds to infer injury. Those methods do not reliably detect injuries or locate wounds, so they have NOT been copied into this project. This ZIP does not contain labelled injury photographs, trained injury weights, bounding boxes or segmentation masks. Consequently, it cannot yet provide genuine image-specific injury diagnoses, red injury overlays, or treatment recommendations. To add those safely, provide a labelled animal injury dataset (images plus YOLO bounding-box or segmentation annotations) so a suitable model can be trained and evaluated.


## Updated AI engine (CLIP zero-shot screening)
- Run `setup_ai.bat` once with an internet connection. It installs Transformers and CPU PyTorch.
- On first server start, the OpenAI CLIP ViT-B/32 weights are downloaded and cached. Later image analysis runs locally.
- CLIP compares each uploaded image against text descriptions for the supported species and visible visual cues (wound/cut, bleeding, bandage, eye abnormality, skin/coat abnormality, swelling/limb abnormality, or no obvious visible injury).
- This is a prototype image-level screening feature, not a validated veterinary diagnostic system. CLIP does not draw an exact injury bounding box or segmentation mask, cannot confirm diagnosis, and may misclassify unusual images. Do not use its score as calibrated medical confidence.
- If the CLIP model cannot load, the app falls back to the original MobileNet classifier. The fallback does not perform injury screening.
- Human/non-animal images are compared against explicit human and non-animal prompts, but false positives/negatives remain possible; this is not a safety-certified classifier.
- Treatment guidance is cautious first-response advice tied to the visual cue category. It does not prescribe medicine or dosage.
