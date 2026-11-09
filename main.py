from pathlib import Path
from datetime import datetime, timezone
import os, uuid, csv, io, json
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from PIL import Image, UnidentifiedImageError

ROOT=Path(__file__).resolve().parent
for name in ["uploads","reports","models"]: (ROOT/name).mkdir(exist_ok=True)
app=FastAPI(title="WildID: AI-Assisted Wildlife Monitoring System")
app.mount("/static",StaticFiles(directory=ROOT/"static"),name="static")
app.mount("/uploads",StaticFiles(directory=ROOT/"uploads"),name="uploads")
templates=Jinja2Templates(directory=ROOT/"templates")

collection=None
mongo_status="MongoDB not connected"
try:
    from pymongo import MongoClient
    client=MongoClient(os.getenv("MONGODB_URI","mongodb://127.0.0.1:27017/"),serverSelectionTimeoutMS=1000)
    client.admin.command("ping")
    collection=client["wildid_ai"]["animal_records"]
    mongo_status="MongoDB connected"
except Exception:
    pass


def get_ai():
    """Load CLIP zero-shot model first; fall back to MobileNet if CLIP is unavailable."""
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        model_id="openai/clip-vit-base-patch32"
        processor=CLIPProcessor.from_pretrained(model_id)
        model=CLIPModel.from_pretrained(model_id)
        model.eval()
        return {"kind":"clip","model":model,"processor":processor}, None
    except Exception:
        try:
            import torch
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
            weights=MobileNet_V3_Small_Weights.DEFAULT
            model=mobilenet_v3_small(weights=weights)
            model.eval()
            return {"kind":"mobilenet","model":model,"weights":weights}, weights
        except Exception:
            return None,None


AI_ENGINE,AI_WEIGHTS=get_ai()
AI_MODEL = AI_ENGINE.get('model') if AI_ENGINE else None


# ImageNet often predicts dog breeds rather than the word "dog".
# This vocabulary maps common ImageNet animal labels to animal groups.
ANIMAL_TERMS = {
    "dog","retriever","terrier","spaniel","hound","shepherd","poodle","pinscher",
    "chihuahua","boxer","mastiff","husky","malamute","corgi","collie","dalmatian",
    "rottweiler","beagle","pug","basset","weimaraner","whippet","greyhound","doberman",
    "akita","samoyed","papillon","chow","pekinese","shih-tzu","schipperke","affenpinscher",
    "cat","tabby","tiger_cat","siamese","persian_cat","egyptian_cat",
    "cow","ox","bull","water_buffalo","bison","zebu",
    "zebra","giraffe","lion","tiger","leopard","jaguar","cheetah","lynx",
    "wolf","fox","coyote","dingo","bear","polar_bear","grizzly","panda",
    "monkey","macaque","baboon","gorilla","orangutan","chimpanzee","lemur",
    "kangaroo","wallaby","koala","wombat","donkey","horse","zebra","elephant",
    "deer","antelope","gazelle","moose","elk","sheep","ram","goat","ibex",
    "pig","boar","wild_boar","rabbit","hare","squirrel","chipmunk","raccoon",
    "otter","seal","sea_lion","walrus","whale","dolphin","porpoise",
    "snake","python","cobra","lizard","gecko","iguana","turtle","tortoise",
    "frog","toad","newt","salamander","bird","eagle","owl","parrot","macaw",
    "penguin","chicken","hen","rooster","duck","goose","swan","flamingo",
    "ostrich","peacock","pigeon","dove","woodpecker","hummingbird","king_penguin",
    "camel","dromedary","rhinoceros","hippopotamus","platypus","armadillo",
    "weasel","badger","skunk","mole","hedgehog","porcupine","bat","beaver",
    "hamster","guinea_pig","mouse","rat","shrew","starfish","jellyfish","fish",
    "goldfish","shark","ray","seahorse","octopus","crab","lobster","shrimp"
}
SUPPORTED = ["Giraffe","Polar Bear","Donkey","Monkey","Gorilla","Jaguar","Wolf","Kangaroo","Tiger","Dog","Zebra","Lion","Cow","Cat"]

def animal_group(label):
    """Map ImageNet's detailed breed/class names to a user-friendly animal species."""
    x=label.lower().replace(" ","_").replace("-","_")
    if any(k in x for k in ["retriever","terrier","spaniel","hound","shepherd","poodle","pinscher","chihuahua","boxer","mastiff","husky","malamute","corgi","collie","dalmatian","rottweiler","beagle","pug","basset","weimaraner","whippet","greyhound","doberman","akita","samoyed","papillon","chow","pekinese","shih_tzu","schipperke","affenpinscher"]) or x=="dog":
        return "Dog"
    if any(k in x for k in ["tabby","tiger_cat","siamese","persian_cat","egyptian_cat"]) or x=="cat":
        return "Cat"
    if any(k in x for k in ["ox","water_buffalo","bison","zebu","cow","bull"]): return "Cow"
    if "polar_bear" in x: return "Polar Bear"
    if "jaguar" in x: return "Jaguar"
    if "giraffe" in x: return "Giraffe"
    if "donkey" in x: return "Donkey"
    if "gorilla" in x: return "Gorilla"
    if "kangaroo" in x or "wallaby" in x: return "Kangaroo"
    if "zebra" in x: return "Zebra"
    if "lion" in x: return "Lion"
    if "tiger" in x: return "Tiger"
    if "wolf" in x: return "Wolf"
    if any(k in x for k in ["monkey","macaque","baboon","orangutan","chimpanzee","lemur"]): return "Monkey"
    return label.replace("_"," ").title()




def _softmax(values):
    import math
    m=max(values)
    ex=[math.exp(v-m) for v in values]
    total=sum(ex) or 1.0
    return [v/total for v in ex]

def predict_image(path):
    if AI_MODEL is None:
        return {"available":False,"is_animal":None,"species":None,"confidence":None,
                "top_predictions":[],"message":"Local AI model is unavailable. Run setup_ai.bat and restart WildID."}
    try:
        img=Image.open(path).convert("RGB")
        if AI_ENGINE and AI_ENGINE.get("kind")=="clip":
            import torch
            species_prompts=[
                ("Dog","a photograph of a dog"),
                ("Cat","a photograph of a cat"),
                ("Cow","a photograph of a cow or ox"),
                ("Donkey","a photograph of a donkey or ass"),
                ("Giraffe","a photograph of a giraffe"),
                ("Polar Bear","a photograph of a polar bear"),
                ("Monkey","a photograph of a monkey"),
                ("Gorilla","a photograph of a gorilla"),
                ("Jaguar","a photograph of a jaguar"),
                ("Wolf","a photograph of a wolf"),
                ("Kangaroo","a photograph of a kangaroo"),
                ("Tiger","a photograph of a tiger"),
                ("Zebra","a photograph of a zebra"),
                ("Lion","a photograph of a lion"),
            ]
            negatives=[
                ("Human","a photograph of a human person"),
                ("Non-animal","a photograph of a building, vehicle, object, food, or landscape with no animal"),
            ]
            texts=[x[1] for x in species_prompts+negatives]
            inputs=AI_ENGINE["processor"](text=texts,images=img,return_tensors="pt",padding=True)
            with torch.inference_mode():
                outputs=AI_MODEL(**inputs)
                logits=outputs.logits_per_image[0].tolist()
            probs=_softmax(logits)
            ranked=sorted(zip(species_prompts+negatives,probs),key=lambda x:x[1],reverse=True)
            winner,score=ranked[0]
            if winner[0] in ("Human","Non-animal"):
                return {"available":True,"is_animal":False,"species":None,"confidence":round(score*100,1),
                        "top_predictions":[],"message":"This image is not belong to an animal. Please upload animal image."}
            # Use relative margin against the strongest non-animal prompt to avoid weak guesses.
            negative_score=max(p for (item,p) in zip(species_prompts+negatives,probs) if item[0] in ("Human","Non-animal"))
            if score < 0.20 or score-negative_score < 0.015:
                return {"available":True,"is_animal":None,"species":winner[0],"confidence":round(score*100,1),
                        "top_predictions":[],"message":"The image is unclear. Please upload a clearer animal photograph."}
            top=[{"label":item[0],"confidence":round(p*100,1)} for item,p in ranked[:5]]
            # CLIP's visual injury cue classifier is advisory, not a veterinary diagnosis/localizer.
            cue_prompts=[
                ("Visible wound or cut","a photograph of an animal with a visible open wound or cut on its body"),
                ("Visible bleeding","a photograph of an animal with visible blood on its body"),
                ("Bandage or dressing","a photograph of an animal with a bandage or medical dressing on a body part"),
                ("Eye abnormality","a photograph of an animal with a visibly cloudy, swollen, or injured eye"),
                ("Skin or fur abnormality","a photograph of an animal with a visible skin lesion, hair loss, or abnormal skin patch"),
                ("Swelling or limb abnormality","a photograph of an animal with a visibly swollen or abnormal limb"),
                ("No obvious visible injury","a photograph of a healthy animal with no obvious visible wound, blood, bandage, or abnormality"),
            ]
            cue_inputs=AI_ENGINE["processor"](text=[x[1] for x in cue_prompts],images=img,return_tensors="pt",padding=True)
            with torch.inference_mode():
                cue_logits=AI_MODEL(**cue_inputs).logits_per_image[0].tolist()
            cue_probs=_softmax(cue_logits)
            cue_rank=sorted(zip(cue_prompts,cue_probs),key=lambda x:x[1],reverse=True)
            cue_name,cue_score=cue_rank[0][0][0],cue_rank[0][1]
            cue_margin=cue_score-cue_rank[1][1]
            return {"available":True,"is_animal":True,"species":winner[0],"confidence":round(score*100,1),
                    "top_predictions":top,"visual_cue":cue_name,"visual_cue_confidence":round(cue_score*100,1),
                    "visual_cue_margin":round(cue_margin*100,1),"engine":"CLIP"}
        # Fallback classifier: keep existing ImageNet logic, but clearly mark its limitation.
        import torch
        tensor=AI_WEIGHTS.transforms()(img).unsqueeze(0)
        with torch.inference_mode():
            probs=AI_MODEL(tensor).softmax(1)[0]
        vals,inds=probs.topk(10)
        labels=[AI_WEIGHTS.meta["categories"][int(i)].lower() for i in inds]
        scores=[float(v) for v in vals]
        animal_candidates=[]
        for label,score in zip(labels,scores):
            normalized=label.replace(" ","_").replace("-","_")
            if any(term in normalized for term in ANIMAL_TERMS):
                animal_candidates.append((label,score,animal_group(label)))
        if not animal_candidates or animal_candidates[0][1] < 0.12:
            return {"available":True,"is_animal":False,"species":None,"confidence":None,"top_predictions":[],
                    "message":"This image is not belong to an animal. Please upload animal image."}
        raw,score,display=animal_candidates[0]
        top=[{"label":labels[i].replace("_"," ").title(),"confidence":round(scores[i]*100,1)} for i in range(min(5,len(labels)))]
        return {"available":True,"is_animal":True,"species":display,"raw_label":raw,
                "confidence":round(score*100,1),"top_predictions":top,"engine":"MobileNet fallback"}
    except Exception as e:
        return {"available":False,"is_animal":None,"species":None,"confidence":None,"top_predictions":[],
                "message":f"The local AI could not analyse this image ({type(e).__name__}). Check setup_ai.bat and restart."}

def records():
    if collection is None: return []
    try: return list(collection.find({},{"_id":0}).sort("created_at",-1).limit(500))
    except Exception: return []

@app.get("/",response_class=HTMLResponse)
async def home(request:Request):
    return templates.TemplateResponse("index.html",{"request":request})

@app.get("/api/status")
async def status():
    return {"mongodb":collection is not None,"mongodb_message":mongo_status,
            "ai_ready":AI_MODEL is not None,"ai_message":("CLIP zero-shot model ready" if AI_ENGINE and AI_ENGINE.get("kind")=="clip" else ("MobileNet fallback ready" if AI_MODEL is not None else "AI model not installed"))}

@app.get("/api/dashboard")
async def dashboard():
    data=records()
    injured=sum(1 for x in data if x.get("injury_status")=="Needs veterinary assessment")
    healthy=sum(1 for x in data if x.get("injury_status")=="No visible injury confirmed")
    species={}
    for x in data: species[x.get("species","Unknown")]=species.get(x.get("species","Unknown"),0)+1
    return {"total":len(data),"injured":injured,"healthy":healthy,"species":species,"recent":data[:6]}

@app.post("/api/analyze")
async def analyze(file:UploadFile=File(...)):
    raw=await file.read()
    try:
        image=Image.open(io.BytesIO(raw)).convert("RGB")
        if image.width<120 or image.height<120: raise ValueError()
    except Exception:
        raise HTTPException(400,"Please upload a valid JPG, PNG or WEBP image with sufficient resolution.")
    ext=(Path(file.filename or "").suffix.lower())
    if ext not in [".jpg",".jpeg",".png",".webp"]: ext=".jpg"
    case_id=uuid.uuid4().hex[:12]
    path=ROOT/"uploads"/f"{case_id}{ext}"
    image.save(path,quality=92)
    pred=predict_image(path)
    if pred["available"] and pred["is_animal"] is False:
        return {"ok":True,"is_animal":False,"message":"This image is not belong to an animal. Please upload animal image."}
    if not pred["available"]:
        return {"ok":False,"needs_setup":True,"message":pred["message"],"ai_ready":False}
    # Image-conditioned visual cue assessment from CLIP. This is a screening aid, not diagnosis.
    width,height=image.size
    image_hash=__import__("hashlib").sha256(raw).hexdigest()[:10]
    confidence=pred["confidence"]
    assessment=f"Species estimate: {pred['species']} ({confidence}% model score)."
    cue=pred.get("visual_cue","Not assessed by the fallback classifier")
    cue_conf=pred.get("visual_cue_confidence")
    cue_margin=pred.get("visual_cue_margin",0)
    if pred.get("engine")=="CLIP" and cue_conf is not None and cue_margin >= 8 and cue != "No obvious visible injury":
        injury_status="Possible visible abnormality — veterinary confirmation required"
        finding=f"The image model's strongest visual cue is: {cue.lower()} ({cue_conf}% relative score). This is an image-level screening clue, not a confirmed injury or diagnosis."
        if cue=="Visible bleeding":
            guidance=[
                "Visible blood may indicate an open wound. Contact a veterinarian or wildlife rescue professional promptly for examination.",
                "Do not wash deep wounds, remove embedded objects, or apply household chemicals. Avoid touching the animal if it is frightened or wild.",
                "Do not give human painkillers, antibiotics, or other medicines. A veterinarian must select treatment and dosage.",
                "If bleeding is heavy, seek emergency veterinary assistance immediately."
            ]
        elif cue=="Bandage or dressing":
            guidance=[
                "A bandage or dressing is visible. Keep it clean and dry if the animal is already under veterinary care, and arrange a veterinary recheck if it is wet, slipping, tight, or soiled.",
                "Do not remove or tighten a dressing unless a veterinarian instructs you to; an improperly applied bandage can restrict circulation.",
                "Do not give medicines unless prescribed for this animal by a veterinarian."
            ]
        elif cue=="Eye abnormality":
            guidance=[
                "A visible eye abnormality is suspected. Arrange prompt veterinary examination because eye problems can worsen quickly.",
                "Prevent rubbing if this can be done safely. Do not apply human eye drops, ointments, or leftover medicines.",
                "If the eye is severely swollen, bleeding, closed, or painful, seek urgent veterinary care."
            ]
        elif cue=="Swelling or limb abnormality":
            guidance=[
                "A limb or swelling abnormality is suspected. Restrict activity only if safe and arrange veterinary assessment to check for injury or fracture.",
                "Do not manipulate, massage, splint, or force the limb to move.",
                "Do not administer human painkillers or other medicines; a veterinarian must choose treatment."
            ]
        elif cue=="Skin or fur abnormality":
            guidance=[
                "A skin or coat abnormality is suspected. A veterinarian should examine it to distinguish wounds, infection, parasites, allergy, or other causes.",
                "Avoid applying antiseptics, creams, insecticides, or human medicines without veterinary advice.",
                "Avoid direct contact with affected skin and wash hands after handling animal bedding or equipment."
            ]
        else:
            guidance=[
                "A visible wound or cut is suspected. Arrange veterinary examination to assess depth, contamination, pain, and infection risk.",
                "Do not apply household chemicals, powders, or human medicines. Do not attempt to close a wound yourself.",
                "Seek urgent veterinary help for deep wounds, heavy bleeding, severe pain, or weakness."
            ]
    elif pred.get("engine")=="CLIP" and cue=="No obvious visible injury" and cue_margin >= 8:
        injury_status="No obvious visible injury cue detected"
        finding="The image model did not find a strong visible injury cue in this photograph. This does not rule out pain, internal injury, or subtle disease."
        guidance=[
            "No treatment should be started from this image alone. Observe the animal's movement, appetite, breathing, and behaviour.",
            "Arrange veterinary assessment if the animal is limping, not eating, unusually quiet, in pain, or behaving differently.",
            "Do not give medicines or supplements without veterinary advice."
        ]
    else:
        injury_status="Injury unclear — veterinary assessment recommended"
        finding="The image model could not distinguish a specific visible injury cue with enough confidence. A clearer photograph and veterinary examination may be needed."
        guidance=[
            "The photograph does not support a reliable injury-specific conclusion. Do not assume the animal is healthy or injured from this result.",
            "If you notice bleeding, difficulty breathing, inability to stand, severe weakness, or significant pain, seek urgent veterinary help.",
            "Do not give human medicines, antibiotics, painkillers, or home remedies. A veterinarian must choose any medicine and dose."
        ]
    observation=(f"Image size: {width} × {height} pixels. Image fingerprint: {image_hash}. "
                 "The visual cue model does not identify the exact body location or draw a wound mask.")
    case={"case_id":case_id,"species":pred["species"],"confidence":confidence,
          "image":f"/uploads/{path.name}","created_at":datetime.now(timezone.utc).isoformat(),
          "injury_status":injury_status,"image_specific_assessment":assessment,
          "image_specific_observation":finding+" "+observation,
          "treatment_guidance":guidance}
    if collection is not None:
        try: collection.insert_one(case.copy())
        except Exception: pass
    return {"ok":True,"is_animal":True,"case":case,
            "notice":"Species result is an AI estimate. Injury status and location are not diagnosed by this model."}

@app.get("/api/history")
async def history():
    return {"connected":collection is not None,"records":records()}

@app.delete("/api/history/{case_id}")
async def delete_record(case_id:str):
    if collection is None: raise HTTPException(503,"MongoDB is not connected.")
    result=collection.delete_one({"case_id":case_id})
    return {"deleted":result.deleted_count==1}

@app.get("/api/export.csv")
async def export_csv():
    data=records()
    output=io.StringIO()
    fields=["case_id","species","confidence","injury_status","injury_location","created_at"]
    writer=csv.DictWriter(output,fieldnames=fields,extrasaction="ignore")
    writer.writeheader(); writer.writerows(data)
    return StreamingResponse(iter([output.getvalue()]),media_type="text/csv",
        headers={"Content-Disposition":"attachment; filename=wildid_records.csv"})

@app.get("/api/report/{case_id}")
async def report(case_id:str):
    if collection is None: raise HTTPException(503,"MongoDB is not connected.")
    item=collection.find_one({"case_id":case_id},{"_id":0})
    if not item: raise HTTPException(404,"Record not found.")
    # HTML report opened in browser; user can Print > Save as PDF.
    html=f"""<html><head><title>WildID Case {case_id}</title><style>body{{font:16px Arial;max-width:760px;margin:50px auto;color:#20372c}}h1{{color:#176345}}li{{margin:12px 0}}</style></head><body><h1>WildID Case Report</h1><h2>{item.get('species','Unknown')}</h2><p>Case: {case_id}</p><p>Created: {item.get('created_at','')}</p><p>Assessment: {item.get('injury_status','')}</p><p>{item.get('injury_note','')}</p><h3>Care guidance</h3><ul>{''.join('<li>'+str(x)+'</li>' for x in item.get('care_guidance',[]))}</ul><hr><p><b>Disclaimer:</b> AI-generated information is for reference only and is not a substitute for professional veterinary advice. Consult a qualified veterinarian.</p></body></html>"""
    return HTMLResponse(html)
