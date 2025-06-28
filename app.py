import os
from flask import Flask, request, jsonify
from supabase import create_client, Client
import cv2
import numpy as np

app = Flask(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "images")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route("/healthz")
def healthz():
    return "ok", 200

@app.route("/petri-upload", methods=["POST"])
def petri_upload():
    event = request.json
    file_path = event["record"]["name"]  # e.g., "uploads/IMG_2950.jpg"
    print(f"Processing image: {file_path}")

    # Download image from Supabase Storage
    resp = supabase.storage().from_(SUPABASE_BUCKET).download(file_path)
    img_bytes = resp.read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Image load failed"}), 400

    h, w = img.shape[:2]
    crops = [img[:, :w // 2], img[:, w // 2:]]  # Left/right split

    # For demonstration: create 2 observation records for each crop
    created = []
    for idx, crop in enumerate(crops):
        _, buf = cv2.imencode(".jpg", crop)
        crop_bytes = buf.tobytes()
        crop_name = f"{os.path.splitext(file_path)[0]}_petri{idx+1}.jpg"
        supabase.storage().from_(SUPABASE_BUCKET).upload(crop_name, crop_bytes)
        # Insert into petri_observations - adjust field names as needed for your schema!
        obs = supabase.table("petri_observations").insert({
            "image_url": crop_name,
            "is_imputed": False,
            "created_by": "auto-opencv"
            # add other fields like petri_code, program_id, etc. if required!
        }).execute()
        created.append(obs)

    return jsonify({"message": "2 petri crops created", "details": [str(c) for c in created]}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
