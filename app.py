import os
from flask import Flask, request, jsonify
from supabase import create_client
import cv2
import numpy as np
import requests

app = Flask(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_BUCKET = "petri-images"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

@app.route("/healthz")
def healthz():
    return "ok", 200

def find_vertical_split_line(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    col_sum = np.sum(thresh, axis=0)
    w = image.shape[1]
    center = w // 2
    search_window = w // 8
    search_slice = col_sum[center - search_window : center + search_window]
    rel_x = np.argmax(search_slice)
    split_x = (center - search_window) + rel_x
    return split_x

@app.route("/split-petri-image", methods=["POST"])
def split_petri_image():
    data = request.json
    parent_image_url = data["parent_image_url"]
    left_obs_id = data["left_obs_id"]
    right_obs_id = data["right_obs_id"]

    # Download parent image (PNG or JPG)
    resp = requests.get(parent_image_url)
    img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Failed to load parent image"}), 400

    # Detect bold vertical black line and split
    split_x = find_vertical_split_line(img)
    left_img = img[:, :split_x]
    right_img = img[:, split_x:]

    # Encode as PNG for upload
    _, left_buf = cv2.imencode('.png', left_img)
    _, right_buf = cv2.imencode('.png', right_img)
    left_bytes = left_buf.tobytes()
    right_bytes = right_buf.tobytes()

    # Upload each crop to Supabase Storage
    left_path = f"{left_obs_id}.png"
    right_path = f"{right_obs_id}.png"
    supabase.storage().from_(SUPABASE_BUCKET).upload(left_path, left_bytes)
    supabase.storage().from_(SUPABASE_BUCKET).upload(right_path, right_bytes)

    # Public URL construction
    base_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/"
    left_url = f"{base_url}{left_path}"
    right_url = f"{base_url}{right_path}"

    # Update DB with new image URLs
    supabase.table("petri_observations").update({"image_url": left_url}).eq("observation_id", left_obs_id).execute()
    supabase.table("petri_observations").update({"image_url": right_url}).eq("observation_id", right_obs_id).execute()

    return jsonify({
        "split_x": int(split_x),
        "left_url": left_url,
        "right_url": right_url
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
