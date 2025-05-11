# import os
# from flask import Flask, render_template, request
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration
# import torch

# app = Flask(__name__)
# UPLOAD_FOLDER = "static/uploaded"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# # Load pre-trained BLIP model
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# @app.route("/", methods=["GET", "POST"])
# def index():
#     caption = None
#     image_path = None

#     if request.method == "POST":
#         file = request.files.get("image")
#         if file and file.filename:
#             path = os.path.join(UPLOAD_FOLDER, file.filename)
#             file.save(path)

#             image = Image.open(path).convert("RGB")
#             inputs = processor(image, return_tensors="pt")

#             with torch.no_grad():
#                 output = model.generate(**inputs)
#                 caption = processor.decode(output[0], skip_special_tokens=True)

#             image_path = path

#     return render_template("index.html", caption=caption, image_path=image_path)

# if __name__ == "__main__":
#     app.run(debug=True)

import os
os.environ['HF_HOME'] = "D:/huggingface_cache"
import io
import base64
from flask import Flask, render_template, request
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

app = Flask(__name__)

# Load BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    image_data = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            # Open image from memory
            image = Image.open(io.BytesIO(file.read())).convert("RGB")

            # Generate caption
            inputs = processor(image, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(**inputs)
                caption = processor.decode(output[0], skip_special_tokens=True)

            # Convert image to base64 for display
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_data = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return render_template("index.html", caption=caption, image_data=image_data)

if __name__ == "__main__":
    app.run(debug=True)
