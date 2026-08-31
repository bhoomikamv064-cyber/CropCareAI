from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

from class_names import class_names
from disease_info import disease_data


app = Flask(__name__)


# =========================================================
# LOAD AI MODELS
# =========================================================

# Existing PlantVillage disease model
model = tf.keras.models.load_model("model.h5")

# New leaf / non-leaf detector
leaf_detector = tf.keras.models.load_model("leaf_detector.h5")


# =========================================================
# LANGUAGE TEXT
# =========================================================

ui_text = {

    "en": {
        "title": "Crop Disease Detection",
        "upload": "Upload Crop Image",
        "detect": "Detect Disease",
        "back": "Go Back",

        "invalid":
            "Invalid image. Please upload a clear crop leaf image.",

        "noleaf":
            "Leaf could not be detected. Please upload a clear crop leaf image.",

        "cause": "Cause",
        "treatment": "Treatment",
        "pesticide": "Pesticide",
        "prevention": "Prevention",
        "confidence": "Confidence",

        "pesticide_photo": "Pesticide Photo",

        "listen": "🔊 Listen",
        "stop": "⏹ Stop",

        "safety":
            "Use pesticides only according to the product label and local agricultural guidance."
    },


    "kn": {
        "title": "ಬೆಳೆ ರೋಗ ಪತ್ತೆ",
        "upload": "ಬೆಳೆ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
        "detect": "ರೋಗ ಪತ್ತೆಮಾಡಿ",
        "back": "ಹಿಂತಿರುಗಿ",

        "invalid":
            "ತಪ್ಪಾದ ಚಿತ್ರ. ದಯವಿಟ್ಟು ಸ್ಪಷ್ಟವಾದ ಬೆಳೆ ಎಲೆ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",

        "noleaf":
            "ಎಲೆಯನ್ನು ಪತ್ತೆಹಚ್ಚಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಸ್ಪಷ್ಟವಾದ ಬೆಳೆ ಎಲೆಯ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ.",

        "cause": "ಕಾರಣ",
        "treatment": "ಚಿಕಿತ್ಸೆ",
        "pesticide": "ಕೀಟನಾಶಕ",
        "prevention": "ತಡೆಗಟ್ಟುವಿಕೆ",
        "confidence": "ನಿಖರತೆ",

        "pesticide_photo": "ಕೀಟನಾಶಕದ ಚಿತ್ರ",

        "listen": "🔊 ಕೇಳಿ",
        "stop": "⏹ ನಿಲ್ಲಿಸಿ",

        "safety":
            "ಕೀಟನಾಶಕಗಳನ್ನು ಉತ್ಪನ್ನದ ಲೇಬಲ್ ಮತ್ತು ಸ್ಥಳೀಯ ಕೃಷಿ ಮಾರ್ಗದರ್ಶನದಂತೆ ಮಾತ್ರ ಬಳಸಿ."
    }
}


# =========================================================
# PESTICIDE IMAGE MAPPING
# =========================================================

pesticide_images = {

    "Copper oxychloride":
        "copper_oxychloride.jpg",

    "Mancozeb":
        "mancozeb.jpg",

    "Metalaxyl":
        "metalaxyl.jpg",

    "Copper fungicide":
        "copper_fungicide.jpg",

    "Ridomil Gold":
        "ridomil_gold.jpg",

    "Chlorothalonil":
        "chlorothalonil.jpg",

    "Abamectin":
        "abamectin.jpg",

    "Imidacloprid":
        "imidacloprid.jpg",

    "None":
        None,

    "N/A":
        None
}


# =========================================================
# DISEASE MODEL PREPROCESSING
# =========================================================

def preprocess(image):

    image = image.resize((224, 224))

    image = np.array(image).astype("float32") / 255.0

    image = image.reshape(1, 224, 224, 3)

    return image


# =========================================================
# LEAF DETECTOR PREPROCESSING
# =========================================================

def preprocess_leaf_detector(image):

    image = image.resize((224, 224))

    image = np.array(image).astype("float32") / 255.0

    image = image.reshape(1, 224, 224, 3)

    return image


# =========================================================
# LEAF / NON-LEAF DETECTION
# =========================================================

def detect_leaf(image):

    try:

        processed = preprocess_leaf_detector(image)

        prediction = leaf_detector.predict(
            processed,
            verbose=0
        )

        # IMPORTANT:
        # Training mapping:
        # leaf = 0
        # non_leaf = 1
        #
        # Sigmoid output represents probability of class 1.
        # Therefore:
        # prediction close to 0 = leaf
        # prediction close to 1 = non-leaf

        non_leaf_probability = float(prediction[0][0])

        leaf_probability = 1.0 - non_leaf_probability

        print("--------------------------------")
        print(
            "Leaf probability:",
            round(leaf_probability, 4)
        )
        print(
            "Non-leaf probability:",
            round(non_leaf_probability, 4)
        )
        print("--------------------------------")

        # -------------------------------------------------
        # DECISION
        # -------------------------------------------------
        #
        # If non-leaf probability >= 0.50,
        # reject the image.
        #
        # Otherwise it is considered a leaf.
        # -------------------------------------------------

        if non_leaf_probability >= 0.50:

            return False, leaf_probability, non_leaf_probability

        return True, leaf_probability, non_leaf_probability

    except Exception as e:

        print("Leaf detector error:", e)

        return False, 0.0, 0.0


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    lang = request.args.get("lang", "en")

    if lang not in ui_text:
        lang = "en"

    return render_template(
        "index.html",
        lang=lang,
        ui=ui_text[lang]
    )


# =========================================================
# UPLOAD PAGE
# =========================================================

@app.route("/upload")
def upload():

    lang = request.args.get("lang", "en")

    if lang not in ui_text:
        lang = "en"

    return render_template(
        "upload.html",
        lang=lang,
        ui=ui_text[lang]
    )


# =========================================================
# PREDICTION
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # -----------------------------------------------------
    # GET LANGUAGE
    # -----------------------------------------------------

    lang = request.form.get("lang", "en")

    if lang not in ui_text:
        lang = "en"


    # -----------------------------------------------------
    # CHECK IMAGE EXISTS
    # -----------------------------------------------------

    if "image" not in request.files:

        return render_template(
            "result.html",
            error=ui_text[lang]["invalid"],
            lang=lang,
            ui=ui_text[lang]
        )


    file = request.files["image"]


    if file.filename == "":

        return render_template(
            "result.html",
            error=ui_text[lang]["invalid"],
            lang=lang,
            ui=ui_text[lang]
        )


    # -----------------------------------------------------
    # OPEN IMAGE
    # -----------------------------------------------------

    try:

        image = Image.open(file).convert("RGB")

    except Exception as e:

        print("Image loading error:", e)

        return render_template(
            "result.html",
            error=ui_text[lang]["invalid"],
            lang=lang,
            ui=ui_text[lang]
        )


    # =====================================================
    # STEP 1: LEAF / NON-LEAF DETECTION
    # =====================================================

    is_leaf, leaf_probability, non_leaf_probability = detect_leaf(
        image
    )


    # =====================================================
    # REJECT NON-LEAF IMAGE
    # =====================================================

    if not is_leaf:

        print("--------------------------------")
        print("RESULT: NON-LEAF IMAGE")
        print("--------------------------------")

        return render_template(
            "result.html",
            error=ui_text[lang]["noleaf"],
            lang=lang,
            ui=ui_text[lang]
        )


    # =====================================================
    # LEAF DETECTED
    # =====================================================

    print("--------------------------------")
    print("RESULT: LEAF IMAGE")
    print("--------------------------------")


    # =====================================================
    # STEP 2: DISEASE MODEL PREPROCESSING
    # =====================================================

    try:

        processed = preprocess(image)

    except Exception as e:

        print("Preprocessing error:", e)

        return render_template(
            "result.html",
            error=ui_text[lang]["invalid"],
            lang=lang,
            ui=ui_text[lang]
        )


    # =====================================================
    # STEP 3: DISEASE MODEL PREDICTION
    # =====================================================

    try:

        pred = model.predict(
            processed,
            verbose=0
        )

    except Exception as e:

        print("Prediction error:", e)

        return render_template(
            "result.html",
            error="Model prediction failed.",
            lang=lang,
            ui=ui_text[lang]
        )


    # =====================================================
    # CONFIDENCE
    # =====================================================

    confidence = float(np.max(pred))

    predicted_index = int(np.argmax(pred))


    # =====================================================
    # CLASS NAME
    # =====================================================

    try:

        result = class_names[predicted_index]

    except (IndexError, TypeError):

        result = "default"


    print("--------------------------------")
    print("Predicted disease class:", result)
    print("Disease confidence:", confidence)
    print("--------------------------------")


    # =====================================================
    # LOW CONFIDENCE CHECK
    # =====================================================

    if confidence < 0.50:

        return render_template(
            "result.html",
            error=ui_text[lang]["invalid"],
            lang=lang,
            ui=ui_text[lang]
        )


    # =====================================================
    # GET DISEASE INFORMATION
    # =====================================================

    info = disease_data.get(
        result,
        disease_data.get("default")
    )


    # -----------------------------------------------------
    # SAFETY CHECK
    # -----------------------------------------------------

    if info is None:

        return render_template(
            "result.html",
            error=ui_text[lang]["invalid"],
            lang=lang,
            ui=ui_text[lang]
        )


    # =====================================================
    # GET LANGUAGE DATA
    # =====================================================

    disease_name = info["name"].get(
        lang,
        info["name"]["en"]
    )


    cause = info["cause"].get(
        lang,
        info["cause"]["en"]
    )


    treatment = info["treatment"].get(
        lang,
        info["treatment"]["en"]
    )


    pesticide = info["pesticide"].get(
        lang,
        info["pesticide"]["en"]
    )


    prevention = info["prevention"].get(
        lang,
        info["prevention"]["en"]
    )


    # =====================================================
    # FIND PESTICIDE IMAGE
    # =====================================================

    pesticide_english = info["pesticide"].get(
        "en",
        ""
    )


    pesticide_image = pesticide_images.get(
        pesticide_english,
        None
    )


    print("--------------------------------")
    print("Pesticide:", pesticide_english)
    print("Pesticide image:", pesticide_image)
    print("--------------------------------")


    # =====================================================
    # FINAL CONFIDENCE
    # =====================================================

    confidence_percentage = round(
        confidence * 100,
        1
    )


    # =====================================================
    # FINAL RESULT
    # =====================================================

    return render_template(

        "result.html",

        disease=disease_name,

        cause=cause,

        treatment=treatment,

        pesticide=pesticide,

        prevention=prevention,

        confidence=confidence_percentage,

        pesticide_image=pesticide_image,

        lang=lang,

        ui=ui_text[lang]
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )