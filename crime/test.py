import os
import tensorflow as tf
from tensorflow.keras.models import load_model

def find_and_check_model():
    # 1. Automatically find any .h5 file in your folder
    current_files = os.listdir('.')
    h5_files = [f for f in current_files if f.endswith('.h5')]

    if not h5_files:
        print("❌ ERROR: No .h5 file found in this folder!")
        print(f"Current folder is: {os.getcwd()}")
        return

    # Use the first .h5 file found
    actual_model_name = h5_files[0]
    print(f"✅ Found model file: {actual_model_name}")

    try:
        # 2. Load the model
        model = load_model(actual_model_name, compile=False)
        
        # 3. Get the number of classes
        # This looks at the very last layer's output size
        num_classes = model.output_shape[-1]
        
        print("-" * 30)
        print(f"MODEL ANALYSIS:")
        print(f"Number of Classes: {num_classes}")
        print(f"Input Image Size:  {model.input_shape[1]}x{model.input_shape[2]}")
        print("-" * 30)
        
        if num_classes == 1:
            print("This is a BINARY model (e.g., 0=Normal, 1=Violence).")
        elif num_classes == 2:
            print("This model has 2 categories.")
        elif num_classes == 3:
            print("This model has 3 categories (Likely: Normal, Crime, and Violence).")
        else:
            print(f"This model has {num_classes} different categories.")

    except Exception as e:
        print(f"❌ Failed to analyze model: {e}")

if __name__ == "__main__":
    find_and_check_model()
    
# import os
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import cv2
# import numpy as np

# # --- 1. SETUP ---
# # Update these two names to match your EXACT files
# MY_MODEL_FILE = 'crime_model.h5' # <-- CHANGE THIS to your filename
# MY_IMAGE_FILE = 'test_image.jpeg'                # <-- CHANGE THIS to your image name
# CLASSES = ["Safety/Normal", "safety", "Crime","fist-fight"] 

# def check_files():
#     """Verify if files exist before starting"""
#     print(f"Checking current directory: {os.getcwd()}")
    
#     if not os.path.exists(MY_MODEL_FILE):
#         print(f"\n[ERROR] Cannot find: {MY_MODEL_FILE}")
#         print("Files currently in this folder are:")
#         for f in os.listdir():
#             if f.endswith(".h5"):
#                 print(f" -> Found this model instead: {f}")
#         return False
    
#     if not os.path.exists(MY_IMAGE_FILE):
#         print(f"\n[ERROR] Cannot find image: {MY_IMAGE_FILE}")
#         return False
        
#     return True

# def run_inference():
#     if not check_files():
#         return

#     # 2. LOAD MODEL
#     print("Loading model... please wait...")
#     model = load_model(MY_MODEL_FILE, compile=False)
    
#     # 3. DIAGNOSE SHAPE (Fixing your previous error)
#     # We get the shape directly from the model so it never mismatches again
#     input_shape = model.input_shape 
#     # input_shape is usually (None, Height, Width, Channels)
#     req_h = input_shape[1]
#     req_w = input_shape[2]
#     print(f"Success! Model loaded. It requires images of size: {req_w}x{req_h}")

#     # 4. LOAD AND PREPROCESS IMAGE
#     img = cv2.imread(MY_IMAGE_FILE)
#     display_img = img.copy()
    
#     img = cv2.resize(img, (req_w, req_h))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = img.astype("float32") / 255.0
#     img = np.expand_dims(img, axis=0)

#     # 5. PREDICT
#     preds = model.predict(img)
#     class_idx = np.argmax(preds[0])
#     label = CLASSES[class_idx]
#     conf = preds[0][class_idx]

#     # 6. SHOW RESULT
#     print(f"\nRESULT: {label} ({conf*100:.2f}%)")
#     cv2.putText(display_img, f"{label} {conf*100:.1f}%", (20, 50), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     cv2.imshow("Detection Result", display_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     run_inference()
