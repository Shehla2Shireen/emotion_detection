# convert_model.py
import tensorflow as tf

saved_model_path = "model"  # folder containing saved_model.pb + variables/
model = tf.keras.models.load_model(saved_model_path, compile=False)
model.save("model.keras")
print("Model converted to model.keras successfully")
