import os
import tensorflow as tf
from transformers import TFAutoModelForTokenClassification, AutoTokenizer

def convert_to_tflite(model_dir, tflite_model_path, tokenizer_path=None):
    """
    Convert a Hugging Face model to TFLite format.

    Args:
        model_dir (str): Path to the Hugging Face SavedModel directory.
        tflite_model_path (str): Path to save the TFLite model.
        tokenizer_path (str): Path to the tokenizer. If None, uses model_dir.
    """
    print("Loading TensorFlow model...")
    model = TFAutoModelForTokenClassification.from_pretrained(model_dir)

    # Define a dummy input for the TFLite conversion
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_dir)
    dummy_input = tokenizer("Convert this to TFLite!", return_tensors="tf", padding="max_length", max_length=128)

    # Convert to TensorFlow Lite format
    print("Converting to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model
    print(f"Saving TFLite model to {tflite_model_path}...")
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print("Conversion completed!")

if __name__ == "__main__":
    # Define paths
    model_dir = os.path.expanduser("~/.mozhi/models/hf/sroie2019v1")  # Path to your Hugging Face model
    tflite_model_path = "./sroie2019v1.tflite"  # Path to save the TFLite model
    tokenizer_path = None  # Optional: specify a custom tokenizer path

    # Convert the model
    convert_to_tflite(model_dir, tflite_model_path, tokenizer_path)

    print("TFLite model saved!")
