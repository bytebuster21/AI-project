import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
from PIL import Image
import argparse

def create_dummy_test_set(test_path):
    """Creates a dummy test set for demonstration."""
    if os.path.exists(test_path):
        shutil.rmtree(test_path)

    print("Creating dummy test set...")
    os.makedirs(os.path.join(test_path, 'class_A'), exist_ok=True)
    os.makedirs(os.path.join(test_path, 'class_B'), exist_ok=True)
    os.makedirs(os.path.join(test_path, 'class_C'), exist_ok=True)

    def create_dummy_image(path, color, size=(299, 299)):
        img = Image.new('RGB', size, color=color)
        img.save(path)

    # Create dummy testing images
    for i in range(5): # 5 images per class
        create_dummy_image(os.path.join(test_path, 'class_A', f'testA_{i:03d}.png'), (200, 50, 50))
        create_dummy_image(os.path.join(test_path, 'class_B', f'testB_{i:03d}.png'), (50, 200, 50))
        create_dummy_image(os.path.join(test_path, 'class_C', f'testC_{i:03d}.png'), (50, 50, 200))

    print(f"Dummy test set created at {test_path}")

def test_model(model_path, test_path):
    """Loads a trained model and tests it on the provided test set."""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Create a dummy test set if one doesn't exist
    if not os.path.exists(test_path):
        create_dummy_test_set(test_path)

    # Load the model
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    TARGET_SIZE = (299, 299)
    BATCH_SIZE = 20

    # Test Data Generator (ONLY rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Flow Images from Directory
    print("\nLoading test images...")
    test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Get class names
    class_names = list(test_generator.class_indices.keys())
    print(f"Found {test_generator.samples} test images belonging to {test_generator.num_classes} classes.")
    print("Class Names:", class_names)


    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    filenames = test_generator.filenames

    # Print results
    print("\n--- Test Results ---")
    for i in range(len(filenames)):
        print(f"File: {filenames[i]}, True Label: {class_names[true_classes[i]]}, Predicted Label: {class_names[predicted_classes[i]]}")

    # Clean up the dummy test set
    if "dummy_test_set" in test_path:
        print("\nCleaning up dummy test set...")
        shutil.rmtree(test_path)
        print("Dummy test set removed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained Keras model.')
    parser.add_argument('model', type=str, help='Path to the saved model file (e.g., model_checkpoints/best_model_vgg16.h5).')
    parser.add_argument('--test_path', type=str, default='dummy_test_set', help='Path to the test data directory.')
    args = parser.parse_args()

    test_model(args.model, args.test_path)
