# Import TensorFlow 
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds
import tensorflow_hub as hub

# TODO: Make all other necessary imports.
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image
import argparse

# Predict function
def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_image(image)
    expanded_image = np.expand_dims(processed_image, axis=0)

    prediction = model.predict(expanded_image)
    
    top_k_probs, top_k_classes = tf.math.top_k(prediction, top_k)
    
    top_k_probs = top_k_probs.numpy()
    top_k_probs_conv = convert_to_list(top_k_probs)
    
    top_k_classes = top_k_classes.numpy()
    top_k_classes_conv = convert_to_list(top_k_classes)
    
    return top_k_probs_conv, top_k_classes_conv

# Process image function
def process_image(image):
    image_size = 224
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()

# Convert ndarray to list function
def convert_to_list(array):
    temp_list = []
    for element in array.flat:
        temp_list.append(element)
        
    return temp_list

# Print statement function
def output_msg(input1, input2):
    for x, y in zip(input1, input2):
        print('Image is {} with a probability of {:,.3f}'.format(y, x))

              
# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Classifier Project')

    # Image and model are nonoptional arguments. tok_k amd category names are optional
    parser.add_argument('image', help='Input image path')
    parser.add_argument('model', help='Input model path')
    parser.add_argument('--top_k', type=int, help='Input # of classes to view')
    parser.add_argument('--category_names', help='Input path to json file')
    
    args = parser.parse_args()
    
    image_path = args.image
    
    model = load_model(args.model, compile = False, custom_objects={'KerasLayer':hub.KerasLayer})
    
    # Basic usage
    if args.top_k is None and args.category_names is None:
        top_k = 1
        probs, classes = predict(image_path, model, top_k)
        output_msg(probs,classes)
             
    # Option to return top_k most likely classes with labels
    elif args.top_k is not None and args.category_names is None:
        top_k = args.top_k
        probs, classes = predict(image_path, model, top_k)
        output_msg(probs,classes)
        
    # Option to return most likely class with flower name
    elif args.top_k is None and args.category_names is not None:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        
        top_k = 1
        probs, classes = predict(image_path, model, top_k)
        
        label_name = []
        for element in classes:
            label_name.append(class_names[str(element+1)])
        output_msg(probs,label_name)
     
    # Option to return top_k most likely classes with flower name
    else:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
            
        top_k = args.top_k
        probs, classes = predict(image_path, model, top_k)
        
        label_name = []
        for element in classes:
            label_name.append(class_names[str(element+1)])
        output_msg(probs,label_name)