import torch
print("**----------------checking for gpu's---------------------**")
# Check if MPS (Apple GPU) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device (Apple GPU) is available and will be used.")
else:
    device = torch.device("cpu")
    print("MPS device is not available. Using CPU instead.")

# Print available devices
print("Available devices:")
print("MPS available:", torch.backends.mps.is_available())
print("CUDA available:", torch.cuda.is_available())  # This will be False on Mac since CUDA is not supported


print("**----------------reading the images---------------------**")

import numpy as np
import tensorflow as tf
import pandas as pd

import glob
from os.path import join

from PIL import Image,ImageOps

#train_path = 'all_data/train/radiology/images'
train_path='/home/bmlokesh/bmlokesh/MainProject/Images/radiologytraindata/Chest/'
import os
import glob
import json
from os.path import join
from collections import defaultdict
from nltk.tokenize import word_tokenize
import re

# Function to clean tokens (remove numbers and special characters)
def clean_token(token):
    return re.sub(r'[^a-z]', '', token.lower())  # Keep only lowercase alphabetic characters

# Build a unified vocabulary without unknown symbols or numbers
def build_vocab(file_paths, frequency_threshold=2):
    counts = defaultdict(int)
    
    # Count token frequencies across all files
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                tokens = word_tokenize(line.lower())[1:]  # Skip the ID token
                clean_tokens = [clean_token(token) for token in tokens if clean_token(token)]  # Remove unwanted tokens
                for token in clean_tokens:
                    counts[token] += 1
    
    # Filter tokens by frequency threshold
    useful_tokens = [k for k, v in counts.items() if v >= frequency_threshold]
    
    # Create vocabulary
    vocab = {'<start>': 0, '<end>': 1, '<padding>': 2}
    value = 3
    for token in useful_tokens:
        vocab[token] = value
        value += 1
    
    return vocab

# Save the unified vocabulary
def save_vocab(vocab, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as vocab_file:
        json.dump(vocab, vocab_file)
    print(f"Vocabulary saved to {output_path}")

# Tokenize lines without unknown symbols or numbers
def tokenize_lines(file_paths, vocab, embed_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    padding_token = vocab['<padding>']
    
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        filename = os.path.basename(file_path).split('.')[0]
        tokenized_lines = {}
        
        for line in lines:
            tokens = word_tokenize(line.lower())
            id = tokens[0]  # Extract ID
            clean_tokens = [clean_token(token) for token in tokens[1:] if clean_token(token)]  # Remove unwanted tokens
            
            # Convert tokens to vocabulary indices
            token_indices = [vocab['<start>']] + [vocab[token] for token in clean_tokens if token in vocab] + [vocab['<end>']]
            
            # Pad or truncate tokens to embed_size
            token_indices = token_indices[:embed_size] + [padding_token] * max(0, embed_size - len(token_indices))
            
            tokenized_lines[id] = token_indices
        
        # Save tokenized lines
        output_path = os.path.join(output_dir, f"{filename}_tokenized.json")
        with open(output_path, "w") as tokenized_file:
            json.dump(tokenized_lines, tokenized_file)
        print(f"Tokenized data saved to {output_path}")

# Paths and parameters
train_path = '/home/bmlokesh/bmlokesh/MainProject/Images/radiologytraindata/Chest/radiology_filtered'
output_vocab_path = "Processed_Data/unified_vocab_cleaned.json"
output_tokenized_dir = "Processed_Data"

# Get all text files
all_files_paths = glob.glob(join(train_path, "*.txt"))

# Build vocabulary and save it
vocab = build_vocab(all_files_paths, frequency_threshold=2)
save_vocab(vocab, output_vocab_path)

# Tokenize lines with cleaned vocabulary
tokenize_lines(all_files_paths, vocab, embed_size=50, output_dir=output_tokenized_dir)

images = glob.glob(join(train_path,"*"))

scale_to = 224

print(len(images))

'''
print("**----------------Changing the file names to roco Id's---------------------**")

#  TO change the file names to the respected ROCO ID
import os
import pandas as pd

# Load the CSV file that contains the mapping of ROCO IDs to PMC filenames
train_data_path = 'all_data/test/radiologytestdata.csv'
data_df = pd.read_csv(train_data_path)
print("Sample data from CSV:")
print(data_df.head())

# Extract base filename starting with "PMC" and ending with ".jpg"
data_df['base_name'] = data_df['name'].str.extract(r'(PMC.*?\.jpg)', expand=False).str.lower()

# Directory where the images are stored
images_dir = 'all_data/train/radiology/images'

# Extract ROCO IDs for all images in the directory
for filename in os.listdir(images_dir):
    if filename.lower().startswith('pmc'):
        base_name = filename.split('_')[0].lower()
        # Match with the CSV to find the corresponding ROCO ID
        matching_row = data_df[data_df['base_name'] == base_name]
        if not matching_row.empty:
            roco_id = matching_row['id'].values[0].lower()
            old_path = os.path.join(images_dir, filename)
            new_filename = roco_id + os.path.splitext(filename)[-1]
            new_path = os.path.join(images_dir, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed {filename} to {new_filename}")

print("Image renaming completed.")
'''

"""
print("**----------------Deleting the duplicate file names---------------------**")

# Finding the duplicate file names and deleting the files by keeping the first image in the over lapping.
import hashlib
import glob
from collections import defaultdict

# Directory where the images are stored
images_dir = 'all_data/train/radiology/images'

# Dictionary to store hash values and corresponding file paths
hash_dict = defaultdict(list)

# Variable to count the number of duplicate sets
duplicate_count = 0

# Array to store all duplicate file names
duplicate_files = []

def get_file_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Iterate through all files in the directory using glob
for file_path in glob.glob(os.path.join(images_dir, "*")):
    if os.path.isfile(file_path):
        file_hash = get_file_hash(file_path)
        hash_dict[file_hash].append(file_path)

# Find and delete duplicate files
print("Deleting duplicate files:")
for file_hash, files in hash_dict.items():
    if len(files) > 1:
        duplicate_count += 1
        # Keep the first file and delete the rest
        for file_to_delete in files[1:]:
            duplicate_files.append(file_to_delete)
            os.remove(file_to_delete)
            print(f"Deleted: {file_to_delete}")

print(f"Total number of duplicate sets: {duplicate_count}")
print(f"All duplicate files deleted: {duplicate_files}")
"""

print("**----------------checking the length of total images---------------------**")
train_path = '/home/bmlokesh/bmlokesh/MainProject/Images/radiologytraindata/Chest'
images = glob.glob(join(train_path,"*"))

scale_to = 224

print("**----------------length of total images---------------------**")
print(len(images))

print("**----------------head of the images---------------------**")
print(images[:5])

img = Image.open(images[0]).convert("RGB")
print(np.array(img).shape)
# img = np.array(ImageOps.fit(img, (128, 128)), dtype=np.float32) # Commented to test the script in linux
print("**----------------generating normalized images---------------------**")
all_imgs = []
counter = 0

for image in images:
    #if counter % 100 == 0:
    #  print(counter)
    #Opens Image
    img = Image.open(image).convert("RGB")
    #Resizes Image
    img = np.array(ImageOps.fit(img,(scale_to,scale_to)),dtype=np.float32)
    #Normlizes Pixel Data
    img /= 255.
    #Append to numpy array
    all_imgs.append(img)
    #counter+=1
# Transforming into numpy array and stacking
all_imgs = np.stack(([element for element in all_imgs]), axis = 0)
all_imgs.shape

np.save('Processed_Data/224x244 Normalized Images', all_imgs)


print(len(all_imgs))
print("**----------------generating image Id's---------------------**")
roco_ids = []
for image in images:
    #if counter % 100 == 0:
    #  print(counter)
    roco_id = image.split('/')[-1]
    roco_id = roco_id.lower()[:10]
    #print(roco_id)
    roco_ids.append(roco_id)

roco_ids = np.array(roco_ids)
np.save('Processed_Data/224x244 Images IDs', roco_ids)

roco_ids

imgs = np.load('Processed_Data/224x244 Normalized Images.npy')

means = np.mean(imgs, axis = (0, 1,2))
print(means.shape, means)
print("**----------------Printing image ---------------------**")
img = imgs[100, :,:, :]
print(Image.fromarray(np.uint8(img*255)))

means = np.mean(imgs, axis = (0,1,2))
stds = np.std(imgs, axis = (0,1,2))
print(means, stds)
print(means.shape, stds.shape)