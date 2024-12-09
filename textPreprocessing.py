import torch
import numpy as np
import tensorflow as tf
import pandas as pd

import glob
import os
from os.path import join

from nltk.tokenize import word_tokenize
import nltk
# nltk.download('punkt')
import re
from collections import defaultdict
import json

# Define File Paths
train_path = '/home/bmlokesh/bmlokesh/MainProject/Images/radiologytraindata/Chest/radiology_filtered'
train_files_paths = glob.glob(join(train_path, "*.txt"))
all_files_paths = train_files_paths

# Function to clean tokens (remove numbers and special characters)
def clean_token(token):
    return re.sub(r'[^a-z]', '', token.lower())  # Keep only lowercase alphabetic characters

# Calculate Maximum Embedding Size for Each File
embed_sizings = {}

for file_path in all_files_paths:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    maximum_embedding = 0

    for line in lines:
        line_lower = line.lower()
        tokens = [clean_token(token) for token in word_tokenize(line_lower)[1:] if clean_token(token)]

        if len(tokens) > maximum_embedding:
            maximum_embedding = len(tokens)

    filename = file_path.split('/')[-1].split('.')[0]
    embed_sizings[filename] = maximum_embedding

# Process Each File for Tokenization and Vocabulary Creation
counts = defaultdict(int)

# Tokenize and count across all files
for file_path in all_files_paths:
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            tokens = word_tokenize(line.lower())[1:]  # Skip ID
            clean_tokens = [clean_token(token) for token in tokens if clean_token(token)]
            for token in clean_tokens:
                counts[token] += 1

# Include all tokens with frequency >= threshold
frequency_threshold = 1  # Include all tokens that appear at least once
useful_tokens = [k for k, v in counts.items() if v >= frequency_threshold]

# Create unified vocabulary
vocab = {'?': 0}
value = 1
for token in useful_tokens:
    vocab[token] = value
    value += 1

# Add special tokens
vocab['<start>'] = value
value += 1
vocab['<end>'] = value
value += 1
vocab['<padding>'] = value

# Save unified vocabulary
with open("Processed_Data/unified_vocab.json", "w") as vocab_file:
    json.dump(vocab, vocab_file)

# Tokenization Using Unified Vocabulary
def tokenize_lines_with_vocab(file_paths, vocab, embed_sizings_path):
    embed_sizings = {}
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            lines = file.readlines()

        filename = file_path.split('/')[-1].split('.')[0]
        tokenized_lines = {}
        max_length = 0

        for line in lines:
            line_lower = line.lower()
            tokens = [clean_token(token) for token in word_tokenize(line_lower)[1:] if clean_token(token)]
            id = word_tokenize(line_lower)[0]
            tmp = [vocab['<start>']]
            
            # Debug missing tokens
            missing_tokens = []
            for token in tokens:
                if token not in vocab:
                    missing_tokens.append(token)
                tmp.append(vocab.get(token, vocab['?']))

            if missing_tokens:
                print(f"Missing tokens in ID {id}: {missing_tokens}")

            tmp.append(vocab['<end>'])

            # Update max_length
            if len(tmp) > max_length:
                max_length = len(tmp)

            # Pad or truncate tokens to fit embed_size
            tmp = tmp[:max_length] + [vocab['<padding>']] * max(0, max_length - len(tmp))

            tokenized_lines[id] = tmp

        # Save tokenized lines
        with open(f"Processed_Data/padded_tokenized_targets_{filename}.json", "w") as tokenized_file:
            json.dump(tokenized_lines, tokenized_file)

        embed_sizings[filename] = max_length

    # Save embedding sizes
    with open(embed_sizings_path, "w") as embed_file:
        json.dump(embed_sizings, embed_file)

# Tokenize lines and save embedding sizes
tokenize_lines_with_vocab(all_files_paths, vocab, "Processed_Data/Embed_sizings_unified.json")

print("embed_sigzing_unified done")

# Calculate the percentage of unknown tokens in the dataset
def calculate_unknown_token_percentage(tokenized_lines, vocab):
    total_tokens = 0
    unknown_tokens = 0

    for tokens in tokenized_lines.values():
        total_tokens += len(tokens)
        unknown_tokens += tokens.count(vocab['?'])  # Count occurrences of unknown token

    unknown_percentage = (unknown_tokens / total_tokens) * 100 if total_tokens > 0 else 0
    print(f"Unknown tokens: {unknown_percentage:.2f}% of total tokens.")
    return unknown_percentage

# Load tokenized captions
tokenized_file_path = "Processed_Data/padded_tokenized_targets_captions_chest.json"
with open(tokenized_file_path, "r") as tokenized_file:
    tokenized_lines = json.load(tokenized_file)

# Calculate and print the percentage of unknown tokens
calculate_unknown_token_percentage(tokenized_lines, vocab)


import json
import numpy as np
from nltk.tokenize import word_tokenize

# Load the unified vocabulary
with open("Processed_Data/unified_vocab.json", "r") as vocab_file:
    vocab = json.load(vocab_file)

# Load a sample preprocessed file
test_file_path = "Processed_Data/padded_tokenized_targets_captions_chest.json"
with open(test_file_path, "r") as tokenized_file:
    tokenized_lines = json.load(tokenized_file)

# Create a reverse vocabulary to map tokens back to words
reverse_vocab = {v: k for k, v in vocab.items()}

# Function to test dynamic padding on a sample of tokenized lines
def test_dynamic_padding_sample(tokenized_lines, reverse_vocab, sample_size=5):
    print("Testing dynamic padding on a sample...")
    
    padding_token = vocab.get('<padding>')  # Get the padding token value from the vocab
    unk_token = '<UNK>'
    updated_tokens = {}
    
    # Iterate through a sample of tokenized lines
    sample_keys = list(tokenized_lines.keys())[:sample_size]
    
    for id in sample_keys:
        tokens = tokenized_lines[id]
        print(f"\nProcessing ID: {id}")
        
        # Show original tokens
        print(f"Original Tokens: {tokens}")
        
        # Find the last meaningful token (not padding)
        last_token_index = len(tokens) - 1
        while last_token_index >= 0 and tokens[last_token_index] == padding_token:
            last_token_index -= 1
        
        # Remove padding after the last meaningful token
        tokens_no_padding = tokens[:last_token_index + 1]
        print(f"Tokens after removing padding: {tokens_no_padding}")
        
        # Reconstruct sentence from tokens
        words = [reverse_vocab.get(int(token), unk_token) for token in tokens_no_padding]
        reconstructed_sentence = ' '.join(words)
        print(f"Reconstructed Sentence: {reconstructed_sentence}")
        
        # Save updated tokens without padding
        updated_tokens[id] = tokens_no_padding
        
    # Save updated tokens to a new file
    with open("Processed_Data/updated_tokens_no_padding.json", "w") as updated_file:
        json.dump(updated_tokens, updated_file)
    print("\nUpdated tokens saved to 'Processed_Data/updated_tokens_no_padding.json'.")

# Run the function on a sample
test_dynamic_padding_sample(tokenized_lines, reverse_vocab, sample_size=5)
