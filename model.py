import torch
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
import json


# Check if MPS (Apple GPU) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device (Apple GPU) is available and will be used.")
else:
    device = torch.device("cpu")
    print("MPS device is not available. Using CPU instead.")

print("*****************CNN Encoder and Decoder Model**************")

# CNN Encoder with Multi-Scale Convolutions
def CNN_Encoder(features_size):
    input_layer = tf.keras.layers.Input(shape=(224, 224, 3), dtype='float32')
    print("Building CNN Encoder...")

    conv_3x3 = tf.keras.layers.Conv2D(filters=50, kernel_size=3, padding='same', activation=tf.nn.relu)(input_layer)
    conv_3x3 = tf.keras.layers.Conv2D(filters=50, kernel_size=3, padding='same', activation=tf.nn.relu)(conv_3x3)

    conv_4x4 = tf.keras.layers.Conv2D(filters=50, kernel_size=4, padding='same', activation=tf.nn.relu)(input_layer)
    conv_4x4 = tf.keras.layers.Conv2D(filters=50, kernel_size=4, padding='same', activation=tf.nn.relu)(conv_4x4)

    conv_5x5 = tf.keras.layers.Conv2D(filters=50, kernel_size=5, padding='same', activation=tf.nn.relu)(input_layer)
    conv_5x5 = tf.keras.layers.Conv2D(filters=50, kernel_size=5, padding='same', activation=tf.nn.relu)(conv_5x5)

    concat = tf.keras.layers.Concatenate(axis=3)([conv_3x3, conv_4x4, conv_5x5])
    final_conv = tf.keras.layers.Conv2D(filters=3, kernel_size=1, padding='same')(concat)
    flat = tf.keras.layers.Flatten()(final_conv)

    image_features = tf.keras.layers.Dense(features_size, activation=tf.nn.relu)(flat)

    model = keras.Model(inputs=input_layer, outputs=image_features)
    print("CNN Encoder model built successfully.")
    return model

# LSTM Decoder for Sequence Generation
def LSTM_Decoder(vocab_size, embed_size, features_size):
    print("Building LSTM Decoder...")
    input_layer = tf.keras.layers.Input(shape=(features_size,), dtype='float32')
    input_repeated = tf.keras.layers.RepeatVector(embed_size)(input_layer)

    lstm_1 = tf.keras.layers.LSTM(units=512, return_sequences=True, dropout=0.4)(input_repeated)
    lstm_2 = tf.keras.layers.LSTM(units=512, return_sequences=True, dropout=0.4)(lstm_1)
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='softmax'))(lstm_2)

    model = keras.Model(inputs=input_layer, outputs=output)
    print("LSTM Decoder model built successfully.")
    return model

# Complete Encoder-Decoder Model
def make_encoder_decoder_model(scale, vocab_size, embed_size, features_size=1024):
    print("Building the complete encoder-decoder model...")
    image_input = tf.keras.layers.Input(shape=(scale, scale, 3), dtype='float32')

    # Encoder
    encoder = CNN_Encoder(features_size)
    encoded_features = encoder(image_input)

    # Decoder
    decoder = LSTM_Decoder(vocab_size, embed_size, features_size)
    decoded_output = decoder(encoded_features)

    # Full encoder-decoder model
    encoder_decoder = keras.Model(inputs=image_input, outputs=decoded_output)

    # Learning rate scheduler
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )

    # Compile the model
    encoder_decoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    print("Encoder-decoder model built and compiled successfully.")
    return encoder, decoder, encoder_decoder

print("*****************Data Preprocessing**************")

# Features size definition
features_size = 1024  # Define globally or pass explicitly in the function
# Define function for padding or truncating captions
def pad_or_truncate_caption(caption, embed_size, padding_token):
    """
        caption (list): List of tokens in the caption.
        embed_size (int): Expected size of the padded caption.
        padding_token (int): Token value used for padding.

    Returns:
        list: Caption padded or truncated to embed_size.
    """
    if len(caption) < embed_size:
        # Pad with the padding token
        caption = caption + [padding_token] * (embed_size - len(caption))
    elif len(caption) > embed_size:
        # Truncate to the embed_size
        caption = caption[:embed_size]
    return caption

# Load data
print("Loading data...")
imgs = np.load('Processed_Data/224x244 Normalized Images.npy')
print(f"Loaded images: {imgs.shape}")
img_ids = np.load('Processed_Data/224x244 Images IDs.npy')
print(f"Loaded image IDs: {len(img_ids)}")

# Load captions and vocabulary
print("Loading captions and vocabulary...")
with open('Processed_Data/unified_vocab.json') as f:
    vocab = json.load(f)
vocab_size = len(vocab) + 1
print(f"Vocabulary size: {vocab_size}")

with open('Processed_Data/padded_tokenized_targets_captions_chest.json') as f:
    targets = json.load(f)
print("Loaded target captions.")

with open('Processed_Data/Embed_sizings_unified.json') as f:
    embed_sizings = json.load(f)
embed_size = embed_sizings['captions_chest'] + 2
print(f"Embed size: {embed_size}")

# Prepare target data
padding_token = vocab.get('<padding>', 0)  # Define padding token value
y_train = []
err_ids = []
print("Preparing target data...")
'''
if len(img_ids) < 7000:
    print("Not enough images available. At least 65000 images are required.")
    raise ValueError("Not enough images available. At least 65000 images are required.")
'''
for id in img_ids:  # Limit to first 7000 images
    if id in targets:
        caption = targets[id]
        padded_caption = pad_or_truncate_caption(caption, embed_size, padding_token)
        y_train.append(np.array(padded_caption))
    else:
        print(f"Error: Caption not found for ID {id}")
        err_ids.append(id)
# Debug tokenized captions for out-of-bound tokens
invalid_tokens = []
for id, tokens in targets.items():
    for token in tokens:
        if token >= vocab_size:  # Check if token exceeds vocab size
            invalid_tokens.append((id, token))

if invalid_tokens:
    print(f"Found {len(invalid_tokens)} invalid tokens:")
    for id, token in invalid_tokens[:10]:  # Show first 10 invalid tokens for debugging
        print(f"ID: {id}, Token: {token}")
    #raise ValueError("Invalid tokens detected. Fix tokenization or vocabulary.")



print(f"Total captions processed: {len(y_train)}")
y_train = np.array(y_train)
print(f"Data preparation complete: {len(y_train)} valid captions, {len(err_ids)} errors.")

# Normalize images
print("Normalizing images...")
imgs = imgs / 255.0
print("Image normalization complete.")

# Train-test split
print("Splitting data into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split(imgs[:len(y_train)], y_train, test_size=0.1, random_state=42)
print(f"Training samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")

print("*****************Training the Model**************")
print("Training dataset shape: ", x_train.shape, y_train.shape)
print("Testing dataset shape: ", x_test.shape, y_test.shape)


# Ensure all target tokens are within vocab_size
for tokens in y_train:
    for token in tokens:
        if token >= vocab_size:
            raise ValueError(f"Invalid token {token} in target data. Maximum allowed: {vocab_size - 1}.")



# Build the model
encoder, decoder, encoder_decoder = make_encoder_decoder_model(224, vocab_size=vocab_size, embed_size=embed_size, features_size=features_size)

# Train the model
# Train with a smaller dataset for debugging
x_train_debug, y_train_debug = x_train, y_train
encoder_decoder.fit(
    x=x_train_debug,
    y=y_train_debug,
    validation_data=(x_test, y_test),
    batch_size=8,
    epochs=30
)

print("Model training complete.")

# Save the trained model
encoder_decoder.save('Trained_Encoder_Decoder_Model.h5')
print("Model saved to 'Trained_Encoder_Decoder_Model.h5'.")

np.save('Processed_Data/x_test.npy', x_test)
np.save('Processed_Data/y_test.npy', y_test)







import torch
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu
import json


def decode_caption(predictions, reverse_vocab):
    """
    Decodes tokenized predictions into readable text captions.

    Args:
        predictions (array): Predicted token indices for the caption.
        reverse_vocab (dict): Mapping from token indices to words.

    Returns:
        str: Decoded caption as a string.
    """
    words = []
    for idx in predictions:
        if idx == 0:  # Ignore padding token
            continue
        word = reverse_vocab.get(idx, '<unk>')  # Default to <unk> if word not found
        if word in ['<end>', '<padding>']:  # Stop decoding at the end token or ignore padding
            break
        words.append(word)
    return ' '.join(words)


# Check if MPS (Apple GPU) is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS device (Apple GPU) is available and will be used.")
else:
    device = torch.device("cpu")
    print("MPS device is not available. Using CPU instead.")

print("*****************CNN Encoder and Decoder Model**************")

# CNN Encoder with Multi-Scale Convolutions
def CNN_Encoder(features_size):
    input_layer = tf.keras.layers.Input(shape=(224, 224, 3), dtype='float32')
    print("Building CNN Encoder...")

    conv_3x3 = tf.keras.layers.Conv2D(filters=50, kernel_size=3, padding='same', activation=tf.nn.relu)(input_layer)
    conv_3x3 = tf.keras.layers.Conv2D(filters=50, kernel_size=3, padding='same', activation=tf.nn.relu)(conv_3x3)

    conv_4x4 = tf.keras.layers.Conv2D(filters=50, kernel_size=4, padding='same', activation=tf.nn.relu)(input_layer)
    conv_4x4 = tf.keras.layers.Conv2D(filters=50, kernel_size=4, padding='same', activation=tf.nn.relu)(conv_4x4)

    conv_5x5 = tf.keras.layers.Conv2D(filters=50, kernel_size=5, padding='same', activation=tf.nn.relu)(input_layer)
    conv_5x5 = tf.keras.layers.Conv2D(filters=50, kernel_size=5, padding='same', activation=tf.nn.relu)(conv_5x5)

    concat = tf.keras.layers.Concatenate(axis=3)([conv_3x3, conv_4x4, conv_5x5])
    final_conv = tf.keras.layers.Conv2D(filters=3, kernel_size=1, padding='same')(concat)
    flat = tf.keras.layers.Flatten()(final_conv)

    image_features = tf.keras.layers.Dense(features_size, activation=tf.nn.relu)(flat)

    model = keras.Model(inputs=input_layer, outputs=image_features)
    print("CNN Encoder model built successfully.")
    return model

# LSTM Decoder for Sequence Generation
def LSTM_Decoder(vocab_size, embed_size, features_size):
    print("Building LSTM Decoder...")
    input_layer = tf.keras.layers.Input(shape=(features_size,), dtype='float32')
    input_repeated = tf.keras.layers.RepeatVector(embed_size)(input_layer)

    lstm_1 = tf.keras.layers.LSTM(units=512, return_sequences=True, dropout=0.4)(input_repeated)
    lstm_2 = tf.keras.layers.LSTM(units=512, return_sequences=True, dropout=0.4)(lstm_1)
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='softmax'))(lstm_2)

    model = keras.Model(inputs=input_layer, outputs=output)
    print("LSTM Decoder model built successfully.")
    return model

# Complete Encoder-Decoder Model
def make_encoder_decoder_model(scale, vocab_size, embed_size, features_size=1024):
    print("Building the complete encoder-decoder model...")
    image_input = tf.keras.layers.Input(shape=(scale, scale, 3), dtype='float32')

    # Encoder
    encoder = CNN_Encoder(features_size)
    encoded_features = encoder(image_input)

    # Decoder
    decoder = LSTM_Decoder(vocab_size, embed_size, features_size)
    decoded_output = decoder(encoded_features)

    # Full encoder-decoder model
    encoder_decoder = keras.Model(inputs=image_input, outputs=decoded_output)

    # Learning rate scheduler
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )

    # Compile the model
    encoder_decoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    print("Encoder-decoder model built and compiled successfully.")
    return encoder, decoder, encoder_decoder

print("*****************Data Preprocessing**************")

# Features size definition
features_size = 1024  # Define globally or pass explicitly in the function
# Define function for padding or truncating captions
def pad_or_truncate_caption(caption, embed_size, padding_token):
    """
        caption (list): List of tokens in the caption.
        embed_size (int): Expected size of the padded caption.
        padding_token (int): Token value used for padding.

    Returns:
        list: Caption padded or truncated to embed_size.
    """
    if len(caption) < embed_size:
        # Pad with the padding token
        caption = caption + [padding_token] * (embed_size - len(caption))
    elif len(caption) > embed_size:
        # Truncate to the embed_size
        caption = caption[:embed_size]
    return caption

# Load data
print("Loading data...")
imgs = np.load('Processed_Data/224x244 Normalized Images.npy')
print(f"Loaded images: {imgs.shape}")
img_ids = np.load('Processed_Data/224x244 Images IDs.npy')
print(f"Loaded image IDs: {len(img_ids)}")

# Load captions and vocabulary
print("Loading captions and vocabulary...")
with open('Processed_Data/unified_vocab.json') as f:
    vocab = json.load(f)
vocab_size = len(vocab) + 1
print(f"Vocabulary size: {vocab_size}")

with open('Processed_Data/padded_tokenized_targets_captions_chest.json') as f:
    targets = json.load(f)
print("Loaded target captions.")

with open('Processed_Data/Embed_sizings_unified.json') as f:
    embed_sizings = json.load(f)
embed_size = embed_sizings['captions_chest'] + 2
print(f"Embed size: {embed_size}")

# Prepare target data
padding_token = vocab.get('<padding>', 0)  # Define padding token value
y_train = []
err_ids = []
print("Preparing target data...")
'''
if len(img_ids) < 7000:
    print("Not enough images available. At least 65000 images are required.")
    raise ValueError("Not enough images available. At least 65000 images are required.")
'''
for id in img_ids:  # Limit to first 7000 images
    if id in targets:
        caption = targets[id]
        padded_caption = pad_or_truncate_caption(caption, embed_size, padding_token)
        y_train.append(np.array(padded_caption))
    else:
        print(f"Error: Caption not found for ID {id}")
        err_ids.append(id)
# Debug tokenized captions for out-of-bound tokens
invalid_tokens = []
for id, tokens in targets.items():
    for token in tokens:
        if token >= vocab_size:  # Check if token exceeds vocab size
            invalid_tokens.append((id, token))

if invalid_tokens:
    print(f"Found {len(invalid_tokens)} invalid tokens:")
    for id, token in invalid_tokens[:10]:  # Show first 10 invalid tokens for debugging
        print(f"ID: {id}, Token: {token}")
    #raise ValueError("Invalid tokens detected. Fix tokenization or vocabulary.")



print(f"Total captions processed: {len(y_train)}")
y_train = np.array(y_train)
print(f"Data preparation complete: {len(y_train)} valid captions, {len(err_ids)} errors.")

# Normalize images
print("Normalizing images...")
imgs = imgs / 255.0
print("Image normalization complete.")

# Train-test split
print("Splitting data into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split(imgs[:len(y_train)], y_train, test_size=0.1, random_state=42)
print(f"Training samples: {x_train.shape[0]}, Test samples: {x_test.shape[0]}")

print("*****************Training the Model**************")
print("Training dataset shape: ", x_train.shape, y_train.shape)
print("Testing dataset shape: ", x_test.shape, y_test.shape)


print(y_train[:15])

# Ensure all target tokens are within vocab_size
for tokens in y_train:
    for token in tokens:
        if token >= vocab_size:
            raise ValueError(f"Invalid token {token} in target data. Maximum allowed: {vocab_size - 1}.")



# Build the model
encoder, decoder, encoder_decoder = make_encoder_decoder_model(224, vocab_size=vocab_size, embed_size=embed_size, features_size=features_size)


# Train the model
# Train with a smaller dataset for debugging
x_train_debug, y_train_debug = x_train, y_train
encoder_decoder.fit(
    x=x_train_debug,
    y=y_train_debug,
    validation_data=(x_test, y_test),
    batch_size=8,
    epochs=30
)

print("Model training complete.")

# Save the trained model
encoder_decoder.save('Trained_Encoder_Decoder_Model.h5')
print("Model saved to 'Trained_Encoder_Decoder_Model.h5'.")

np.save('Processed_Data/x_test.npy', x_test)
np.save('Processed_Data/y_test.npy', y_test)
