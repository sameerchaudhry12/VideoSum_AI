# fix_model_compatibility.py
# Fixes Keras version mismatch for both CNN and LSTM models.
# Run once: python fix_model_compatibility.py

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CNN_MODEL_PATH, LSTM_MODEL_PATH

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization,
    Bidirectional, LSTM, Input
)

# ── Fix CNN ───────────────────────────────────────────────────────────────────

def fix_cnn():
    print("\n[1/2] Fixing CNN model...")

    base_model = MobileNetV2(input_shape=(48, 48, 3), include_top=False, weights=None)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    output = Dense(7, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)

    try:
        model.load_weights(CNN_MODEL_PATH)
        print("  Weights loaded OK")
    except Exception as e:
        print(f"  load_weights failed: {e}")
        saved_path = CNN_MODEL_PATH.replace(".h5", "_savedmodel")
        if os.path.exists(saved_path):
            print("  Loading from existing SavedModel instead...")
            model = tf.keras.models.load_model(saved_path)
            print("  Loaded from SavedModel OK")
        else:
            print("  ERROR: Could not load CNN weights.")
            return

    dummy = np.zeros((1, 48, 48, 3), dtype=np.float32)
    pred = model.predict(dummy, verbose=0)
    print(f"  Forward pass OK — output shape: {pred.shape}, sum: {pred.sum():.4f}")

    out_path = CNN_MODEL_PATH.replace(".h5", "_savedmodel")
    model.save(out_path, save_format="tf")
    print(f"  Saved: {out_path}")


# ── Fix LSTM ──────────────────────────────────────────────────────────────────

def fix_lstm():
    print("\n[2/2] Fixing LSTM model...")

    from tensorflow.keras.regularizers import l2

    model = Sequential([
        Input(shape=(200, 82)),
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),

        Bidirectional(LSTM(64, return_sequences=True)),
        BatchNormalization(),
        Dropout(0.3),

        Bidirectional(LSTM(32, return_sequences=False)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(8, activation='softmax'),
    ])

    try:
        model.load_weights(LSTM_MODEL_PATH)
        print("  Weights loaded OK")
    except Exception as e:
        print(f"  load_weights failed: {e}")
        print("  ERROR: Could not load LSTM weights. Check LSTM_MODEL_PATH in config.py")
        return

    dummy = np.zeros((1, 200, 82), dtype=np.float32)
    pred = model.predict(dummy, verbose=0)
    print(f"  Forward pass OK — output shape: {pred.shape}, sum: {pred.sum():.4f}")

    out_path = LSTM_MODEL_PATH.replace(".h5", "_savedmodel")
    model.save(out_path, save_format="tf")
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  Model Compatibility Fixer")
    print("=" * 50)

    fix_cnn()
    fix_lstm()

    print("\n" + "=" * 50)
    print("  Done! Now update config.py with these two lines:")
    print()
    print("  CNN_MODEL_PATH  = ...models/cnn/mobilenetv2_fer2013_savedmodel")
    print("  LSTM_MODEL_PATH = ...models/lstm/lstm_ravdess_savedmodel")
    print()
    print("  Then re-run: python fusion.py --video ... --skip_bert")
    print("=" * 50)