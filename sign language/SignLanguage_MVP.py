import argparse
import os
import time
import glob
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --------------------------- Utilities: landmarks extraction -------------------------
mp_hands = mp.solutions.hands

def extract_hand_landmarks_from_frame(frame, hands):
    """Return flattened landmark vector (both hands) or None if no hand detected."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None
    # We will store up to 2 hands; each hand 21 landmarks (x,y,z) => 63 values
    # If only one hand found, second hand is zeros.ش
    hands_sorted = results.multi_hand_landmarks
    vectors = []
    for hand_landmarks in hands_sorted[:2]:
        for lm in hand_landmarks.landmark:
            vectors.extend([lm.x, lm.y, lm.z])
    # pad if only one hand
    if len(hands_sorted) == 1:
        vectors.extend([0.0] * 63)
    return np.array(vectors, dtype=np.float32)

# --------------------------- Data collection ---------------------------
def collect_samples(label, frames_per_sample, out_dir, samples_count=10):
    """Open camera and collect `samples_count` sequences, each of length frames_per_sample."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    print(f"Collecting for label '{label}': {samples_count} samples of {frames_per_sample} frames.")
    collected = 0
    try:
        while collected < samples_count:
            seq = []
            start = time.time()
            while len(seq) < frames_per_sample:
                ret, frame = cap.read()
                if not ret:
                    break
                
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                
                if not results.multi_hand_landmarks:
                    # show instruction
                    cv2.putText(frame, 'Show your hand to the camera', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    cv2.imshow('Collect', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt
                    continue
                
                # استخراج البيانات
                hands_sorted = results.multi_hand_landmarks
                vectors = []
                for hand_landmarks in hands_sorted[:2]:
                    for lm in hand_landmarks.landmark:
                        vectors.extend([lm.x, lm.y, lm.z])
                # pad if only one hand
                if len(hands_sorted) == 1:
                    vectors.extend([0.0] * 63)
                vec = np.array(vectors, dtype=np.float32)
                
                seq.append(vec)
                
                # show progress
                cv2.putText(frame, f'Collecting {label} {collected+1}/{samples_count} frame {len(seq)}/{frames_per_sample}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                
                # رسم اليد
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                cv2.imshow('Collect', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt
                    
            # finished one sample
            arr = np.stack(seq)  # shape (frames_per_sample, features)
            file_path = os.path.join(out_dir, f'{label}_{int(time.time())}_{collected}.npy')
            np.save(file_path, arr)
            print('Saved', file_path)
            collected += 1
            # small pause between samples
            time.sleep(0.8)
    except KeyboardInterrupt:
        print('Interrupted by user')
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

# --------------------------- Dataset loading & preprocessing ---------------------------

def load_dataset(data_dir, seq_len=50):
    """Load .npy sequence files and pad/truncate to seq_len. Return X, y.
    Each .npy is expected to be (T, F) where F=126 (2 hands x 21 landmarks x 3 coords) or (T,63) for single-hand
    """
    files = glob.glob(os.path.join(data_dir, '*.npy'))
    X, y = [], []
    for f in files:
        arr = np.load(f, allow_pickle=False)
        # flatten feature dim if 63 -> 63, if 126 -> 126
        # pad/truncate time dimension
        if arr.shape[0] < seq_len:
            pad = np.zeros((seq_len - arr.shape[0], arr.shape[1]), dtype=np.float32)
            arr = np.vstack([arr, pad])
        else:
            arr = arr[:seq_len]
        X.append(arr)
        # label from filename
        label = Path(f).stem.split('_')[0]
        y.append(label)
    X = np.stack(X)  # (N, seq_len, features)
    y = np.array(y)
    print(f'Loaded {len(X)} samples, seq_len={seq_len}, features={X.shape[-1]}')
    return X, y

# --------------------------- Model definition ---------------------------

def build_lstm_model(input_shape, num_classes):
    model = Sequential()
    model.add(Masking(mask_value=0.0, input_shape=input_shape))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --------------------------- Train pipeline ---------------------------

def train_model(data_dir, model_path, seq_len=50, epochs=30, batch_size=8):
    X, y_labels = load_dataset(data_dir, seq_len=seq_len)
    le = LabelEncoder()
    y = le.fit_transform(y_labels)
    # save label encoder mapping
    mapping_path = model_path + '.labels.npy'
    np.save(mapping_path, le.classes_)
    print('Label classes:', le.classes_)
    # split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = build_lstm_model(input_shape=(seq_len, X.shape[2]), num_classes=len(le.classes_))
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
    print('Training finished. Best model saved to', model_path)

# --------------------------- Real-time demo ---------------------------
def demo_realtime(model_path, seq_len=50, threshold=0.3):
    if not os.path.exists(model_path):
        print('Model file not found:', model_path)
        return
    model = tf.keras.models.load_model(model_path)
    classes = np.load(model_path + '.labels.npy', allow_pickle=True)

    cap = cv2.VideoCapture(0)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    sequence = []
    last_prediction = ''
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                # استخراج البيانات
                hands_sorted = results.multi_hand_landmarks
                vectors = []
                for hand_lms in hands_sorted[:2]:
                    for lm in hand_lms.landmark:
                        vectors.extend([lm.x, lm.y, lm.z])
                if len(hands_sorted) == 1:
                    vectors.extend([0.0] * 63)
                vec = np.array(vectors, dtype=np.float32)
                
                sequence.append(vec)
                if len(sequence) > seq_len:
                    sequence.pop(0)
                    
                if len(sequence) == seq_len:
                    arr = np.expand_dims(np.stack(sequence), axis=0)
                    probs = model.predict(arr, verbose=0)[0]
                    idx = np.argmax(probs)
                    prob = float(probs[idx])
                    if prob > threshold:
                        last_prediction = classes[idx]
            
            # عرض الكلمة فقط
            cv2.putText(frame, last_prediction, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            
            cv2.imshow('Demo', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
# --------------------------- CLI ---------------------------

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    p_col = sub.add_parser('collect')
    p_col.add_argument('--label', required=True)
    p_col.add_argument('--frames', type=int, default=50)
    p_col.add_argument('--output', default='data')
    p_col.add_argument('--samples', type=int, default=8)

    p_train = sub.add_parser('train')
    p_train.add_argument('--data_dir', default='data')
    p_train.add_argument('--model_path', default='model.h5')
    p_train.add_argument('--seq_len', type=int, default=50)
    p_train.add_argument('--epochs', type=int, default=30)

    p_demo = sub.add_parser('demo')
    p_demo.add_argument('--model_path', default='model.h5')
    p_demo.add_argument('--seq_len', type=int, default=50)

    args = parser.parse_args()
    if args.cmd == 'collect':
        collect_samples(args.label, args.frames, args.output, samples_count=args.samples)
    elif args.cmd == 'train':
        train_model(args.data_dir, args.model_path, seq_len=args.seq_len, epochs=args.epochs)
    elif args.cmd == 'demo':
        demo_realtime(args.model_path, seq_len=args.seq_len)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()


   