import pyaudio
import numpy as np
import librosa
import tensorflow as tf


# โหลดโมเดล Autopitch ที่ฝึกไว้
model = tf.keras.models.load_model("MelNoteClassifier.h5")  # ใส่ path โมเดลของนาย

def predict_note():
    """รับเสียงจากไมค์ -> แปลงเป็น MFCC -> ทำนายโน้ต"""
    mfcc_input = pyaudio.get_mfcc_from_audio().reshape(1, -1)  # Reshape ให้ตรงกับ Input ของโมเดล
    predicted_note = model.predict(mfcc_input)
    
    # แปลงผลลัพธ์จากโมเดลเป็นโน้ตดนตรี (เช่น C, D, E, F, G, A, B)
    note_labels = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    predicted_label = note_labels[np.argmax(predicted_note)]
    
    return predicted_label

# ทดสอบทำนายโน้ต
predicted = predict_note()
print("🎵 Predicted Note:", predicted)