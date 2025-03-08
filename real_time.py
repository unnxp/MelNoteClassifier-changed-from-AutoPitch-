import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import pyaudio
import time

# ตั้งค่าการรับเสียง
SAMPLE_RATE = 22050  # ค่ามาตรฐานของ librosa
CHUNK = 1024  # ขนาดบัฟเฟอร์

# เปิดไมโครโฟน
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, 
                    input=True, frames_per_buffer=CHUNK)

# โหลดโมเดล Noise Suppression จาก TensorFlow Hub
model_url = "https://tfhub.dev/google/speech_noise_canceling/1"
model = hub.load(model_url)

def noise_suppression(input_audio):
    """ใช้โมเดลเพื่อลดเสียงนอยส์"""
    # Convert numpy array to tensor
    input_tensor = tf.convert_to_tensor(input_audio, dtype=tf.float32)
    
    # ปรับขนาดให้เป็นรูปแบบที่โมเดลต้องการ
    input_tensor = tf.reshape(input_tensor, (1, -1))
    
    # ใช้โมเดลเพื่อลดเสียงนอยส์
    output_tensor = model(input_tensor)
    
    # Convert output back to numpy
    output_audio = output_tensor.numpy().flatten()
    return output_audio

def get_mel_from_audio():
    """รับเสียงจากไมค์และแปลงเป็น Mel spectrogram"""
    try:
        audio_data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.float32)
        
        # ใช้ Noise Suppression
        cleaned_audio = noise_suppression(audio_data)
        
        # สร้าง Mel spectrogram จากเสียงที่ปรับแล้ว
        mel = librosa.feature.melspectrogram(y=cleaned_audio, sr=SAMPLE_RATE, n_mels=128, fmax=8000)  # สร้าง Mel spectrogram
        return mel
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการรับข้อมูลเสียง: {e}")
        return None



# โหลดโมเดล Autopitch ที่ฝึกไว้
model = tf.keras.models.load_model("MelNoteClassifier.h5")  # ใส่ path โมเดลของนาย

# ฟังก์ชันแปลง MIDI pitch เป็นชื่อโน้ต
def midi_to_note(midi):
    # รายการโน้ตที่สัมพันธ์กับ MIDI pitch
    note_names = [
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    ]
    octave = midi // 12 - 1
    note = note_names[midi % 12]
    return f"{note}{octave}"

def predict_note():
    """รับเสียงจากไมค์ -> แปลงเป็น Mel spectrogram -> ทำนายโน้ต"""
    mel_input = get_mel_from_audio()
    if mel_input is None:
        return "ไม่มีข้อมูลเพียงพอ"
    
    # เพิ่มมิติให้ตรงกับ input shape ของโมเดล (128, 128, 1)
    mel_input_resized = np.pad(mel_input, ((0, 0), (0, 128 - mel_input.shape[1])), 'constant')  # padding ให้กว้างเป็น 128
    mel_input_resized = np.reshape(mel_input_resized, (1, 128, 128, 1))  # reshape ให้ตรงกับ Input ของโมเดล

    predicted_note = model.predict(mel_input_resized)
    
    # เลือกโน้ตที่ทำนายได้จาก 0-127 (ใช้ np.argmax เพื่อเลือก pitch)
    predicted_midi = np.argmax(predicted_note)
    
    # แปลง MIDI pitch เป็นโน้ต
    predicted_note_name = midi_to_note(predicted_midi)
    
    return predicted_note_name

# ทดสอบทำนายโน้ต
print("🎤 เริ่มการทำนายโน้ต... (กด Ctrl+C เพื่อหยุด)")

try:
    while True:
        predicted = predict_note()
        print("🎵 Predicted Note:", predicted)
        time.sleep(0.5)  # ป้องกันการโหลด CPU มากเกินไป
except KeyboardInterrupt:
    print("🔴 หยุดการทำงาน")
    stream.stop_stream()
    stream.close()
    audio.terminate()
