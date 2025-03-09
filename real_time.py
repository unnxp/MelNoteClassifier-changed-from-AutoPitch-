import pyaudio
import numpy as np
import librosa
import tensorflow as tf
import time
import scipy.signal as signal

# ตั้งค่าการรับเสียง
SAMPLE_RATE = 22050  # ค่ามาตรฐานของ librosa
CHUNK = 1024       # ขนาดบัฟเฟอร์

# เปิดไมโครโฟน
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, 
                    input=True, frames_per_buffer=CHUNK)

def noise_gate(audio_data, threshold=0.02):
    """
    ใช้ Noise Gate เพื่อตัดเสียงที่มีระดับต่ำกว่า threshold ออก
    """
    return np.where(np.abs(audio_data) >= threshold, audio_data, 0)

def bandpass_filter(audio_data, low_cutoff=90, high_cutoff=8000, sample_rate=22050, order=3):
    """
    ใช้ Bandpass Filter เพื่อกรองความถี่ที่ไม่ต้องการ
    """
    nyquist = 0.5 * sample_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_audio = signal.filtfilt(b, a, audio_data)
    return filtered_audio

def get_mel_from_audio():
    """
    รับเสียงจากไมค์และแปลงเป็น Mel spectrogram หลังจากลดเสียงรบกวนแล้ว
    หากเป็นช่วงเงียบ (silence) จะคืนค่า None
    """
    try:
        audio_data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.float32)
        
        # ใช้ Noise Gate เพื่อตัดเสียงเบา
        gated_audio = noise_gate(audio_data, threshold=0.02)
        
        # ใช้ Bandpass Filter เพื่อตัดความถี่ที่ไม่ต้องการ
        filtered_audio = bandpass_filter(gated_audio, low_cutoff=90, high_cutoff=8000, sample_rate=SAMPLE_RATE, order=3)
        
        # ตรวจจับเสียงเงียบโดยคำนวณพลังงาน (RMS energy)
        energy = np.mean(np.abs(filtered_audio))
        silence_threshold = 0.005  # ปรับค่า threshold ตามสภาพแวดล้อม
        if energy < silence_threshold:
            # ถือว่าเป็นช่วงเงียบ ไม่ส่งข้อมูลกลับ
            return None
        
        # สร้าง Mel spectrogram จากเสียงที่ลด noise แล้ว
        mel = librosa.feature.melspectrogram(y=filtered_audio, sr=SAMPLE_RATE, n_mels=128, fmax=8000)
        return mel
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการรับข้อมูลเสียง: {e}")
        return None

# โหลดโมเดล Autopitch ที่ฝึกไว้
model = tf.keras.models.load_model("MelNoteClassifierV6.h5")  # ระบุ path โมเดลของคุณ

def midi_to_note(midi):
    """
    แปลงค่า MIDI pitch เป็นชื่อโน้ต (เช่น C4, D#4)
    """
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = midi // 12 - 1
    note = note_names[midi % 12]
    return f"{note}{octave}"

def predict_note():
    """
    รับเสียงจากไมค์ -> แปลงเป็น Mel spectrogram -> ทำนายโน้ต
    หากไม่พบเสียง (silence) จะคืนค่า None
    """
    mel_input = get_mel_from_audio()
    if mel_input is None:
        return None  # ไม่พบสัญญาณเสียงที่เพียงพอ (เงียบ)
    
    # ปรับข้อมูลให้ตรงกับ input shape ของโมเดล (128, 128, 1)
    mel_input_resized = np.pad(mel_input, ((0, 0), (0, 128 - mel_input.shape[1])), 'constant')
    mel_input_resized = np.reshape(mel_input_resized, (1, 128, 128, 1))
    
    predicted = model.predict(mel_input_resized)
    predicted_midi = np.argmax(predicted)
    predicted_note_name = midi_to_note(predicted_midi)
    
    return predicted_note_name

# ทดสอบทำนายโน้ตแบบ Real-time
print("🎤 เริ่มการทำนายโน้ต... (กด Ctrl+C เพื่อหยุด)")
try:
    while True:
        note_pred = predict_note()
        if note_pred is not None:
            print("🎵 Predicted Note:", note_pred)
        else:
            # หากเป็นช่วงเงียบ ไม่แสดงผลลัพธ์
            print("🔇 Silence")
        time.sleep(0.5)
except KeyboardInterrupt:
    print("🔴 หยุดการทำงาน")
    stream.stop_stream()
    stream.close()
    audio.terminate()
