import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import time
import scipy.signal as signal
import pyaudio
import os

# ตั้งค่าการรับเสียง
SAMPLE_RATE = 22050  # ค่ามาตรฐานของ librosa
CHUNK = 1024         # ขนาดบัฟเฟอร์

# เปิดไมโครโฟน
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, 
                    input=True, frames_per_buffer=CHUNK)

# โหลดโมเดลแค่ครั้งเดียว
if 'model' not in st.session_state:
    st.session_state.model = tf.keras.models.load_model("C:/Users/M S I/Documents/GitHub/AutoPitch/PerfectPitch2/MelNoteClassifierV6.h5")  # ระบุ path โมเดลของคุณ

# ฟังก์ชันแปลง MIDI เป็นชื่อโน้ต
def midi_to_note(midi):
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = midi // 12 - 1
    note = note_names[midi % 12]
    return f"{note}{octave}"

# ฟังก์ชัน Noise Gate เพื่อตัดเสียงที่เบากว่า threshold
def noise_gate(audio_data, threshold=0.02):
    return np.where(np.abs(audio_data) >= threshold, audio_data, 0)

# ฟังก์ชัน Bandpass Filter เพื่อตัดความถี่ที่ไม่ต้องการ
def bandpass_filter(audio_data, low_cutoff=90, high_cutoff=8000, sample_rate=22050, order=3):
    nyquist = 0.5 * sample_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_audio = signal.filtfilt(b, a, audio_data)
    return filtered_audio

# ฟังก์ชันแปลงเสียงจากไมค์เป็น Mel spectrogram
def get_mel_from_audio():
    try:
        audio_data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.float32)
        
        # ใช้ Noise Gate เพื่อตัดเสียงเบา
        gated_audio = noise_gate(audio_data, threshold=0.02)
        
        # ใช้ Bandpass Filter เพื่อตัดความถี่ที่ไม่ต้องการ
        filtered_audio = bandpass_filter(gated_audio, low_cutoff=90, high_cutoff=8000, sample_rate=SAMPLE_RATE, order=3)
        
        # ตรวจจับเสียงเงียบโดยคำนวณพลังงาน (RMS energy)
        energy = np.mean(np.abs(filtered_audio))
        silence_threshold = 0.005
        if energy < silence_threshold:
            return None
        
        # สร้าง Mel spectrogram จากเสียงที่ลด noise แล้ว
        mel = librosa.feature.melspectrogram(y=filtered_audio, sr=SAMPLE_RATE, n_mels=128, fmax=8000)
        return mel
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการรับข้อมูลเสียง: {e}")
        return None

# ฟังก์ชันทำนายโน้ตจาก Mel spectrogram
def predict_note():
    mel_input = get_mel_from_audio()
    if mel_input is None:
        return None  # ไม่มีเสียงหรือเสียงเงียบ
    
    mel_input_resized = np.pad(mel_input, ((0, 0), (0, 128 - mel_input.shape[1])), 'constant')
    mel_input_resized = np.reshape(mel_input_resized, (1, 128, 128, 1))
    
    predicted = st.session_state.model.predict(mel_input_resized)
    predicted_midi = np.argmax(predicted)
    predicted_note_name = midi_to_note(predicted_midi)
    
    return predicted_note_name

# สร้าง UI ใน Streamlit
st.title("🎵 Real-time Note Detection")
st.write("กดปุ่ม 'เริ่มรับเสียง' เพื่อทำนายโน้ตเสียงขณะนี้")

# สร้าง placeholder สำหรับแสดงโน้ต
note_placeholder = st.empty()

# สร้างสถานะการรับเสียง
if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False

# ฟังก์ชันสำหรับเริ่ม/หยุดการรับเสียง
def toggle_recording():
    st.session_state.is_recording = not st.session_state.is_recording

# ตรวจสอบว่าไฟล์พื้นหลังมีอยู่ใน path หรือไม่
background_path = "C:/Users/M S I/Desktop/PerfectPitch2/noisy-grid.png"  # ระบุ path ของภาพพื้นหลังในเครื่อง
if os.path.exists(background_path):
    background_url = f'file://{os.path.abspath(background_path)}'
else:
    background_url = "https://www.transparenttextures.com/patterns/minimalist-lattice.png"  # ใช้ URL สำรอง

st.markdown(f"""
    <style>
        /* พื้นหลังธีมมืด */
        body {{
            background-color: #121212 !important;
            color: #E0E0E0 !important;
            font-family: 'Arial', sans-serif;
        }}

        /* ปรับสไตล์ของปุ่มให้เป็นสีแดง */
        .stButton>button {{
            background-color: #FF6347;  /* สีแดง */
            color: white;  /* ตัวอักษรสีขาว */
            border: none;
            border-radius: 10px;
            padding: 15px 30px;
            font-size: 18px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
            transition: background-color 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: #FF4500;  /* สีแดงเข้มเมื่อ hover */
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.7);  /* เพิ่มเงา */
        }}

        /* สไตล์ของกรอบที่แสดงโน้ต ให้เป็นสีดำ */
        .note-box {{
            text-align: center;
            border: 2px solid #BB86FC;
            padding: 20px;
            border-radius: 10px;
            background-color: #333333;
            font-size: 24px;
            font-weight: bold;
            color: #000000;  /* ตัวอักษรสีดำ */
            transition: background-color 0.3s ease;
        }}

        /* เปลี่ยนสีกรอบเมื่อมีเสียง */
        .note-box.sound {{
            background-color: #03DAC5;
            border-color: #018786;
            color: #018786;
        }}
        /* เปลี่ยนสีกรอบเมื่อไม่มีเสียง */
        .note-box.silence {{
            background-color: #6200EE;
            border-color: #3700B3;
            color: #BB86FC;
        }}

        /* ปรับปุ่มตำแหน่งให้ตรงกลาง */
        .stButton {{
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
        }}
    </style>
""", unsafe_allow_html=True)



# ปุ่มเริ่ม/หยุดรับเสียงที่อยู่กลาง X
if st.button("เริ่ม/หยุดรับเสียง", on_click=toggle_recording):
    if st.session_state.is_recording:
        st.write("🎤 กำลังรับเสียง...")
    else:
        st.write("🎤 หยุดรับเสียง")

# ถ้ากำลังรับเสียง
if st.session_state.is_recording:
    try:
        while True:
            note_pred = predict_note()
            if note_pred is not None:
                # ใช้ st.markdown เพื่อเพิ่มสไตล์ CSS สำหรับกรอบและการจัดตำแหน่ง
                note_placeholder.markdown(f"""
                <div style="text-align: center;">
                    <div style="border: 2px solid #BB86FC; padding: 20px; border-radius: 10px; background-color: #333333; color: #FFFFFF;">
                        🎶 โน้ตเสียงขณะนี้: <strong>{note_pred}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                note_placeholder.markdown("""
                <div style="text-align: center;">
                    <div style="border: 2px solid #BB86FC; padding: 20px; border-radius: 10px; background-color: #333333; color: #FFFFFF;">
                        🔇 ไม่มีโน้ตเสียงขณะนี้
                    </div>
                </div>
                """, unsafe_allow_html=True)

            time.sleep(0.05)  # ช่วยป้องกันการทำงานเร็วเกินไป
    except KeyboardInterrupt:
        st.write("🔴 หยุดการทำงาน")
        stream.stop_stream()
        stream.close()
        audio.terminate()
# ถ้าไม่ได้รับเสียง
else:
    note_placeholder.markdown("""
    <div style="text-align: center;">
        <div style="border: 2px solid black; padding: 20px; border-radius: 10px; background-color: #f0f0f0;">
            🎤 ลองร้องเพลงดูสิฉันจะทายเสียงโน๊ตดู!!
        </div>
    </div>
    """, unsafe_allow_html=True)
