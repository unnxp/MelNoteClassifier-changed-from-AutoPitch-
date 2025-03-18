import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import time
import scipy.signal as signal
import pyaudio
import tempfile
import os

# ตั้งค่าการรับเสียง
SAMPLE_RATE = 22050
CHUNK = 1024  

# โหลดโมเดลแค่ครั้งเดียว
if 'model' not in st.session_state:
    st.session_state.model = tf.keras.models.load_model("MelNoteClassifierV6.h5")

# ฟังก์ชันแปลง MIDI เป็นชื่อโน้ต
def midi_to_note(midi):
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = midi // 12 - 1
    note = note_names[midi % 12]
    return f"{note}{octave}"

# ฟังก์ชันคำนวณค่า RMS
def rms_value(audio_data):
    return np.sqrt(np.mean(np.square(audio_data)))

# ฟังก์ชัน Noise Gate
def noise_gate(audio_data, threshold=0.02):
    return np.where(np.abs(audio_data) >= threshold, audio_data, 0)

# ฟังก์ชัน Bandpass Filter
def bandpass_filter(audio_data, low_cutoff=90, high_cutoff=8000, sample_rate=22050, order=3):
    nyquist = 0.5 * sample_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, audio_data)

def extract_mel_spectrogram(audio_data, sr=22050):
    """ แปลงเสียงเป็น Mel Spectrogram """
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128, fmax=8000)
    mel = np.maximum(mel, 0)  # ปรับให้ไม่มีค่าติดลบ

    # ทำให้ขนาดของ Mel Spectrogram เป็น (128, 128)
    mel_resized = librosa.util.fix_length(mel, size=128, axis=-1)  # padding หรือย่อขนาด
    mel_resized = librosa.util.fix_length(mel_resized, size=128, axis=-2)  # padding หรือย่อขนาด
    return mel_resized

# ฟังก์ชันทำนายโน้ตจาก Mel spectrogram
def predict_note(mel_input):
    if mel_input is None:
        return None
    
    # เพิ่มมิติของข้อมูลเป็น (1, 128, 128, 1)
    mel_input_resized = np.reshape(mel_input, (1, 128, 128, 1))
    
    predicted = st.session_state.model.predict(mel_input_resized)
    predicted_midi = np.argmax(predicted)
    return midi_to_note(predicted_midi)

# 🎵 UI ของ Streamlit
st.title("🎵 Note Detection (ไมโครโฟน & ไฟล์เสียง)")
st.write("เลือกวิธีวิเคราะห์เสียงที่ต้องการ")

# ตัวเลือกการรับเสียง
option = st.radio("เลือกแหล่งเสียง:", ("🎤 ไมโครโฟน", "📂 ไฟล์เสียง"))

# 🎤 ถ้าเลือกไมโครโฟน
if option == "🎤 ไมโครโฟน":
    st.write("กดปุ่มเพื่อเริ่มวิเคราะห์เสียงจากไมโครโฟน")

    # เปิดไมโครโฟน
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paFloat32, channels=1, rate=SAMPLE_RATE, 
                        input=True, frames_per_buffer=CHUNK)

    # เริ่มหรือหยุดการบันทึกเสียง
    if st.button("🎤 เริ่ม / หยุดรับเสียง"):
        st.session_state.is_recording = not st.session_state.get("is_recording", False)

    # แสดงผลโน้ตแบบสวยงาม
    note_placeholder = st.empty()

    if st.session_state.get("is_recording", False):
        st.write("🎙️ กำลังรับเสียง...")

        while True:
            audio_data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.float32)
            
            # คำนวณค่า RMS
            rms = rms_value(audio_data)
            
            # ตรวจสอบว่าเสียงเพียงพอหรือไม่
            if rms < 0.02:  # กำหนด threshold ของ RMS ที่เหมาะสม
                note_placeholder.markdown(""" 
                <div style="text-align: center;">
                    <div style="border: 2px solid #BB86FC; padding: 20px; border-radius: 10px; 
                                background-color: #333333; color: #FFFFFF; font-size: 24px;">
                        🔇 ไม่มีโน้ตเสียงขณะนี้
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                mel_spectrogram = extract_mel_spectrogram(audio_data)
                note = predict_note(mel_spectrogram)

                if note:
                    note_placeholder.markdown(f"""
                    <div style="text-align: center;">
                        <div style="border: 2px solid #BB86FC; padding: 20px; border-radius: 10px; 
                                    background-color: #333333; color: #FFFFFF; font-size: 24px;">
                            🎶 โน้ตเสียงขณะนี้: <strong>{note}</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    note_placeholder.markdown(""" 
                    <div style="text-align: center;">
                        <div style="border: 2px solid #BB86FC; padding: 20px; border-radius: 10px; 
                                    background-color: #333333; color: #FFFFFF; font-size: 24px;">
                            🔇 ไม่มีเสียงที่สามารถวิเคราะห์ได้
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            time.sleep(0.1)

        stream.stop_stream()
        stream.close()
        audio.terminate()

# 📂 ถ้าเลือกไฟล์เสียง
elif option == "📂 ไฟล์เสียง":
    uploaded_file = st.file_uploader("อัปโหลดไฟล์เสียง (wav, mp3)", type=["wav", "mp3"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_filename = temp_file.name  # เก็บชื่อไฟล์ชั่วคราว

        try:
            # โหลดไฟล์เสียง
            audio_data, _ = librosa.load(temp_filename, sr=SAMPLE_RATE)

            # ตรวจสอบว่ามีเสียงหรือไม่
            if len(audio_data) == 0 or np.max(np.abs(audio_data)) < 0.001:
                st.warning("🔇 ไม่พบเสียงที่สามารถวิเคราะห์ได้")
            else:
                # แปลงเป็น Mel Spectrogram
                mel_spectrogram = extract_mel_spectrogram(audio_data)
                
                # ทำนายโน้ตเสียง
                note = predict_note(mel_spectrogram)

                if note:
                    st.success(f"🎶 โน้ตเสียงจากไฟล์: {note}")
                else:
                    st.warning("🔇 ไม่สามารถวิเคราะห์โน้ตได้")

        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาดในการอ่านไฟล์: {e}")

        finally:
            # ลบไฟล์ชั่วคราว
            os.remove(temp_filename)

# 🎨 **CSS Styling**
st.markdown("""
    <style>
        .stButton>button {
            background-color: #FF6347;
            color: white;
            border: none;
            border-radius: 10px;
            padding: 15px 30px;
            font-size: 18px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.5);
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #FF4500;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.7);
        }
        .note-box {
            text-align: center;
            border: 2px solid #BB86FC;
            padding: 20px;
            border-radius: 10px;
            background-color: #333333;
            font-size: 24px;
            font-weight: bold;
            color: #FFFFFF;
            transition: background-color 0.3s ease;
        }
    </style>
""", unsafe_allow_html=True)
