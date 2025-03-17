# MelNoteClassifier

MelNoteClassifier เป็นเว็บแอปพลิเคชันสำหรับ **Pitch Classification** หรือการทำนายเสียงโน้ต (Pitch) ของเสียงแบบเรียลไทม์ โดยใช้ **Convolutional Neural Network (CNN)** เป็นโมเดลหลักในการวิเคราะห์ **Mel Spectrogram** ที่แปลงมาจากไฟล์เสียง

## วิธีการติดตั้งและใช้งาน

### 1. ดาวน์โหลดและติดตั้งโปรเจกต์

```
git clone <https://github.com/unnxp/MelNoteClassifier-changed-from-AutoPitch-.git>
cd MelNoteClassifier
```

### 2. ติดตั้ง Dependencies

**ใช้ Python เวอร์ชัน 3.10**

```
pip install -r requirements.txt
```

### 4. รันแอปพลิเคชัน

```
streamlit run app.py
```

หลังจากรันคำสั่งนี้ ระบบจะเปิดเว็บแอปพลิเคชันใน **localhost** โดยอัตโนมัติ (กด a

---

## คุณสมบัติหลักของแอปพลิเคชัน

-  รองรับการทำนายเสียงโน้ตแบบเรียลไทม์
-  ใช้ **CNN** ในการวิเคราะห์เสียง
-  แปลงเสียงเป็น **Mel Spectrogram** ก่อนนำไปวิเคราะห์
-  พัฒนาโดยใช้ **Streamlit** เพื่อสร้างอินเทอร์เฟซที่ใช้งานง่าย

---

## ข้อมูลเพิ่มเติม

- โมเดลที่ใช้สำหรับทำนายเสียงโน้ต: `MelNoteClassifierV6.h5`
- รองรับการใช้งานเฉพาะบนเครื่อง **Localhost**
- ต้องใช้ **ไมโครโฟน** ในการรับเสียงขณะใช้งาน



