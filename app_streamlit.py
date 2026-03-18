import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
import os
import tensorflow as tf

# 1. إضافة مسارات مجلدات المشروع لكي يتعرف عليها بايثون
sys.path.append('./ImageProcess')
sys.path.append('./Results')
sys.path.append('./Solver')

# 2. استيراد دوال المشروع الأصلية
try:
    from processing import main_processing
    from drawGrid import mainDraw
    from solver import mainSolver
except ImportError as e:
    st.error(f"⚠️ خطأ في الاستيراد: تأكد من تشغيل التطبيق من المجلد الرئيسي للمشروع. التفاصيل: {e}")
    st.stop()

# 3. تحميل النموذج بطريقة "الأوزان فقط" لتجنب خطأ TypeError
@st.cache_resource
def load_ai_model():
    # التحقق من وجود الملف (سواء كان بصيغة keras أو h5)
    model_path = 'model.keras'
    if not os.path.exists(model_path):
        model_path = 'model.h5'
        if not os.path.exists(model_path):
            st.error("⚠️ لم يتم العثور على ملف النموذج (model.keras أو model.h5)!")
            st.stop()

    # بناء هيكل النموذج يدوياً (مطابق تماماً لما تم تدريبه في ملف AI.py)
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(strides=(2,2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # تحميل "الأوزان" فقط بدلاً من تحميل الملف بالكامل
    model.load_weights(model_path)
    return model

# --- دوال المعالجة المستخرجة من واجهة المستخدم القديمة ---

def process_image_binary(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)
    return binary

def find_grid(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

def recognize_digits(images, model):
    grid = [[0 for _ in range(9)] for _ in range(9)]
    L = len(images)
    for i in range(L):
        x = images[i][1]
        y = images[i][2]
        if images[i][3]:
            image = images[i][0]
            image = cv2.resize(image, (28, 28))
            image = np.array(image)
            image_digit = image.reshape(1, 28, 28, 1)
            # توقع الرقم باستخدام النموذج
            prediction = np.argmax(model.predict(image_digit, verbose=0)[0])
            grid[y][x] = prediction
    return grid

# --- بناء واجهة Streamlit ---

st.set_page_config(page_title="SudokuSense AI", page_icon="🧩", layout="centered")

st.title("🧩 SudokuSense: الذكاء الاصطناعي")
st.markdown("قم برفع صورة للغز سودوكو (مكتوب بخط اليد) وسيقوم البرنامج بحله واستخراج النتيجة.")

# تحميل النموذج في الخلفية
model = load_ai_model()

# أداة رفع الصورة
uploaded_file = st.file_uploader("اختر صورة السودوكو...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة الأصلية
    pil_image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_image, caption='الصورة الأصلية', use_column_width=True)
    
    with col2:
        if st.button('🚀 ابدأ الحل', type="primary", use_container_width=True):
            
            # تحويل الصورة لصيغة OpenCV
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            with st.status("جاري المعالجة...", expanded=True) as status:
                try:
                    # الخطوة 1 و 2
                    st.write("🔍 معالجة الصورة والبحث عن الشبكة...")
                    binary = process_image_binary(cv_image)
                    grid_contour = find_grid(binary)
                    
                    if grid_contour is None:
                        status.update(label="فشل المعالجة", state="error", expanded=True)
                        st.error("لم أتمكن من العثور على شبكة سودوكو واضحة في الصورة.")
                        st.stop()

                    # الخطوة 3
                    st.write("✂️ قص الشبكة واستخراج الخانات...")
                    process_status, images_list, cut_image = main_processing(grid_contour, binary)
                    
                    if not process_status:
                        status.update(label="فشل القص", state="error", expanded=True)
                        st.error("حدث خطأ أثناء تقسيم الشبكة.")
                        st.stop()

                    # الخطوة 4
                    st.write("🤖 قراءة الأرقام بالذكاء الاصطناعي...")
                    grid_numbers = recognize_digits(images_list, model)

                    # الخطوة 5
                    st.write("⚙️ تشغيل خوارزمية الحل...")
                    result_grid = mainSolver(grid_numbers)
                    
                    if result_grid == -1:
                        st.warning("الشبكة المستخرجة غير قابلة للحل (قد يكون الذكاء الاصطناعي أخطأ في قراءة رقم).")
                        mainDraw(grid_numbers)
                        status.update(label="اكتمل مع وجود أخطاء", state="complete", expanded=False)
                    else:
                        st.write("✅ تم إيجاد الحل! جاري الرسم...")
                        mainDraw(result_grid)
                        status.update(label="تم الحل بنجاح!", state="complete", expanded=False)

                    # الخطوة 6: عرض النتيجة
                    result_image_path = "sudoku_completed.jpg"
                    if os.path.exists(result_image_path):
                        final_img = Image.open(result_image_path)
                        st.image(final_img, caption="الشبكة المحلولة", use_column_width=True)
                        st.success("🎉 اكتملت العملية بنجاح!")
                    else:
                        st.error("تعذر العثور على صورة النتيجة النهائية.")

                except Exception as e:
                    status.update(label="حدث خطأ", state="error", expanded=True)
                    st.error(f"تفاصيل الخطأ: {e}")
