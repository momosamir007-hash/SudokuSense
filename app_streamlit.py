import streamlit as st
import cv2
import numpy as np
from PIL import Image
import copy
import sys
import os

# ─── مسارات المشروع ───
sys.path.append('./ImageProcess')
sys.path.append('./Results')
sys.path.append('./Solver')

# ─── استيراد TensorFlow ───
try:
    import tensorflow as tf
except ImportError:
    st.error("⚠️ TensorFlow غير مثبت! قم بتثبيته: pip install tensorflow")
    st.stop()

# ─── استيراد وحدات المشروع (اختياري) ───
try:
    from processing import main_processing
except ImportError:
    main_processing = None
try:
    from solver import mainSolver
except ImportError:
    mainSolver = None

# ══════════════════════════════════════════════
# دوال مساعدة
# ══════════════════════════════════════════════
def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

def draw_sudoku_image(grid, original_grid=None):
    cell = 60
    size = cell * 9
    img = np.ones((size, size, 3), dtype=np.uint8) * 255

    for br in range(3):
        for bc in range(3):
            if (br + bc) % 2 == 0:
                y1, x1 = br * 3 * cell, bc * 3 * cell
                cv2.rectangle(img, (x1, y1), (x1 + 3 * cell, y1 + 3 * cell), (240, 240, 248), -1)

    for i in range(10):
        thick = 3 if i % 3 == 0 else 1
        p = i * cell
        cv2.line(img, (0, p), (size, p), (0, 0, 0), thick)
        cv2.line(img, (p, 0), (p, size), (0, 0, 0), thick)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for y in range(9):
        for x in range(9):
            val = int(grid[y][x])
            if val != 0:
                is_orig = (original_grid is not None and int(original_grid[y][x]) != 0)
                color = (0, 0, 0) if is_orig else (0, 140, 0)
                txt = str(val)
                ts = cv2.getTextSize(txt, font, 1.3, 2)[0]
                tx = x * cell + (cell - ts[0]) // 2
                ty = y * cell + (cell + ts[1]) // 2
                cv2.putText(img, txt, (tx, ty), font, 1.3, color, 2, cv2.LINE_AA)
    return img

@st.cache_resource
def load_ai_model():
    for path in ['model.keras', 'model.h5']:
        if os.path.exists(path):
            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
                tf.keras.layers.MaxPool2D(),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
                tf.keras.layers.MaxPool2D(strides=(2, 2)),
                tf.keras.layers.Dropout(0.25),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            model.load_weights(path)
            return model
    return None

def process_image_binary(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

def find_grid_contour(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted([c for c in contours if cv2.contourArea(c) > 1000], key=cv2.contourArea, reverse=True)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

def recognize_digits(images, model):
    grid = [[0] * 9 for _ in range(9)]
    for item in images:
        if item[3]:
            img = cv2.resize(item[0], (28, 28)).reshape(1, 28, 28, 1)
            pred = int(np.argmax(model.predict(img, verbose=0)[0]))
            grid[item[2]][item[1]] = pred
    return grid

def _valid(g, r, c, n):
    if n in g[r]: return False
    if n in [g[i][c] for i in range(9)]: return False
    sr, sc = 3 * (r // 3), 3 * (c // 3)
    for i in range(sr, sr + 3):
        for j in range(sc, sc + 3):
            if g[i][j] == n: return False
    return True

def _backtrack(g):
    for i in range(9):
        for j in range(9):
            if g[i][j] == 0:
                for n in range(1, 10):
                    if _valid(g, i, j, n):
                        g[i][j] = n
                        if _backtrack(g): return True
                        g[i][j] = 0
                return False
    return True

def solve_puzzle(grid):
    if mainSolver is not None:
        try:
            result = mainSolver(copy.deepcopy(grid))
            if result is not None and result != -1:
                return result if isinstance(result, list) else None
        except Exception:
            pass
    g = copy.deepcopy(grid)
    if _backtrack(g): return g
    return None

# ══════════════════════════════════════════════
# إعدادات الصفحة و CSS المخصص
# ══════════════════════════════════════════════
st.set_page_config(page_title="SudokuSense AI", page_icon="🧩", layout="centered")

# CSS لجعل حقول الإدخال تبدو كخلايا سودوكو متلاصقة
st.markdown("""
<style>
    /* تصغير حجم الحاوية لتناسب الشبكة */
    .block-container { max-width: 600px; }
    
    /* إخفاء أزرار الزيادة والنقصان في حقل الرقم */
    input[type=number]::-webkit-inner-spin-button, 
    input[type=number]::-webkit-outer-spin-button { 
        -webkit-appearance: none; 
        margin: 0; 
    }
    input[type=number] { 
        -moz-appearance: textfield; 
        text-align: center; 
        font-size: 24px; 
        font-weight: bold;
        padding: 0 !important;
        height: 50px !important;
    }
    
    /* تنسيق الخلايا لتكون متلاصقة ومربعة */
    div[data-testid="stNumberInput"] {
        margin: -2px; /* لتقليل المسافات بين الأعمدة */
    }
    
    /* تلوين الخلفية للمربعات 3x3 لتمييز الشبكة */
    .sudoku-cell input {
        border-radius: 0 !important;
        border: 1px solid #ccc !important;
    }
    /* جعل الخط فارغاً (0) غير مرئي لتبدو الخلية فارغة تماماً */
    .sudoku-empty input {
        color: transparent !important;
    }
    .sudoku-empty input:focus {
        color: inherit !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧩 SudokuSense AI")
st.markdown("ارفع صورة سودوكو **أو** أدخل الأرقام يدوياً ← عدّل ← اضغط **حل**")

# Session State
for key, val in [('grid', None), ('solved', None), ('original', None)]:
    if key not in st.session_state:
        st.session_state[key] = val

model = load_ai_model()

# ╔═══════════════════════════════════════════╗
# ║ الإدخال (صورة أو يدوي)                     ║
# ╚═══════════════════════════════════════════╝
tab_img, tab_manual = st.tabs(["📷 استخراج من صورة", "✏️ إدخال يدوي"])

with tab_img:
    uploaded = st.file_uploader("ارفع صورة سودوكو", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded:
        pil_img = Image.open(uploaded).convert('RGB')
        st.image(pil_img, caption="الصورة المرفوعة", width=250)
        if st.button("🔍 استخراج الأرقام", type="primary", use_container_width=True):
            if model is None:
                st.error("⚠️ ملف النموذج غير موجود")
            else:
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                with st.spinner("جاري المعالجة …"):
                    binary = process_image_binary(cv_img)
                    contour = find_grid_contour(binary)
                    if contour is None:
                        st.error("❌ لم يُعثر على شبكة")
                    else:
                        ok, imgs, _ = main_processing(contour, binary)
                        if ok:
                            g = recognize_digits(imgs, model)
                            st.session_state.grid = g
                            st.session_state.original = copy.deepcopy(g)
                            st.session_state.solved = None
                            safe_rerun()

with tab_manual:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📝 شبكة فارغة", use_container_width=True):
            st.session_state.grid = [[0] * 9 for _ in range(9)]
            st.session_state.original = [[0] * 9 for _ in range(9)]
            st.session_state.solved = None
            safe_rerun()
    with c2:
        if st.button("🧪 مثال جاهز", use_container_width=True):
            example = [
                [5, 3, 0, 0, 7, 0, 0, 0, 0], [6, 0, 0, 1, 9, 5, 0, 0, 0], [0, 9, 8, 0, 0, 0, 0, 6, 0],
                [8, 0, 0, 0, 6, 0, 0, 0, 3], [4, 0, 0, 8, 0, 3, 0, 0, 1], [7, 0, 0, 0, 2, 0, 0, 0, 6],
                [0, 6, 0, 0, 0, 0, 2, 8, 0], [0, 0, 0, 4, 1, 9, 0, 0, 5], [0, 0, 0, 0, 8, 0, 0, 7, 9],
            ]
            st.session_state.grid = example
            st.session_state.original = copy.deepcopy(example)
            st.session_state.solved = None
            safe_rerun()

# ╔═══════════════════════════════════════════╗
# ║ شبكة السودوكو التفاعلية (التصميم الجديد)   ║
# ╚═══════════════════════════════════════════╝
if st.session_state.grid is not None and st.session_state.solved is None:
    st.markdown("---")
    st.subheader("📋 تعديل الشبكة")
    st.caption("أدخل الأرقام في المربعات (اترك المربع فارغاً أو 0 للخانة المجهولة)")

    # إنشاء واجهة شبكية باستخدام الأعمدة
    edited_grid = [[0 for _ in range(9)] for _ in range(9)]
    
    # حاوية للشبكة بخلفية لتبدو كإطار
    with st.container():
        for row in range(9):
            # فواصل أفقية سميكة كل 3 صفوف
            if row % 3 == 0 and row != 0:
                st.markdown("<hr style='margin: 0px; border-top: 3px solid #333;'>", unsafe_allow_html=True)
            
            cols = st.columns(9)
            for col in range(9):
                val = st.session_state.grid[row][col]
                # جعل الخانة الفارغة تعرض كفراغ بدلاً من صفر لتسهيل القراءة
                display_val = "" if val == 0 else val
                
                with cols[col]:
                    # تطبيق CSS مخصص لإخفاء الصفر
                    css_class = "sudoku-empty" if val == 0 else "sudoku-cell"
                    
                    # الفواصل العمودية السميكة (لا يمكن عملها بالـ CSS بسهولة في Streamlit، لكننا نعتمد على التقارب)
                    user_input = st.text_input(
                        label=f"r{row}c{col}", 
                        value=str(display_val), 
                        label_visibility="collapsed",
                        key=f"cell_{row}_{col}"
                    )
                    
                    # التحقق من الإدخال (يجب أن يكون رقماً من 1 إلى 9 أو فارغاً)
                    if user_input.isdigit() and 1 <= int(user_input) <= 9:
                        edited_grid[row][col] = int(user_input)
                    else:
                        edited_grid[row][col] = 0

    st.markdown("<br>", unsafe_allow_html=True)
    
    # أزرار التحكم
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🚀 حل السودوكو", type="primary", use_container_width=True):
            st.session_state.grid = edited_grid
            st.session_state.original = copy.deepcopy(edited_grid)
            with st.spinner("⚙️ جاري الحل …"):
                result = solve_puzzle(edited_grid)
                if result is not None:
                    st.session_state.solved = result
                    safe_rerun()
                else:
                    st.error("❌ الشبكة غير قابلة للحل! تحقق من الأرقام.")
    with c2:
        if st.button("🔄 إعادة تعيين", use_container_width=True):
            st.session_state.grid = copy.deepcopy(st.session_state.original)
            safe_rerun()
    with c3:
        if st.button("🗑️ مسح الكل", use_container_width=True):
            st.session_state.grid = [[0]*9 for _ in range(9)]
            safe_rerun()

# ╔═══════════════════════════════════════════╗
# ║ عرض النتيجة النهائية                       ║
# ╚═══════════════════════════════════════════╝
if st.session_state.solved is not None:
    st.markdown("---")
    st.subheader("✅ الحل النهائي")

    result_img = draw_sudoku_image(st.session_state.solved, st.session_state.original)
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:
        st.image(result_rgb, caption="🟢 أخضر = أرقام محلولة ⚫ أسود = أصلي", use_container_width=True)

    if st.button("🔙 تعديل الشبكة مرة أخرى", use_container_width=True):
        st.session_state.solved = None
        safe_rerun()

    st.balloons()
