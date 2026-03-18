import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import sys
import os
import copy

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
    """إعادة تشغيل الصفحة بشكل متوافق مع جميع إصدارات Streamlit"""
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

def draw_sudoku_image(grid, original_grid=None):
    """رسم شبكة السودوكو كصورة مع تلوين الأرقام الأصلية والمحلولة"""
    cell = 60
    size = cell * 9
    img = np.ones((size, size, 3), dtype=np.uint8) * 255

    # تلوين خلفية المربعات 3×3 بالتناوب
    for br in range(3):
        for bc in range(3):
            if (br + bc) % 2 == 0:
                y1, x1 = br * 3 * cell, bc * 3 * cell
                cv2.rectangle(img, (x1, y1), (x1 + 3 * cell, y1 + 3 * cell), (240, 240, 248), -1)

    # رسم خطوط الشبكة
    for i in range(10):
        thick = 3 if i % 3 == 0 else 1
        p = i * cell
        cv2.line(img, (0, p), (size, p), (0, 0, 0), thick)
        cv2.line(img, (p, 0), (p, size), (0, 0, 0), thick)

    # كتابة الأرقام
    font = cv2.FONT_HERSHEY_SIMPLEX
    for y in range(9):
        for x in range(9):
            val = int(grid[y][x])
            if val != 0:
                is_orig = (original_grid is not None and int(original_grid[y][x]) != 0)
                # أسود = أصلي ، أخضر = محلول
                color = (0, 0, 0) if is_orig else (0, 140, 0)
                txt = str(val)
                ts = cv2.getTextSize(txt, font, 1.3, 2)[0]
                tx = x * cell + (cell - ts[0]) // 2
                ty = y * cell + (cell + ts[1]) // 2
                cv2.putText(img, txt, (tx, ty), font, 1.3, color, 2, cv2.LINE_AA)

    return img

# ─── تحميل نموذج الذكاء الاصطناعي ───
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

# ─── معالجة الصورة ───
def process_image_binary(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 4
    )

def find_grid_contour(binary):
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(
        [c for c in contours if cv2.contourArea(c) > 1000],
        key=cv2.contourArea,
        reverse=True
    )
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

# ─── حل السودوكو بخوارزمية Backtracking ───
def _valid(g, r, c, n):
    if n in g[r]:
        return False
    if n in [g[i][c] for i in range(9)]:
        return False
    sr, sc = 3 * (r // 3), 3 * (c // 3)
    for i in range(sr, sr + 3):
        for j in range(sc, sc + 3):
            if g[i][j] == n:
                return False
    return True

def _backtrack(g):
    for i in range(9):
        for j in range(9):
            if g[i][j] == 0:
                for n in range(1, 10):
                    if _valid(g, i, j, n):
                        g[i][j] = n
                        if _backtrack(g):
                            return True
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
    if _backtrack(g):
        return g
    return None

# ══════════════════════════════════════════════
# إعدادات الصفحة و CSS الشبكة الاحترافية
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="SudokuSense AI",
    page_icon="🧩",
    layout="centered"
)

st.markdown("""
<style>
    /* تصغير العرض ليناسب شكل المربع */
    .block-container { max-width: 650px; }

    /* إخفاء تسميات الحقول (Labels) لتنظيف المظهر */
    div[data-testid="stTextInput"] label { display: none !important; }

    /* تنسيق حاويات 3x3 لتكوين الإطار السميك للشبكة */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border: 3px solid #111 !important;
        border-radius: 0px !important;
        padding: 1px !important;
        margin-bottom: 5px !important;
        background-color: #fcfcfc;
    }

    /* منع انهيار الأعمدة على شاشات الهواتف للحفاظ على شكل الشبكة */
    div[data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
    }

    /* تقليل المسافات بين المربعات الداخلية */
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="column"] {
        padding: 0px 1px !important;
    }

    /* تنسيق خانات الإدخال لتصبح مربعات سودوكو حقيقية */
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stTextInput"] input {
        text-align: center !important;
        font-size: 24px !important;
        font-weight: bold !important;
        padding: 0px !important;
        height: 50px !important;
        border: 1px solid #ddd !important;
        border-radius: 0px !important;
        color: #222 !important;
    }

    /* لون الخانة عند النقر عليها للتعديل */
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stTextInput"] input:focus {
        background-color: #d1e8ff !important;
        border: 2px solid #0056b3 !important;
        box-shadow: none !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🧩 SudokuSense AI")
st.markdown("ارفع صورة سودوكو **أو** أدخل الأرقام يدوياً ← عدّل ← اضغط **حل**")

# ─── Session State ───
for key, val in [('grid', None), ('solved', None), ('original', None)]:
    if key not in st.session_state:
        st.session_state[key] = val

model = load_ai_model()

# ╔═══════════════════════════════════════════╗
# ║ أقسام الإدخال                             ║
# ╚═══════════════════════════════════════════╝
tab_img, tab_manual = st.tabs(["📷 استخراج من صورة", "✏️ إدخال يدوي"])

with tab_img:
    uploaded = st.file_uploader("ارفع صورة سودوكو", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded:
        pil_img = Image.open(uploaded).convert('RGB')
        st.image(pil_img, caption="الصورة المرفوعة", width=320)
        if model is None:
            st.warning("⚠️ ملف النموذج غير موجود (model.keras / model.h5)")
        elif main_processing is None:
            st.warning("⚠️ وحدة processing غير متوفرة")
        else:
            if st.button("🔍 استخراج الأرقام", type="primary", use_container_width=True, key="btn_extract"):
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                with st.spinner("جاري المعالجة …"):
                    binary = process_image_binary(cv_img)
                    contour = find_grid_contour(binary)
                    if contour is None:
                        st.error("❌ لم يُعثر على شبكة سودوكو في الصورة")
                    else:
                        ok, imgs, _ = main_processing(contour, binary)
                        if not ok:
                            st.error("❌ فشل تقسيم الشبكة")
                        else:
                            g = recognize_digits(imgs, model)
                            st.session_state.grid = g
                            st.session_state.original = copy.deepcopy(g)
                            st.session_state.solved = None
                            st.success("✅ تم الاستخراج بنجاح!")
                            safe_rerun()

with tab_manual:
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("📝 شبكة فارغة", use_container_width=True, key="btn_empty"):
            st.session_state.grid = [[0] * 9 for _ in range(9)]
            st.session_state.original = [[0] * 9 for _ in range(9)]
            st.session_state.solved = None
            safe_rerun()
    with col_b:
        if st.button("🧪 مثال تجريبي", use_container_width=True, key="btn_example"):
            example = [
                [5, 3, 0, 0, 7, 0, 0, 0, 0],
                [6, 0, 0, 1, 9, 5, 0, 0, 0],
                [0, 9, 8, 0, 0, 0, 0, 6, 0],
                [8, 0, 0, 0, 6, 0, 0, 0, 3],
                [4, 0, 0, 8, 0, 3, 0, 0, 1],
                [7, 0, 0, 0, 2, 0, 0, 0, 6],
                [0, 6, 0, 0, 0, 0, 2, 8, 0],
                [0, 0, 0, 4, 1, 9, 0, 0, 5],
                [0, 0, 0, 0, 8, 0, 0, 7, 9],
            ]
            st.session_state.grid = example
            st.session_state.original = copy.deepcopy(example)
            st.session_state.solved = None
            safe_rerun()

    with st.expander("📋 لصق أرقام من نص"):
        txt = st.text_area(
            "9 أسطر — كل سطر 9 أرقام (0 = فارغ)",
            height=200,
            placeholder=(
                "530070000\n600195000\n098000060\n"
                "800060003\n400803001\n700020006\n"
                "060000280\n000419005\n000080079"
            ),
        )
        if st.button("⬇️ تحميل", key="btn_paste"):
            try:
                lines = [l.strip() for l in txt.strip().splitlines() if l.strip()]
                if len(lines) != 9:
                    st.error("❌ يجب إدخال 9 أسطر بالضبط")
                else:
                    g = []
                    for line in lines:
                        digits = [int(c) for c in line.replace(' ', '').replace(',', '') if c.isdigit()]
                        if len(digits) != 9:
                            st.error(f"❌ السطر «{line}» لا يحتوي 9 أرقام")
                            break
                        g.append(digits)
                    else:
                        st.session_state.grid = g
                        st.session_state.original = copy.deepcopy(g)
                        st.session_state.solved = None
                        safe_rerun()
            except Exception as e:
                st.error(f"❌ خطأ: {e}")

# ╔═══════════════════════════════════════════╗
# ║ شبكة السودوكو التفاعلية (التصميم الحقيقي)  ║
# ╚═══════════════════════════════════════════╝
if st.session_state.grid is not None and st.session_state.solved is None:
    st.markdown("---")
    st.subheader("📋 الشبكة الحالية")
    st.caption("اضغط على أي مربع لتعديل الرقم (اتركه فارغاً للخانة المجهولة)")

    edited_grid = [[0 for _ in range(9)] for _ in range(9)]
    
    # محاذاة الشبكة في المنتصف
    _, col_grid, _ = st.columns([1, 6, 1])
    
    with col_grid:
        # بناء 3 صفوف رئيسية للمربعات الكبيرة
        for big_row in range(3):
            # كل صف رئيسي يحتوي على 3 أعمدة (تمثل المربعات الكبيرة)
            big_cols = st.columns(3)
            for big_col in range(3):
                with big_cols[big_col]:
                    # إنشاء الإطار السميك للمربع 3x3
                    with st.container(border=True):
                        # بناء 3 صفوف صغيرة داخل المربع الكبير
                        for small_row in range(3):
                            small_cols = st.columns(3)
                            for small_col in range(3):
                                # حساب الإحداثيات الفعلية في المصفوفة 9x9
                                row = big_row * 3 + small_row
                                col = big_col * 3 + small_col
                                val = st.session_state.grid[row][col]
                                
                                # عرض الخانة فارغة إذا كانت 0
                                display_val = "" if val == 0 else str(val)
                                
                                with small_cols[small_col]:
                                    user_input = st.text_input(
                                        label=f"r{row}c{col}",
                                        value=display_val,
                                        key=f"cell_{row}_{col}"
                                    )
                                    # التحقق من أن الإدخال رقم من 1 إلى 9
                                    if user_input.isdigit() and 1 <= int(user_input) <= 9:
                                        edited_grid[row][col] = int(user_input)
                                    else:
                                        edited_grid[row][col] = 0

    st.markdown("<br>", unsafe_allow_html=True)
    
    # ── أزرار التحكم ──
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🚀 حل السودوكو", type="primary", use_container_width=True, key="btn_solve"):
            st.session_state.grid = edited_grid
            st.session_state.original = copy.deepcopy(edited_grid)
            with st.spinner("⚙️ جاري الحل …"):
                result = solve_puzzle(edited_grid)
                if result is not None:
                    st.session_state.solved = result
                    safe_rerun()
                else:
                    st.error("❌ الشبكة الحالية مستحيلة الحل! يرجى التحقق من الأرقام.")

    with c2:
        if st.button("🔄 إعادة تعيين", use_container_width=True, key="btn_reset"):
            st.session_state.grid = (
                copy.deepcopy(st.session_state.original)
                if st.session_state.original
                else [[0] * 9 for _ in range(9)]
            )
            safe_rerun()

    with c3:
        if st.button("🗑️ مسح الكل", use_container_width=True, key="btn_clear"):
            st.session_state.grid = None
            st.session_state.solved = None
            st.session_state.original = None
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

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔙 تعديل الشبكة", use_container_width=True):
            st.session_state.solved = None
            safe_rerun()
    with c2:
        _, img_enc = cv2.imencode('.png', result_img)
        st.download_button(
            "📥 تحميل صورة الحل",
            data=img_enc.tobytes(),
            file_name="sudoku_solved.png",
            mime="image/png",
            use_container_width=True
        )

    st.balloons()
