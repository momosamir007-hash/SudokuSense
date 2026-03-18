import streamlit as st
import streamlit.components.v1 as components
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

# ══════════════════════════════════════════════════════════════
# دوال مساعدة
# ══════════════════════════════════════════════════════════════
def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

# ─────────── عرض HTML للنتيجة بشكل شبكة سودوكو حقيقية ───────────
def render_sudoku_html(grid, original_grid=None, editable=False, title=""):
    """
    شبكة سودوكو احترافية بـ HTML / CSS
    - حدود سميكة للمربعات 3×3
    - ألوان مختلفة للأرقام الأصلية والمحلولة
    - تأثيرات hover
    """
    html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');
    .sudoku-wrapper {{
        display: flex;
        flex-direction: column;
        align-items: center;
        font-family: 'Poppins', sans-serif;
        padding: 10px;
    }}
    .sudoku-title {{
        font-size: 18px;
        font-weight: 700;
        color: #1a237e;
        margin-bottom: 12px;
    }}
    .sudoku-board {{
        border-collapse: collapse;
        border: 4px solid #1a237e;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(26,35,126,0.18);
        background: #fff;
    }}
    .sudoku-board td {{
        width: 56px;
        height: 56px;
        text-align: center;
        vertical-align: middle;
        font-size: 28px;
        font-weight: 700;
        border: 1px solid #cfd8dc;
        transition: background 0.15s;
        position: relative;
    }}
    .sudoku-board td:hover {{
        background: #fff9c4 !important;
    }}
    /* ─── حدود المربعات 3×3 ─── */
    .sudoku-board td.border-right {{
        border-right: 3px solid #1a237e !important;
    }}
    .sudoku-board td.border-bottom {{
        border-bottom: 3px solid #1a237e !important;
    }}
    .sudoku-board td.border-left {{
        border-left: 3px solid #1a237e !important;
    }}
    .sudoku-board td.border-top {{
        border-top: 3px solid #1a237e !important;
    }}
    /* ─── تلوين الخلايا ─── */
    .sudoku-board td.cell-original {{
        color: #1a237e;
        background: #e8eaf6;
    }}
    .sudoku-board td.cell-solved {{
        color: #1b5e20;
        background: #e8f5e9;
    }}
    .sudoku-board td.cell-empty {{
        color: #e0e0e0;
        background: #fafafa;
    }}
    /* ─── تلوين أرضية الكتل بالتناوب ─── */
    .sudoku-board td.block-shaded {{
        background: #f3f0ff;
    }}
    .sudoku-board td.block-shaded.cell-original {{
        background: #e0dcf7;
    }}
    .sudoku-board td.block-shaded.cell-solved {{
        background: #d7f0d9;
    }}
    /* ─── وسيلة الإيضاح ─── */
    .legend {{
        display: flex;
        gap: 24px;
        margin-top: 14px;
        font-size: 13px;
        color: #555;
    }}
    .legend-item {{
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .legend-box {{
        width: 20px;
        height: 20px;
        border-radius: 4px;
        border: 1px solid #bbb;
    }}
    </style>
    <div class="sudoku-wrapper">
    """
    if title:
        html += f'<div class="sudoku-title">{title}</div>'
    html += '<table class="sudoku-board">'
    for r in range(9):
        html += "<tr>"
        for c in range(9):
            val = int(grid[r][c])
            classes = []
            # حدود المربعات 3×3
            if c % 3 == 0 and c > 0:
                classes.append("border-left")
            if c % 3 == 2 and c < 8:
                classes.append("border-right")
            if r % 3 == 0 and r > 0:
                classes.append("border-top")
            if r % 3 == 2 and r < 8:
                classes.append("border-bottom")
            # كتلة مظللة بالتناوب
            block_r, block_c = r // 3, c // 3
            if (block_r + block_c) % 2 == 0:
                classes.append("block-shaded")
            # نوع الخلية
            if val == 0:
                classes.append("cell-empty")
                text = ""
            elif original_grid and int(original_grid[r][c]) != 0:
                classes.append("cell-original")
                text = str(val)
            else:
                classes.append("cell-solved")
                text = str(val)
            cls = " ".join(classes)
            html += f'<td class="{cls}">{text}</td>'
        html += "</tr>"
    html += "</table>"
    # وسيلة الإيضاح
    if original_grid:
        html += """
        <div class="legend">
            <div class="legend-item">
                <div class="legend-box" style="background:#e8eaf6;"></div>
                <span style="color:#1a237e;font-weight:700;">أرقام أصلية</span>
            </div>
            <div class="legend-item">
                <div class="legend-box" style="background:#e8f5e9;"></div>
                <span style="color:#1b5e20;font-weight:700;">أرقام محلولة</span>
            </div>
        </div>
        """
    html += "</div>"
    return html

# ─────────── رسم صورة السودوكو بـ OpenCV ───────────
def draw_sudoku_image(grid, original_grid=None):
    cell = 64
    size = cell * 9
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    for br in range(3):
        for bc in range(3):
            if (br + bc) % 2 == 0:
                y1, x1 = br * 3 * cell, bc * 3 * cell
                cv2.rectangle(img, (x1, y1), (x1 + 3 * cell, y1 + 3 * cell), (235, 232, 248), -1)
    for i in range(10):
        thick = 4 if i % 3 == 0 else 1
        color = (26, 35, 126) if i % 3 == 0 else (180, 180, 180)
        p = i * cell
        cv2.line(img, (0, p), (size, p), color, thick)
        cv2.line(img, (p, 0), (p, size), color, thick)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for y in range(9):
        for x in range(9):
            val = int(grid[y][x])
            if val != 0:
                is_orig = (original_grid is not None and int(original_grid[y][x]) != 0)
                color = (126, 35, 26) if is_orig else (30, 120, 30)
                txt = str(val)
                ts = cv2.getTextSize(txt, font, 1.4, 2)[0]
                tx = x * cell + (cell - ts[0]) // 2
                ty = y * cell + (cell + ts[1]) // 2
                cv2.putText(img, txt, (tx, ty), font, 1.4, color, 2, cv2.LINE_AA)
    return img

# ─────────── تحميل نموذج AI ───────────
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

# ─────────── معالجة الصورة ───────────
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

# ─────────── حل Backtracking (احتياطي) ───────────
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

# ══════════════════════════════════════════════════════════════
# واجهة Streamlit الرئيسية
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SudokuSense AI",
    page_icon="🧩",
    layout="centered"
)

# ─── CSS العام + تنسيق شبكة التعديل ───
st.markdown("""
<style>
.block-container {
    max-width: 820px;
}
/* ===== تنسيق نموذج الشبكة ===== */
div[data-testid="stForm"] {
    border: 4px solid #1a237e !important;
    border-radius: 14px !important;
    padding: 22px 18px !important;
    background: linear-gradient(145deg, #f8f9ff 0%, #eef0ff 100%) !important;
    box-shadow: 0 6px 24px rgba(26,35,126,0.12) !important;
}
/* خلايا الإدخال */
div[data-testid="stForm"] div[data-testid="stTextInput"] input {
    text-align: center !important;
    font-size: 26px !important;
    font-weight: 800 !important;
    color: #1a237e !important;
    height: 54px !important;
    border: 1.5px solid #b0bec5 !important;
    border-radius: 6px !important;
    padding: 0 !important;
    background: #ffffff !important;
    box-shadow: inset 0 2px 4px rgba(0,0,0,0.06) !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stForm"] div[data-testid="stTextInput"] input:hover {
    background: #fffde7 !important;
    border-color: #ffc107 !important;
    transform: scale(1.04);
}
div[data-testid="stForm"] div[data-testid="stTextInput"] input:focus {
    border-color: #1565c0 !important;
    box-shadow: 0 0 0 3px rgba(21,101,192,0.3) !important;
    background: #e3f2fd !important;
    transform: scale(1.06);
}
div[data-testid="stForm"] div[data-testid="stTextInput"] input::placeholder {
    color: #d0d0d0 !important;
    font-size: 20px !important;
}
/* تقليل المسافات */
div[data-testid="stForm"] div[data-testid="stVerticalBlock"] > div {
    gap: 0px !important;
}
/* فواصل الكتل الأفقية */
.block-divider {
    border: none;
    height: 4px;
    background: linear-gradient(90deg, transparent, #1a237e, transparent);
    margin: 5px 0 7px 0;
    border-radius: 4px;
}
/* عنوان الشبكة */
.grid-header {
    text-align: center;
    font-size: 15px;
    color: #5c6bc0;
    margin-bottom: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ─── عنوان التطبيق ───
st.markdown("""
<div style="text-align:center; padding: 10px 0 5px 0;">
    <h1 style="color:#1a237e; margin:0;">🧩 SudokuSense AI</h1>
    <p style="color:#666; font-size:16px; margin-top:6px;">
        ارفع صورة سودوكو <b>أو</b> أدخل الأرقام يدوياً ← عدّل ← اضغط <b>حل</b>
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Session State ───
for k, v in [('grid', None), ('solved', None), ('original', None)]:
    if k not in st.session_state:
        st.session_state[k] = v

model = load_ai_model()

# ╔═══════════════════════════════════════════════╗
# ║ أقسام الإدخال ║
# ╚═══════════════════════════════════════════════╝
tab_img, tab_manual = st.tabs(["📷 استخراج من صورة", "✏️ إدخال يدوي"])

# ── تبويب الصورة ──
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
                            st.success("✅ تم الاستخراج — عدّل ثم اضغط حل")
                            safe_rerun()

# ── تبويب الإدخال اليدوي ──
with tab_manual:
    ca, cb = st.columns(2)
    with ca:
        if st.button("📝 شبكة فارغة", use_container_width=True, key="btn_empty"):
            st.session_state.grid = [[0] * 9 for _ in range(9)]
            st.session_state.original = [[0] * 9 for _ in range(9)]
            st.session_state.solved = None
            safe_rerun()
    with cb:
        if st.button("🧪 مثال تجريبي", use_container_width=True, key="btn_example"):
            ex = [
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
            st.session_state.grid = ex
            st.session_state.original = copy.deepcopy(ex)
            st.session_state.solved = None
            safe_rerun()

    with st.expander("📋 لصق أرقام من نص"):
        txt = st.text_area(
            "9 أسطر — كل سطر 9 أرقام (0 = فارغ)",
            height=200,
            placeholder="530070000\n600195000\n098000060\n"
                        "800060003\n400803001\n700020006\n"
                        "060000280\n000419005\n000080079"
        )
        if st.button("⬇️ تحميل", key="btn_paste"):
            try:
                lines = [l.strip() for l in txt.strip().splitlines() if l.strip()]
                if len(lines) != 9:
                    st.error("❌ يجب إدخال 9 أسطر بالضبط")
                else:
                    g = []
                    for line in lines:
                        digits = [int(ch) for ch in line.replace(' ', '').replace(',', '') if ch.isdigit()]
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
                st.error(f"❌ {e}")

# ╔═══════════════════════════════════════════════╗
# ║ شبكة السودوكو التفاعلية ║
# ╚═══════════════════════════════════════════════╝
if st.session_state.grid is not None:
    st.markdown("---")
    st.subheader("📋 شبكة السودوكو")

    # ── معاينة HTML (للقراءة فقط) ──
    preview_html = render_sudoku_html(
        st.session_state.grid,
        title="الشبكة الحالية — عدّل أدناه"
    )
    components.html(preview_html, height=580, scrolling=False)

    # ── نموذج التعديل ──
    st.markdown(
        '<p class="grid-header">'
        '⬇️ عدّل الأرقام هنا (اتركها فارغة = 0) ثم اضغط <b>🚀 حل</b>'
        '</p>',
        unsafe_allow_html=True
    )

    edited = [[0] * 9 for _ in range(9)]
    with st.form("sudoku_form", clear_on_submit=False):
        for block_row in range(3):
            for local_row in range(3):
                row = block_row * 3 + local_row
                # 9 خلايا + 2 فاصل عمودي = 11 عمود
                widths = [1, 1, 1, 0.1, 1, 1, 1, 0.1, 1, 1, 1]
                cols = st.columns(widths, gap="small")
                mapping = [0, 1, 2, 4, 5, 6, 8, 9, 10]
                for c in range(9):
                    with cols[mapping[c]]:
                        v = st.session_state.grid[row][c]
                        disp = str(v) if v != 0 else ""
                        res = st.text_input(
                            label="​",  # مسافة صفرية
                            value=disp,
                            key=f"sc_{row}_{c}",
                            max_chars=1,
                            label_visibility="collapsed",
                            placeholder="·"
                        )
                        if res and res.strip().isdigit():
                            d = int(res.strip())
                            edited[row][c] = d if 1 <= d <= 9 else 0
                        else:
                            edited[row][c] = 0
            # فاصل أفقي بين كتل 3×3
            if block_row < 2:
                st.markdown('<hr class="block-divider">', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # أزرار التحكم
        b1, b2, b3 = st.columns(3)
        with b1:
            solve_btn = st.form_submit_button("🚀 حل السودوكو", type="primary", use_container_width=True)
        with b2:
            reset_btn = st.form_submit_button("🔄 إعادة تعيين", use_container_width=True)
        with b3:
            clear_btn = st.form_submit_button("🗑️ مسح الكل", use_container_width=True)

        # ── معالجة الأزرار ──
        if solve_btn:
            ok = True
            for r in range(9):
                for c in range(9):
                    if not (0 <= edited[r][c] <= 9):
                        st.error(f"❌ قيمة غير صالحة في ({r + 1},{c + 1})")
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                st.session_state.grid = edited
                st.session_state.original = copy.deepcopy(edited)
                with st.spinner("⚙️ جاري الحل …"):
                    result = solve_puzzle(edited)
                    if result:
                        st.session_state.solved = result
                        safe_rerun()
                    else:
                        st.error("❌ الشبكة غير قابلة للحل! تحقق من الأرقام.")

        if reset_btn:
            st.session_state.grid = (
                copy.deepcopy(st.session_state.original)
                if st.session_state.original
                else [[0] * 9 for _ in range(9)]
            )
            st.session_state.solved = None
            safe_rerun()

        if clear_btn:
            st.session_state.grid = None
            st.session_state.solved = None
            st.session_state.original = None
            safe_rerun()

# ╔═══════════════════════════════════════════════╗
# ║ عرض النتيجة النهائية ║
# ╚═══════════════════════════════════════════════╝
if st.session_state.solved is not None:
    st.markdown("---")
    st.subheader("✅ الحل النهائي")

    # عرض HTML احترافي
    solved_html = render_sudoku_html(
        st.session_state.solved,
        st.session_state.original,
        title="🎉 تم الحل بنجاح!"
    )
    components.html(solved_html, height=620, scrolling=False)

    # تحميل صورة الحل
    result_img = draw_sudoku_image(
        st.session_state.solved,
        st.session_state.original
    )
    _, enc = cv2.imencode('.png', result_img)
    st.download_button(
        "📥 تحميل صورة الحل (PNG)",
        data=enc.tobytes(),
        file_name="sudoku_solved.png",
        mime="image/png",
        use_container_width=True
    )
    st.balloons()
    st.success("🎉 تم حل السودوكو بنجاح!")
