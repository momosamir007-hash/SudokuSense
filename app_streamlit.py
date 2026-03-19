import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
from PIL import Image
import sys
import os
import copy

sys.path.append('./ImageProcess')
sys.path.append('./Results')
sys.path.append('./Solver')

try:
    import tensorflow as tf
except ImportError:
    st.error("⚠️ TensorFlow غير مثبت!")
    st.stop()

try:
    from processing import main_processing
except ImportError:
    main_processing = None

try:
    from solver import mainSolver
except ImportError:
    mainSolver = None

# ══════════════════════════════════════════════════════════════
# ① تحسين معالجة صور الأرقام قبل إرسالها للنموذج
# ══════════════════════════════════════════════════════════════
def preprocess_digit_cell(cell_img):
    """
    معالجة محسّنة لصورة الخلية الواحدة:
    - إزالة ضوضاء الحواف
    - توسيط الرقم
    - تطبيع الحجم بنسق MNIST
    """
    if cell_img is None or cell_img.size == 0:
        return None, False

    # ── تحويل لرمادي ──
    if len(cell_img.shape) == 3:
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell_img.copy()

    # ── عتبة ثنائية ──
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ── إزالة ضوضاء حواف الخلية (قص 18%) ──
    h, w = thresh.shape
    margin_h = int(h * 0.18)
    margin_w = int(w * 0.18)
    cropped = thresh[margin_h:h - margin_h, margin_w:w - margin_w]

    # ── إزالة النقاط الصغيرة (ضوضاء) ──
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cropped = cv2.morphologyEx(cropped, cv2.MORPH_OPEN, kernel)

    # ── البحث عن الرقم ──
    coords = cv2.findNonZero(cropped)
    if coords is None:
        return None, False
    x, y, bw, bh = cv2.boundingRect(coords)
    digit = cropped[y:y + bh, x:x + bw]

    # ── هل هناك رقم فعلاً؟ ──
    pixel_ratio = np.sum(digit > 0) / max(digit.size, 1)
    if pixel_ratio < 0.06 or bh < 5 or bw < 3:
        return None, False

    # ── تربيع الصورة مع حشوة ──
    max_dim = max(bh, bw)
    pad_top = (max_dim - bh) // 2
    pad_bot = max_dim - bh - pad_top
    pad_left = (max_dim - bw) // 2
    pad_right = max_dim - bw - pad_left
    digit = cv2.copyMakeBorder(digit, pad_top, pad_bot, pad_left, pad_right,
                               cv2.BORDER_CONSTANT, value=0)

    # ── هامش خارجي (مثل صيغة MNIST) ──
    margin = int(max_dim * 0.35)
    digit = cv2.copyMakeBorder(digit, margin, margin, margin, margin,
                               cv2.BORDER_CONSTANT, value=0)

    # ── تصغير لـ 28×28 ──
    digit = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)

    # ── تنعيم خفيف لمحاكاة MNIST ──
    digit = cv2.GaussianBlur(digit, (3, 3), 0.5)

    # ── تطبيع ──
    digit = digit.astype(np.float32) / 255.0
    return digit, True

# ══════════════════════════════════════════════════════════════
# ② التعرف مع نسبة الثقة والبدائل
# ══════════════════════════════════════════════════════════════
# أزواج الأرقام المتشابهة (التي يخلط بينها النموذج)
CONFUSED_PAIRS = {1: [7], 7: [1], 9: [4], 4: [9], 5: [2], 2: [5]}

def recognize_with_confidence(images, model, conf_threshold=0.75):
    """
    يقرأ الأرقام مع حفظ:
    - نسبة الثقة لكل خلية
    - أفضل 3 توقعات بديلة
    - علامة "مشكوك فيه" إن كانت الثقة منخفضة
    """
    grid = [[0] * 9 for _ in range(9)]
    confidence = [[1.0] * 9 for _ in range(9)]
    alternatives = [[[] for _ in range(9)] for _ in range(9)]
    uncertain = set()

    for item in images:
        x_pos, y_pos = item[1], item[2]
        has_digit = item[3]
        if not has_digit:
            continue

        processed, valid = preprocess_digit_cell(item[0])
        if not valid:
            continue

        inp = processed.reshape(1, 28, 28, 1)
        probs = model.predict(inp, verbose=0)[0]

        # ── أفضل 3 توقعات ──
        top_indices = np.argsort(probs)[::-1]
        # استبعاد الصفر (ليس رقم سودوكو)
        top_indices = [i for i in top_indices if i != 0][:5]
        if len(top_indices) == 0:
            continue

        pred = top_indices[0]
        conf = float(probs[pred])
        grid[y_pos][x_pos] = pred
        confidence[y_pos][x_pos] = conf

        # حفظ البدائل
        alts = [(int(idx), float(probs[idx])) for idx in top_indices if idx != 0]
        alternatives[y_pos][x_pos] = alts

        # ── فحص الأزواج المتشابهة ──
        if pred in CONFUSED_PAIRS:
            for confused_digit in CONFUSED_PAIRS[pred]:
                alt_prob = float(probs[confused_digit])
                # إذا كان البديل المتشابه قريباً في الاحتمال
                if alt_prob > conf * 0.35:
                    uncertain.add((y_pos, x_pos))

        # ── ثقة منخفضة ──
        if conf < conf_threshold:
            uncertain.add((y_pos, x_pos))

    return grid, confidence, alternatives, uncertain

# ══════════════════════════════════════════════════════════════
# ③ التحقق بقواعد السودوكو والتصحيح التلقائي
# ══════════════════════════════════════════════════════════════
def find_conflicts(grid):
    """إيجاد جميع الخلايا المتعارضة حسب قواعد السودوكو"""
    conflicts = set()
    for i in range(9):
        # ── الصفوف ──
        seen = {}
        for j in range(9):
            v = grid[i][j]
            if v != 0:
                if v in seen:
                    conflicts.add((i, j))
                    conflicts.add((i, seen[v]))
                seen[v] = j

        # ── الأعمدة ──
        seen = {}
        for j in range(9):
            v = grid[j][i]
            if v != 0:
                if v in seen:
                    conflicts.add((j, i))
                    conflicts.add((seen[v], i))
                seen[v] = j

    # ── المربعات 3×3 ──
    for br in range(3):
        for bc in range(3):
            seen = {}
            for r in range(3):
                for c in range(3):
                    row, col = br * 3 + r, bc * 3 + c
                    v = grid[row][col]
                    if v != 0:
                        if v in seen:
                            conflicts.add((row, col))
                            conflicts.add(seen[v])
                        seen[v] = (row, col)

    return conflicts

def auto_correct(grid, confidence, alternatives):
    """
    تصحيح تلقائي: عند وجود تعارض → استبدال الخلية الأقل ثقة ببديلها
    """
    corrected = copy.deepcopy(grid)
    conf = copy.deepcopy(confidence)
    corrections = []  # سجل التصحيحات
    max_iter = 30

    for _ in range(max_iter):
        conflicts = find_conflicts(corrected)
        if not conflicts:
            break

        # أضعف خلية متعارضة
        worst = None
        worst_conf = 2.0
        for (r, c) in conflicts:
            if conf[r][c] < worst_conf:
                worst_conf = conf[r][c]
                worst = (r, c)

        if worst is None:
            break

        r, c = worst
        old_val = corrected[r][c]
        fixed = False

        for alt_val, alt_conf in alternatives[r][c]:
            if alt_val == old_val or alt_val == 0:
                continue
            corrected[r][c] = alt_val
            new_conflicts = find_conflicts(corrected)
            if len(new_conflicts) < len(conflicts):
                corrections.append({
                    'row': r,
                    'col': c,
                    'old': old_val,
                    'new': alt_val,
                    'old_conf': worst_conf,
                    'new_conf': alt_conf
                })
                conf[r][c] = alt_conf
                fixed = True
                break
            else:
                corrected[r][c] = old_val

        if not fixed:
            conf[r][c] = 999  # لا نعيد المحاولة

    return corrected, corrections

# ══════════════════════════════════════════════════════════════
# دوال عامة (حل + رسم + نموذج)
# ══════════════════════════════════════════════════════════════
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
            res = mainSolver(copy.deepcopy(grid))
            if res and res != -1 and isinstance(res, list):
                return res
        except Exception:
            pass
    g = copy.deepcopy(grid)
    return g if _backtrack(g) else None

# ─── التعديل هنا: دالة التحميل المحدثة ───
@st.cache_resource
def load_ai_model():
    model_path = 'model.keras'
    if os.path.exists(model_path):
        try:
            # تحميل النموذج بصيغته الحديثة مع تخطي التهيئة (compile=False) لتجنب أخطاء الإصدارات
            m = tf.keras.models.load_model(model_path, compile=False)
            return m
        except Exception as e:
            st.error(f"⚠️ خطأ أثناء تحميل النموذج: {e}")
            return None
    else:
        st.error("⚠️ ملف النموذج (model.keras) غير موجود!")
        return None
# ─────────────────────────────────────────

def process_image_binary(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 4)

def find_grid_contour(binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted([c for c in contours if cv2.contourArea(c) > 1000],
                      key=cv2.contourArea, reverse=True)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None

def draw_sudoku_image(grid, original_grid=None):
    cell = 64
    size = cell * 9
    img = np.ones((size, size, 3), dtype=np.uint8) * 255
    for br in range(3):
        for bc in range(3):
            if (br + bc) % 2 == 0:
                y1, x1 = br * 3 * cell, bc * 3 * cell
                cv2.rectangle(img, (x1, y1), (x1 + 3 * cell, y1 + 3 * cell),
                              (235, 232, 248), -1)
    for i in range(10):
        thick = 4 if i % 3 == 0 else 1
        clr = (26, 35, 126) if i % 3 == 0 else (180, 180, 180)
        p = i * cell
        cv2.line(img, (0, p), (size, p), clr, thick)
        cv2.line(img, (p, 0), (p, size), clr, thick)

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

# ══════════════════════════════════════════════════════════════
# ④ شبكة HTML تفاعلية مع تمييز الخلايا المشكوك فيها
# ══════════════════════════════════════════════════════════════
def render_sudoku_html(grid, original_grid=None, uncertain_cells=None,
                       corrected_cells=None, confidence=None, title=""):
    """شبكة HTML احترافية مع تلوين ذكي"""
    uncertain_cells = uncertain_cells or set()
    corrected_cells = corrected_cells or set()

    html = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');
    .sw{display:flex;flex-direction:column;align-items:center; font-family:'Poppins',sans-serif;padding:10px}
    .stitle{font-size:18px;font-weight:700;color:#1a237e;margin-bottom:12px}
    .sb{border-collapse:collapse;border:4px solid #1a237e; border-radius:8px;overflow:hidden; box-shadow:0 8px 32px rgba(26,35,126,.18);background:#fff}
    .sb td{width:56px;height:56px;text-align:center;vertical-align:middle; font-size:28px;font-weight:700;border:1px solid #cfd8dc; transition:background .15s;position:relative;cursor:default}
    .sb td:hover{background:#fff9c4!important}
    .sb td.br{border-right:3px solid #1a237e!important}
    .sb td.bb{border-bottom:3px solid #1a237e!important}
    .sb td.bl{border-left:3px solid #1a237e!important}
    .sb td.bt{border-top:3px solid #1a237e!important}
    .sb td.orig{color:#1a237e;background:#e8eaf6}
    .sb td.solved{color:#1b5e20;background:#e8f5e9}
    .sb td.empty{color:#e0e0e0;background:#fafafa}
    .sb td.uncertain{color:#e65100!important; background:#fff3e0!important; animation:pulse 1.5s infinite}
    .sb td.corrected{color:#6a1b9a!important; background:#f3e5f5!important}
    .sb td.shaded{background:#f3f0ff}
    .sb td.shaded.orig{background:#e0dcf7}
    .sb td.shaded.solved{background:#d7f0d9}
    .sb td.shaded.uncertain{background:#ffe0b2!important}
    .sb td.shaded.corrected{background:#e1bee7!important}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
    .sb td .conf{position:absolute;bottom:2px;right:3px; font-size:8px;color:#999;font-weight:400}
    .legend{display:flex;flex-wrap:wrap;gap:16px;margin-top:14px;font-size:12px;color:#555}
    .legend-item{display:flex;align-items:center;gap:5px}
    .legend-box{width:18px;height:18px;border-radius:3px;border:1px solid #bbb}
    </style>
    """
    html += '<div class="sw">'
    if title:
        html += f'<div class="stitle">{title}</div>'
    html += '<table class="sb">'

    for r in range(9):
        html += "<tr>"
        for c in range(9):
            val = int(grid[r][c])
            cls = []

            if c % 3 == 0 and c > 0:
                cls.append("bl")
            if c % 3 == 2 and c < 8:
                cls.append("br")
            if r % 3 == 0 and r > 0:
                cls.append("bt")
            if r % 3 == 2 and r < 8:
                cls.append("bb")
            if (r // 3 + c // 3) % 2 == 0:
                cls.append("shaded")

            # تحديد نوع الخلية
            if (r, c) in corrected_cells:
                cls.append("corrected")
            elif (r, c) in uncertain_cells:
                cls.append("uncertain")
            elif val == 0:
                cls.append("empty")
            elif original_grid and int(original_grid[r][c]) != 0:
                cls.append("orig")
            else:
                cls.append("solved")

            text = str(val) if val != 0 else ""

            # عرض نسبة الثقة
            conf_html = ""
            if confidence and val != 0:
                cv = confidence[r][c]
                if cv < 1.5:
                    conf_html = f'<span class="conf">{cv:.0%}</span>'

            html += f'<td class="{" ".join(cls)}">{text}{conf_html}</td>'
        html += "</tr>"
    html += "</table>"

    # وسيلة الإيضاح
    html += """
    <div class="legend">
        <div class="legend-item">
            <div class="legend-box" style="background:#e8eaf6"></div>
            <span><b style="color:#1a237e">أصلي</b></span>
        </div>
        <div class="legend-item">
            <div class="legend-box" style="background:#e8f5e9"></div>
            <span><b style="color:#1b5e20">محلول</b></span>
        </div>
        <div class="legend-item">
            <div class="legend-box" style="background:#fff3e0; animation:pulse 1.5s infinite"></div>
            <span><b style="color:#e65100">⚠️ مشكوك فيه</b></span>
        </div>
        <div class="legend-item">
            <div class="legend-box" style="background:#f3e5f5"></div>
            <span><b style="color:#6a1b9a">🔧 مُصحَّح تلقائياً</b></span>
        </div>
    </div>
    """
    html += "</div>"
    return html

# ══════════════════════════════════════════════════════════════
# واجهة Streamlit
# ══════════════════════════════════════════════════════════════
def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

st.set_page_config(page_title="SudokuSense AI", page_icon="🧩", layout="centered")

st.markdown("""
<style>
.block-container{max-width:820px}
div[data-testid="stForm"]{
    border:4px solid #1a237e!important;border-radius:14px!important;
    padding:22px 18px!important;
    background:linear-gradient(145deg,#f8f9ff,#eef0ff)!important;
    box-shadow:0 6px 24px rgba(26,35,126,.12)!important}
div[data-testid="stForm"] div[data-testid="stTextInput"] input{
    text-align:center!important;font-size:26px!important;
    font-weight:800!important;color:#1a237e!important;
    height:54px!important;border:1.5px solid #b0bec5!important;
    border-radius:6px!important;padding:0!important;
    background:#fff!important;
    box-shadow:inset 0 2px 4px rgba(0,0,0,.06)!important;
    transition:all .15s ease!important}
div[data-testid="stForm"] div[data-testid="stTextInput"] input:hover{
    background:#fffde7!important;border-color:#ffc107!important}
div[data-testid="stForm"] div[data-testid="stTextInput"] input:focus{
    border-color:#1565c0!important;
    box-shadow:0 0 0 3px rgba(21,101,192,.3)!important;
    background:#e3f2fd!important}
div[data-testid="stForm"] div[data-testid="stTextInput"] input::placeholder{
    color:#d0d0d0!important;font-size:20px!important}
.block-divider{border:none;height:4px;
    background:linear-gradient(90deg,transparent,#1a237e,transparent);
    margin:5px 0 7px 0;border-radius:4px}
</style>""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center;padding:10px 0 5px 0">
    <h1 style="color:#1a237e;margin:0">🧩 SudokuSense AI</h1>
    <p style="color:#666;font-size:15px;margin-top:6px">
        يستخرج الأرقام ← يتحقق ← يُصحّح تلقائياً ← يحل اللغز
    </p>
</div>""", unsafe_allow_html=True)

# ─── Session State ───
defaults = {
    'grid': None,
    'solved': None,
    'original': None,
    'confidence': None,
    'alternatives': None,
    'uncertain': set(),
    'corrections': [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

model = load_ai_model()

# ╔═══════════════════════════════════════════╗
# ║ أقسام الإدخال ║
# ╚═══════════════════════════════════════════╝
tab_img, tab_manual = st.tabs(["📷 استخراج من صورة", "✏️ إدخال يدوي"])

with tab_img:
    uploaded = st.file_uploader("ارفع صورة سودوكو", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded:
        pil_img = Image.open(uploaded).convert('RGB')
        st.image(pil_img, caption="الصورة المرفوعة", width=320)
        if model is None:
            st.warning("⚠️ ملف النموذج غير موجود")
        elif main_processing is None:
            st.warning("⚠️ وحدة processing غير متوفرة")
        else:
            # شريط الثقة
            conf_thresh = st.slider(
                "🎚️ حد الثقة (كلما زاد ← تصحيح أكثر صرامة)",
                0.50, 0.95, 0.75, 0.05,
                help="الخلايا بثقة أقل من هذا الحد تُعلَّم كمشكوك فيها"
            )
            if st.button("🔍 استخراج وتصحيح تلقائي", type="primary", use_container_width=True, key="btn_extract"):
                cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                with st.status("جاري المعالجة …", expanded=True) as status:
                    # 1️⃣ معالجة الصورة
                    st.write("🔍 معالجة الصورة …")
                    binary = process_image_binary(cv_img)
                    contour = find_grid_contour(binary)
                    if contour is None:
                        st.error("❌ لم يُعثر على شبكة سودوكو")
                        st.stop()

                    # 2️⃣ قص الخانات
                    st.write("✂️ تقسيم الشبكة …")
                    ok, imgs, _ = main_processing(contour, binary)
                    if not ok:
                        st.error("❌ فشل تقسيم الشبكة")
                        st.stop()

                    # 3️⃣ قراءة الأرقام مع الثقة
                    st.write("🤖 قراءة الأرقام بالذكاء الاصطناعي …")
                    grid, conf, alts, uncertain = recognize_with_confidence(imgs, model, conf_thresh)

                    # 4️⃣ التحقق والتصحيح التلقائي
                    conflicts_before = find_conflicts(grid)
                    st.write(f"🔎 تعارضات مكتشفة: **{len(conflicts_before)}**")

                    if conflicts_before:
                        st.write("🔧 تصحيح تلقائي بقواعد السودوكو …")
                        grid, corrections = auto_correct(grid, conf, alts)
                        conflicts_after = find_conflicts(grid)
                        st.write(
                            f"✅ تصحيحات: **{len(corrections)}** — "
                            f"تعارضات متبقية: **{len(conflicts_after)}**"
                        )
                    else:
                        corrections = []

                    # حفظ في session
                    st.session_state.grid = grid
                    st.session_state.original = copy.deepcopy(grid)
                    st.session_state.confidence = conf
                    st.session_state.alternatives = alts
                    st.session_state.uncertain = uncertain
                    st.session_state.corrections = corrections
                    st.session_state.solved = None

                    status.update(label="✅ اكتمل!", state="complete")

                # عرض التصحيحات
                if corrections:
                    st.info("🔧 التصحيحات التلقائية:")
                    for fix in corrections:
                        st.write(
                            f" 📍 الصف {fix['row']+1} "
                            f"العمود {fix['col']+1}: "
                            f"**{fix['old']}** → **{fix['new']}** "
                            f"(ثقة: {fix['old_conf']:.0%} → "
                            f"{fix['new_conf']:.0%})"
                        )
                safe_rerun()

with tab_manual:
    ca, cb = st.columns(2)
    with ca:
        if st.button("📝 شبكة فارغة", use_container_width=True, key="btn_empty"):
            st.session_state.grid = [[0] * 9 for _ in range(9)]
            st.session_state.original = [[0] * 9 for _ in range(9)]
            st.session_state.solved = None
            st.session_state.uncertain = set()
            st.session_state.corrections = []
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
            st.session_state.uncertain = set()
            st.session_state.corrections = []
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
                    st.error("❌ يجب 9 أسطر")
                else:
                    g = []
                    for line in lines:
                        ds = [int(ch) for ch in line.replace(' ', '').replace(',', '') if ch.isdigit()]
                        if len(ds) != 9:
                            st.error(f"❌ «{line}» ليس 9 أرقام")
                            break
                        g.append(ds)
                    else:
                        st.session_state.grid = g
                        st.session_state.original = copy.deepcopy(g)
                        st.session_state.solved = None
                        st.session_state.uncertain = set()
                        safe_rerun()
            except Exception as e:
                st.error(f"❌ {e}")

# ╔═══════════════════════════════════════════╗
# ║ عرض الشبكة + محرر التعديل ║
# ╚═══════════════════════════════════════════╝
if st.session_state.grid is not None:
    st.markdown("---")

    # ── معاينة HTML ──
    corrected_set = {(f['row'], f['col']) for f in st.session_state.corrections}
    preview = render_sudoku_html(
        st.session_state.grid,
        title="الشبكة الحالية",
        uncertain_cells=st.session_state.uncertain,
        corrected_cells=corrected_set,
        confidence=st.session_state.confidence
    )
    components.html(preview, height=590, scrolling=False)

    # ── نموذج التعديل ──
    st.markdown(
        '<p style="text-align:center;color:#5c6bc0;font-weight:600">'
        '⬇️ عدّل الأرقام (اتركها فارغة = 0) ثم اضغط <b>🚀 حل</b></p>',
        unsafe_allow_html=True
    )

    edited = [[0] * 9 for _ in range(9)]
    with st.form("sudoku_form", clear_on_submit=False):
        for block_row in range(3):
            for local_row in range(3):
                row = block_row * 3 + local_row
                widths = [1, 1, 1, 0.12, 1, 1, 1, 0.12, 1, 1, 1]
                cols = st.columns(widths, gap="small")
                mapping = [0, 1, 2, 4, 5, 6, 8, 9, 10]
                for c in range(9):
                    with cols[mapping[c]]:
                        v = st.session_state.grid[row][c]
                        disp = str(v) if v != 0 else ""
                        res = st.text_input(
                            "​",
                            value=disp,
                            key=f"sc_{row}_{c}",
                            max_chars=1,
                            label_visibility="collapsed",
                            placeholder="·"
                        )
                        if res and res.strip().isdigit():
                            d = int(res.strip())
                            edited[row][c] = d if 1 <= d <= 9 else 0
            if block_row < 2:
                st.markdown('<hr class="block-divider">', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        with b1:
            solve_btn = st.form_submit_button("🚀 حل السودوكو", type="primary", use_container_width=True)
        with b2:
            reset_btn = st.form_submit_button("🔄 إعادة تعيين", use_container_width=True)
        with b3:
            clear_btn = st.form_submit_button("🗑️ مسح الكل", use_container_width=True)

        if solve_btn:
            ok = all(0 <= edited[r][c] <= 9 for r in range(9) for c in range(9))
            if not ok:
                st.error("❌ قيم غير صالحة!")
            else:
                st.session_state.grid = edited
                st.session_state.original = copy.deepcopy(edited)
                with st.spinner("⚙️ جاري الحل …"):
                    result = solve_puzzle(edited)
                    if result:
                        st.session_state.solved = result
                        safe_rerun()
                    else:
                        st.error("❌ الشبكة غير قابلة للحل!")

        if reset_btn:
            st.session_state.grid = copy.deepcopy(
                st.session_state.original or [[0] * 9 for _ in range(9)]
            )
            st.session_state.solved = None
            safe_rerun()

        if clear_btn:
            for k in defaults:
                st.session_state[k] = defaults[k]
            safe_rerun()

# ╔═══════════════════════════════════════════╗
# ║ عرض النتيجة النهائية ║
# ╚═══════════════════════════════════════════╝
if st.session_state.solved is not None:
    st.markdown("---")
    st.subheader("✅ الحل النهائي")

    solved_html = render_sudoku_html(
        st.session_state.solved,
        st.session_state.original,
        title="🎉 تم الحل بنجاح!"
    )
    components.html(solved_html, height=620, scrolling=False)

    result_img = draw_sudoku_image(st.session_state.solved, st.session_state.original)
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
