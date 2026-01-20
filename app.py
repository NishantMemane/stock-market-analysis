import streamlit as st
from streamlit_cropper import st_cropper
from PIL import Image
import cv2
import numpy as np
import tempfile
import os
import json

# ==========================================
# IMPORT MODULES
# ==========================================
import new_markings as markings_engine
import support_resistance as sr_engine
import dow_theory as dow_engine
import trend_detect as trend_engine

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Stock Market Trend Analysis",
    layout="wide",
    page_icon="🕸️",
    initial_sidebar_state="expanded"
)

# ==========================================
# PREMIUM CSS STYLING
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* GLASSMORPHISM BACKGROUNDS */
    .stApp {
        background-color: #0e1117;
        background-image: radial-gradient(at 0% 0%, rgba(16, 23, 42, 1) 0px, transparent 50%), 
                          radial-gradient(at 100% 0%, rgba(16, 23, 42, 1) 0px, transparent 50%);
    }

    /* SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: rgba(22, 27, 34, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* CARD STYLING */
    .css-1r6slb0, .css-keje6w, .stImage {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        backdrop-filter: blur(10px);
    }

    /* PRIMARY BUTTONS */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        width: 100%;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.39);
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.23);
        border: none;
        color: white;
    }

    /* TAB STYLING */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: #94a3b8;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.05);
        color: #8b5cf6;
        border-bottom: 2px solid #8b5cf6;
    }
    
    /* CUSTOM HEADERS */
    h1, h2, h3 {
        color: #f8fafc;
        font-weight: 700;
        letter-spacing: -0.025em;
    }
    
    .gradient-text {
        background: linear-gradient(to right, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
</style>
""", unsafe_allow_html=True)

# ==========================================
# HELPERS
# ==========================================
def save_temp_file(uploaded_file, suffix):
    if uploaded_file is None: return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name

def pil_to_cv2_path(pil_image):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        pil_image.save(tmp.name)
        return tmp.name

def render_verdict_card(verdict_text):
    color = "#94a3b8"
    icon = "⚖️"
    bg = "rgba(255,255,255,0.05)"
    
    if "Bullish" in verdict_text or "Up" in verdict_text:
        color = "#34d399" # Emerald 400
        icon = "🚀"
        bg = "rgba(16, 185, 129, 0.1)"
    elif "Bearish" in verdict_text or "Down" in verdict_text:
        color = "#f87171" # Red 400
        icon = "🐻"
        bg = "rgba(239, 68, 68, 0.1)"
        
    st.markdown(f"""
    <div style="background: {bg}; border: 1px solid {color}; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 20px;">
        <div style="font-size: 3rem; margin-bottom: 10px;">{icon}</div>
        <h3 style="margin: 0; color: {color}; font-size: 1.5rem;">{verdict_text}</h3>
        <p style="margin-top: 5px; opacity: 0.8; font-size: 0.9rem;">AI Market Structure Analysis</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("## 🧠 Trend Engine")
    st.caption("Quantitative Technical Analysis Engine")
    
    st.markdown("---")
    
    st.subheader("1. Analysis Module")
    tool_mode = st.radio(
        "Select Strategy",
        [
            "Generate Markings (HH/LL)",
            "Dow Theory Analysis",
            "Trend Detection",
            "Support & Resistance"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.subheader("2. Parameters")
    scale = st.slider("Pivot Sensitivity", 1, 10, 5, help="Low = Micro Structure, High = Macro Structure")
    
    st.markdown("---")
    st.info(f"**Mode Selected:**\n{tool_mode}")

# ==========================================
# MAIN LAYOUT
# ==========================================

# HEADER
st.markdown('<h1 class="gradient-text">Stock Market Trend Analysis</h1>', unsafe_allow_html=True)
st.markdown("Upload chart data, verify region of interest, and execute algorithmic structure analysis.")

# FILE UPLOAD
uploaded_file = st.file_uploader("Drop Chart Image Here", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    # -------------------------------------------------------------
    # WORKSPACE
    # -------------------------------------------------------------
    st.markdown("### 👁️ Focus Area Selection")
    
    col_input, col_controls = st.columns([3, 1])
    
    original_pil = Image.open(uploaded_file)
    
    with col_input:
        # CROPPER
        cropped_pil = st_cropper(
            original_pil,
            realtime_update=True,
            box_color='#6366f1',
            aspect_ratio=None,
            should_resize_image=True
        )
        
    with col_controls:
        st.markdown("**Configuration**")
        st.caption("Adjust the selection box to cover only the price action candles.")
        
        custom_json_path = None
        if tool_mode != "Generate Markings (HH/LL)":
            with st.expander("📂 Import Data"):
                json_file = st.file_uploader("Previous .json", type=['json'])
                if json_file:
                    custom_json_path = save_temp_file(json_file, ".json")
                    st.success("Linked!")
        
        st.write("")
        st.write("")
        run_btn = st.button("RUN ANALYSIS", type="primary")

    # -------------------------------------------------------------
    # EXECUTION & RESULTS
    # -------------------------------------------------------------
    if run_btn:
        st.divider()
        
        prog_bar = st.progress(0, text="Initializing Neural Engine...")
        
        try:
            # 1. SETUP
            crop_path = pil_to_cv2_path(cropped_pil)
            temp_dir = tempfile.gettempdir()
            w, h = cropped_pil.size
            
            final_img, final_json, final_verdict = None, None, None
            
            # 2. PIPELINE
            # --- BASE MARKINGS ---
            if tool_mode == "Generate Markings (HH/LL)":
                prog_bar.progress(60, text="Identifying Swing Points...")
                _, final_json, final_img = markings_engine.run_markings_logic(
                    img_path=crop_path, scale=scale, output_prefix="mark", 
                    manual_crop_rect=(0,0,w,h), output_dir=temp_dir
                )
                final_verdict = "Structure Mapped"
                
            else:
                # Need Base JSON?
                if custom_json_path:
                    base_json = custom_json_path
                    prog_bar.progress(20, text="Loading external data...")
                else:
                    prog_bar.progress(30, text="Computing Base Structure...")
                    _, base_json, _ = markings_engine.run_markings_logic(
                        img_path=crop_path, scale=scale, output_prefix="base", 
                        manual_crop_rect=(0,0,w,h), output_dir=temp_dir
                    )

                # SPECIFIC MODULES
                prog_bar.progress(70, text=f"Executing {tool_mode} Logic...")
                
                if tool_mode == "Dow Theory Analysis":
                    analyzer = dow_engine.MarketStructureAnalyzer(crop_path, base_json)
                    analyzer.analyze_structure()
                    out_i = os.path.join(temp_dir, "dow_res.png")
                    out_j = os.path.join(temp_dir, "dow_res.json")
                    analyzer.generate_outputs(out_i, out_j)
                    final_img, final_json = out_i, out_j
                    with open(out_j, 'r') as f: final_verdict = json.load(f).get("verdict", "N/A")

                elif tool_mode == "Trend Detection":
                    analyzer = trend_engine.MarketStructureAnalyzer(crop_path, base_json)
                    analyzer.segment_chart()
                    analyzer.detect_shift_zones()
                    out_i = os.path.join(temp_dir, "trend_res.png")
                    out_j = os.path.join(temp_dir, "trend_res.json")
                    analyzer.generate_outputs(out_i, out_j)
                    final_img, final_json = out_i, out_j
                    with open(out_j, 'r') as f: final_verdict = json.load(f).get("verdict", "N/A")

                elif tool_mode == "Support & Resistance":
                    final_img = sr_engine.run_support_resistance_logic(
                        img_path=crop_path, json_path=base_json, 
                        crop_rect=(0,0,w,h), output_prefix="sr_res", 
                        draw_labels=False, output_dir=temp_dir
                    )
                    final_json = base_json
                    final_verdict = "Zones Detected"

            prog_bar.progress(100, text="Complete")
            prog_bar.empty()
            
            # 3. DISPLAY
            if final_img and os.path.exists(final_img):
                
                # Render Tabs
                tab1, tab2, tab3 = st.tabs(["🖼️ Visual Intelligence", "📊 Data Interpretation", "📥 Export"])
                
                with tab1:
                    # Verdict Card
                    render_verdict_card(final_verdict)
                    # Main Image
                    bgr = cv2.imread(final_img)
                    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), use_column_width=True, caption="AI Overlay Analysis")
                
                with tab2:
                    col_d1, col_d2 = st.columns(2)
                    with col_d1:
                         st.markdown("#### Market Context")
                         st.info(f"Analysis Type: **{tool_mode}**")
                         st.info(f"Pivot Sensitivity: **{scale}**")
                    with col_d2:
                        if final_json and os.path.exists(final_json):
                             with open(final_json, 'r') as f:
                                 data = json.load(f)
                             st.json(data, expanded=False)
                
                with tab3:
                    st.success("Analysis artifacts ready for download.")
                    with open(final_img, "rb") as f:
                        st.download_button("Download High-Res Map", f, "analysis_map.png", "image/png")
                    if final_json:
                        with open(final_json, "r") as f:
                            st.download_button("Download Vector Data (JSON)", json.dumps(json.load(f), indent=4), "vectors.json", "application/json")
                            
            else:
                st.error("Analysis engine failed to produce visual output.")

        except Exception as e:
            st.error(f"Critical Runtime Exception: {e}")
            import traceback
            st.expander("Debug Trace").code(traceback.format_exc())

else:
    # Empty State
    st.markdown("""
    <div style='text-align: center; padding: 50px; color: #64748b;'>
        <h2>waiting for input stream...</h2>
        <p>Drag and drop a financial chart file to initialize the system.</p>
    </div>
    """, unsafe_allow_html=True)