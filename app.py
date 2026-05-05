import streamlit as st
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, detrend, welch
import plotly.graph_objects as go
import tempfile
import os

# ==========================================
# ⚙️ CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(page_title="AEGIS-PROVENANCE | Universal Reality Auditor", layout="wide", page_icon="🔱")
st.markdown("""
    <style>
    .big-font {font-size:30px !important; font-weight: bold; color: #1E90FF;}
    .report-box {padding: 20px; border-radius: 10px; background-color: #1E1E1E; border: 1px solid #333;}
    .layer-card {background-color: #2b2b2b; padding: 15px; border-radius: 8px; margin-bottom: 10px;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 🧠 MATHEMATICAL 5-LAYER ENGINE (FINAL BUILD)
# ==========================================

class AegisEngine:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.prev_face = None 
        
    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def asymmetric_map(self, value, center, scale, invert=False, min_val=0.10, max_val=0.85):
        """Asymmetric mapping: Fails aggressively (0.10), passes cautiously (0.85)."""
        direction = -1 if invert else 1
        prob = 1 / (1 + np.exp(-scale * direction * (value - center)))
        return min_val + prob * (max_val - min_val)

    # LAYER 1: rPPG (Biological Boundaries)
    def extract_rppg(self, green_signal, face_motions, fps):
        if len(green_signal) < fps * 3:
            return 0.50, [], 0.0, 0.0
            
        avg_motion_percent = np.mean(face_motions) if face_motions else 0.0
        if avg_motion_percent > 15.0: 
            return 0.50, [], 0.0, avg_motion_percent # Abstain on high motion
            
        detrended = detrend(green_signal)
        b, a = self.butter_bandpass(0.75, 4.0, fps, order=3)
        filtered_signal = filtfilt(b, a, detrended)
        
        f, pxx = welch(filtered_signal, fs=fps, nperseg=len(filtered_signal))
        pulse_band = (f >= 0.75) & (f <= 4.0)
        snr = np.sum(pxx[pulse_band]) / (np.sum(pxx) + 1e-8)
        
        # Hard Biological Limits
        if snr > 0.60:
            confidence = 0.10 # Impossibly perfect. AI simulation.
        elif snr < 0.015:
            confidence = 0.40 # Likely fake, or destroyed by compression
        else:
            confidence = 0.75 # Natural human pulse variance
            
        return confidence, filtered_signal, snr, avg_motion_percent

    # LAYER 2: Hardware PRNU (Calibrated for Dataset Compression)
    def extract_hardware_noise(self, frame):
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (512, 512))
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        noise_residual = cv2.absdiff(gray, denoised)
        
        var_noise = np.var(noise_residual) + 1e-6
        var_image = np.var(gray) + 1e-6
        nsr = var_noise / var_image
        
        # Center dropped to 0.0008 so compressed real videos pass safely
        confidence = self.asymmetric_map(nsr, center=0.0008, scale=2000.0, max_val=0.75)
        return confidence, nsr

    # LAYER 3: Spatial Entropy
    def calculate_entropy(self, frame):
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (512, 512))
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        hist = hist / (hist.sum() + 1e-7)
        logs = np.log2(hist + 1e-7)
        spatial_entropy = -np.sum(hist * logs)
        
        confidence = self.asymmetric_map(spatial_entropy, center=7.0, scale=4.0, max_val=0.80)
        return confidence, spatial_entropy

    # LAYER 4: Spectral Decay Ratio
    def detect_generator_artifacts(self, frame):
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (512, 512))
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)
        
        h, w = 512, 512
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        r = np.sqrt(x*x + y*y)
        
        mask_lf = r < 50
        mask_hf = r > 150 
        
        energy_lf = np.mean(magnitude_spectrum[mask_lf]) + 1e-8
        energy_hf = np.mean(magnitude_spectrum[mask_hf])
        hf_lf_ratio = energy_hf / energy_lf
        
        confidence = self.asymmetric_map(hf_lf_ratio, center=0.62, scale=40.0, invert=True, max_val=0.85)
        return confidence, hf_lf_ratio

    # LAYER 5: Metadata Veto (AI ONLY)
    def analyze_metadata(self, video_path):
        try:
            with open(video_path, 'rb') as f:
                header = f.read(8192).decode('utf-8', errors='ignore')
                f.seek(-8192, os.SEEK_END)
                footer = f.read().decode('utf-8', errors='ignore')
                metadata = header + footer
            
            # STRICTLY AI MODELS ONLY. No 'Lavf', no 'ffmpeg'.
            suspicious_tags = ['Runway', 'Midjourney', 'HeyGen', 'Sora', 'Veo', 'SynthID', 'Luma', 'Pika', 'Kling', 'Krea']
            found_tags = [tag for tag in suspicious_tags if tag.lower() in metadata.lower()]
            
            if found_tags:
                return 0.01, f"Manipulated ({', '.join(set(found_tags))})" 
            return 0.50, "Metadata Clean / Stripped"
        except Exception:
            return 0.50, "Unreadable"

    # FUSION ENGINE
    def dempster_shafer_fusion(self, masses_list, meta_conf):
        masses = [np.array([m, 1-m]) for m in masses_list]
        combined = masses[0]
        
        for i in range(1, len(masses)):
            m_next = masses[i]
            K = (combined[0] * m_next[1]) + (combined[1] * m_next[0])
            if K >= 1.0: K = 0.999 
            
            new_real = (combined[0] * m_next[0]) / (1 - K)
            new_fake = (combined[1] * m_next[1]) / (1 - K)
            combined = np.array([new_real, new_fake])
            
        final_score = combined[0]
        if meta_conf <= 0.05:
            final_score = 0.01 # Hard Veto
            
        return final_score

# ==========================================
# 🖥️ FRONTEND & EXECUTION
# ==========================================

st.markdown('<p class="big-font">🔱 AEGIS-PROVENANCE: Universal Reality Auditor</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Target Video Evidence (MP4)", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    engine = AegisEngine()
    meta_conf, meta_status = engine.analyze_metadata(video_path)

    col1, col2, col3 = st.columns([1.2, 1, 1])
    
    with col1:
        st.subheader("Layer 1: Live Feed & rPPG")
        video_placeholder = st.empty()
        rppg_chart = st.empty()
        
    with col2:
        st.subheader("Layers 2 & 3: Physics & Entropy")
        metrics_placeholder = st.empty()
        
    with col3:
        st.subheader("Layers 4 & 5: Spectral & Provenance")
        gen_meta_placeholder = st.empty()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    green_signals, face_motions = [], []
    nsr_ratios, entropies, hflf_ratios = [], [], []
    
    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = engine.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            
            if engine.prev_face is not None:
                px, py, pw, ph = engine.prev_face
                motion_percent = ((abs(x - px) + abs(y - py)) / float(pw)) * 100.0
                face_motions.append(motion_percent)
                
                # EMA Stabilization
                x = int(0.90 * px + 0.10 * x)
                y = int(0.90 * py + 0.10 * y)
                w = int(0.90 * pw + 0.10 * w)
                h = int(0.90 * ph + 0.10 * h)
            else:
                face_motions.append(0)
                
            engine.prev_face = (x, y, w, h)
            
            # Extract Forehead for Layer 1
            x_min, x_max = x + int(w * 0.3), x + int(w * 0.7)
            y_min, y_max = y + int(h * 0.05), y + int(h * 0.2)
            
            if x_max > x_min and y_max > y_min:
                forehead_roi = frame[y_min:y_max, x_min:x_max]
                green_signals.append(np.mean(forehead_roi[:, :, 1]))
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
        else:
            green_signals.append(green_signals[-1] if green_signals else 0)
            face_motions.append(face_motions[-1] if face_motions else 0)

        # Physics Extraction (FULL FRAME)
        if frame_count % 5 == 0:
            _, nsr = engine.extract_hardware_noise(frame)
            _, ent = engine.calculate_entropy(frame)
            _, hflf = engine.detect_generator_artifacts(frame)
            
            nsr_ratios.append(nsr)
            entropies.append(ent)
            hflf_ratios.append(hflf)

        # UI Updates
        if frame_count % 10 == 0:
            video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width=450)
            
            if len(green_signals) > fps:
                _, filtered, _, _ = engine.extract_rppg(green_signals, face_motions, fps)
                if len(filtered) > 0:
                    fig = go.Figure(data=go.Scatter(y=filtered[-100:], mode='lines', line=dict(color='lime')))
                    fig.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    rppg_chart.plotly_chart(fig, use_container_width=True)

            if nsr_ratios:
                metrics_placeholder.markdown(f"""
                <div class="layer-card"><b>L2: Global Noise Ratio</b><br>{nsr_ratios[-1]:.5f}</div>
                <div class="layer-card"><b>L3: Global Spatial Entropy</b><br>{entropies[-1]:.2f} bits</div>
                """, unsafe_allow_html=True)
                
                gen_meta_placeholder.markdown(f"""
                <div class="layer-card"><b>L4: Global Spectral Decay</b><br>{hflf_ratios[-1]:.4f} (HF/LF)</div>
                <div class="layer-card"><b>L5: Causal Provenance</b><br>{meta_status}</div>
                """, unsafe_allow_html=True)
                
            progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    try:
        os.remove(video_path)
    except:
        pass
    
    # ==========================================
    # ⚖️ FINAL 5-LAYER AUDIT REPORT
    # ==========================================
    st.markdown("---")
    st.header("⚖️ Legal-Grade Forensic Report")
    
    rppg_conf, _, avg_snr, avg_motion_percent = engine.extract_rppg(green_signals, face_motions, fps)
    
    avg_nsr = np.mean(nsr_ratios) if nsr_ratios else 0.0020
    noise_conf = engine.asymmetric_map(avg_nsr, center=0.0008, scale=2000.0, max_val=0.75)
    
    avg_ent = np.mean(entropies) if entropies else 7.15
    entropy_conf = engine.asymmetric_map(avg_ent, center=7.0, scale=4.0, max_val=0.80)
    
    avg_hflf = np.mean(hflf_ratios) if hflf_ratios else 0.5
    gen_conf = engine.asymmetric_map(avg_hflf, center=0.62, scale=40.0, invert=True, max_val=0.85)
    
    final_belief_real = engine.dempster_shafer_fusion(
        [rppg_conf, noise_conf, entropy_conf, gen_conf, meta_conf], meta_conf
    )
    
    result_color = "#00FF00" if final_belief_real > 0.5 else "#FF0000"
    
    if meta_conf <= 0.05:
        result_text = "SYNTHETIC MEDIA DETECTED (METADATA VETO)"
    else:
        result_text = "AUTHENTICITY VERIFIED" if final_belief_real > 0.5 else "SYNTHETIC MEDIA DETECTED"
    
    motion_status = f"Abstained (Motion {avg_motion_percent:.1f}%)" if avg_motion_percent > 15.0 else f"SNR {avg_snr:.2f}"
    
    st.markdown(f"""
    <div class="report-box" style="border-left: 10px solid {result_color};">
        <h2 style="color: {result_color};">{result_text}</h2>
        <p><b>Dempster-Shafer 5-Layer Combined Belief:</b> {final_belief_real*100:.2f}% Probability of Authenticity</p>
        <hr>
        <h4>EVIDENCE VECTORS (Global Scale-Invariant Math):</h4>
        <ul>
            <li><b>L1 Biological (rPPG):</b> {motion_status} (Conf: {rppg_conf:.2f})</li>
            <li><b>L2 Hardware (PRNU):</b> Global Noise-to-Signal Ratio {avg_nsr:.5f} (Conf: {noise_conf:.2f})</li>
            <li><b>L3 Info Theory:</b> Global Spatial Entropy {avg_ent:.2f} (Conf: {entropy_conf:.2f})</li>
            <li><b>L4 Gen Fingerprint:</b> Global Spectral Decay Ratio (HF/LF) {avg_hflf:.4f} (Conf: {gen_conf:.2f})</li>
            <li><b>L5 Provenance:</b> {meta_status} (Conf: {meta_conf:.2f})</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
