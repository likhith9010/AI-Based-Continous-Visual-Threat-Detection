import customtkinter as ctk
import cv2
from PIL import Image
import threading
import time
import requests
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
import base64
from vjepa_engine import VJEPAEngine



ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# YOLO v8 COCO class names (80 classes) for reference during LLM mapping
YOLO_CLASSES = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "sofa","pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

# Common synonym map: user words → correct YOLO class name
# This runs BEFORE the LLM to guarantee reliable mapping for common terms
SYNONYM_MAP = {
    # Phones
    "mobile": "cell phone", "phone": "cell phone", "smartphone": "cell phone",
    "iphone": "cell phone", "android": "cell phone", "cellphone": "cell phone",
    "handphone": "cell phone", "telephone": "cell phone",
    # Weapons (custom trained model uses 'weapon' class)
    "gun": "weapon", "pistol": "weapon", "firearm": "weapon",
    "rifle": "weapon", "revolver": "weapon", "sword": "weapon", "knife": "weapon",
    # Vehicles
    "bike": "bicycle", "cycle": "bicycle", "motorcycle": "motorbike",
    "plane": "aeroplane", "airplane": "aeroplane", "aircraft": "aeroplane",
    # People
    "human": "person", "man": "person", "woman": "person", "child": "person",
    "people": "person", "individual": "person",
    # Common objects
    "bag": "handbag", "purse": "handbag", "backpack": "backpack",
    "laptop": "laptop", "computer": "laptop", "notebook": "laptop",
    "tv": "tvmonitor", "television": "tvmonitor", "monitor": "tvmonitor",
    "couch": "sofa", "plant": "pottedplant",
    "table": "diningtable", "dining table": "diningtable",
}


class ThreatDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AI Threat Detection System")
        self.geometry("560x500")
        self.resizable(True, True)

        print("Loading YOLOv8 object detectors...")
        self.yolo_general = YOLO("yolov8n.pt")                                          # 80 classes (person, phone, etc.)
        self.yolo_weapon  = YOLO("runs/detect/runs/train/guns_model-4/weights/best.pt")  # custom weapon class
        self.yolo_model = self.yolo_general  # default; switched after threat mapping
        self.llm_model_name = "llama3:8b"

        self.running = False
        self.cap = None
        self.threat_target = ""      # raw user input
        self.yolo_class_target = ""  # LLM-mapped YOLO class name
        self.detection_count = 0
        self.current_context = []
        self.last_llm_time = 0
        self.video_file_path = ""
        self.analysis_log = []       # list of strings, capped at 100

        # Video control state
        self.paused = False
        self.user_seeking = False
        self.total_frames = 0
        self.video_fps = 30.0
        self.seek_var = tk.DoubleVar(value=0)
        self.cap_lock = threading.Lock()  # Prevent concurrent access to VideoCapture

        # Three-Stream State
        self.vjepa_engine = VJEPAEngine()
        self.threat_override = None            # Signal from YOLO to Vision stream
        self.latest_frame = None               # Shared frame for vision stream
        self.anomaly_info = {"score": 0.0, "label": "Normal"}
        self.vision_model_name = "gemma4:e2b"  # Stream 3 model

        # Widget refs (nulled before screen destroy so threads don't crash)
        self.video_label = None
        self.count_label = None
        self.status_label = None
        self.scene_textbox = None
        self.play_pause_btn = None
        self.time_label = None
        self.seek_slider = None

        self.show_setup_screen()

    # ── Thread-safe UI helpers ─────────────────────────────────────────────────
    def _safe_after(self, fn):
        try:
            self.after(0, fn)
        except Exception:
            pass

    def _set_status(self, text):
        def _u():
            try:
                if self.status_label and self.status_label.winfo_exists():
                    self.status_label.configure(text=text)
            except Exception:
                pass
        self._safe_after(_u)

    def _set_threat_status(self, is_threat):
        def _u():
            try:
                if self.count_label and self.count_label.winfo_exists():
                    if is_threat:
                        self.count_label.configure(text="🚨 THREAT!!", text_color="#ff4d4d")
                        self.count_card.configure(fg_color="#4d0000")
                    else:
                        self.count_label.configure(text="✅ Scene Safe", text_color="#00ff88")
                        self.count_card.configure(fg_color="#002d14")
            except Exception:
                pass
        self._safe_after(_u)

    def _set_video_frame(self, ctk_img):
        def _u():
            try:
                if self.video_label and self.video_label.winfo_exists():
                    self.video_label.configure(image=ctk_img, text="")
                    self.video_label.image = ctk_img
            except Exception:
                pass
        self._safe_after(_u)

    def _append_analysis(self, text, is_threat=False):
        """Append a timestamped entry; cap log at 100, prune oldest 25 when full."""
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}]\n{text}\n"
        self.analysis_log.append((entry, is_threat))
        if len(self.analysis_log) > 100:
            self.analysis_log = self.analysis_log[25:]

        def _u():
            try:
                if self.scene_textbox and self.scene_textbox.winfo_exists():
                    self.scene_textbox.configure(state="normal")
                    self.scene_textbox.delete("0.0", "end")
                    # Configure threat tag (No 'font' allowed in CTk scaling)
                    self.scene_textbox.tag_config("threat", foreground="#ff3333")
                    # Newest first
                    for e, thr in reversed(self.analysis_log):
                        if thr:
                            self.scene_textbox.insert("end", e + "\n─────────────────────\n", "threat")
                        else:
                            self.scene_textbox.insert("end", e + "\n─────────────────────\n")
                    self.scene_textbox.configure(state="disabled")
                    self.scene_textbox.see("1.0")
            except Exception:
                pass
        self._safe_after(_u)

    # ── Setup Screen ───────────────────────────────────────────────────────────
    def show_setup_screen(self):
        self.video_label = None
        self.count_label = None
        self.status_label = None
        self.scene_textbox = None
        self.play_pause_btn = None
        self.time_label = None
        self.seek_slider = None
        self.paused = False
        self.user_seeking = False

        for w in self.winfo_children():
            w.destroy()
        self.geometry("560x500")

        container = ctk.CTkFrame(self, corner_radius=16)
        container.pack(expand=True, fill="both", padx=30, pady=30)

        ctk.CTkLabel(container, text="🛡️  AI Threat Detection",
                     font=("Arial", 24, "bold")).pack(pady=(24, 4))
        ctk.CTkLabel(container, text="Configure your session below",
                     font=("Arial", 13), text_color="gray").pack(pady=(0, 20))

        # ── 1. Input Source ────────────────────────────────────────────────────
        ctk.CTkLabel(container, text="1️⃣  Input Source",
                     font=("Arial", 15, "bold")).pack(anchor="w", padx=24)

        self.source_var = ctk.StringVar(value="Camera")
        src_row = ctk.CTkFrame(container, fg_color="transparent")
        src_row.pack(fill="x", padx=24, pady=(6, 0))

        ctk.CTkRadioButton(src_row, text="📷  Live Camera", variable=self.source_var,
                           value="Camera", command=self._on_source_change).pack(side="left", padx=(0, 16))
        ctk.CTkRadioButton(src_row, text="🎞  Video File", variable=self.source_var,
                           value="Video", command=self._on_source_change).pack(side="left")

        # Video file section (shown only when Video selected)
        self.video_section = ctk.CTkFrame(container, fg_color="transparent")
        # Don't pack yet

        vid_row = ctk.CTkFrame(self.video_section, fg_color="transparent")
        vid_row.pack(fill="x", padx=0, pady=(4, 0))

        self.file_path_label = ctk.CTkLabel(
            vid_row, text="No file selected",
            text_color="gray", font=("Arial", 12),
            wraplength=340, justify="left"
        )
        self.file_path_label.pack(side="left", fill="x", expand=True)

        ctk.CTkButton(
            vid_row, text="📁  Browse", width=110,
            command=self._browse_file
        ).pack(side="right", padx=(8, 0))

        # ── 2. Threat to Detect ────────────────────────────────────────────────
        ctk.CTkLabel(container, text="2️⃣  Threat to Detect",
                     font=("Arial", 15, "bold")).pack(anchor="w", padx=24, pady=(20, 0))
        ctk.CTkLabel(container,
                     text='Natural language OK — e.g. "gun", "someone with a phone", "suspicious bag"',
                     font=("Arial", 11), text_color="gray", wraplength=460, justify="left"
                     ).pack(anchor="w", padx=24)

        self.threat_entry = ctk.CTkEntry(
            container, placeholder_text="Describe the threat...",
            height=40, font=("Arial", 14)
        )
        self.threat_entry.pack(fill="x", padx=24, pady=(6, 0))

        self.error_label = ctk.CTkLabel(container, text="", text_color="#ff4d4d", font=("Arial", 12))
        self.error_label.pack(pady=(4, 0))

        ctk.CTkButton(
            container, text="▶  Start Detection",
            height=44, font=("Arial", 15, "bold"),
            fg_color="#d62828", hover_color="#a00000",
            command=self._start_detection
        ).pack(fill="x", padx=24, pady=18)

    def _on_source_change(self):
        if self.source_var.get() == "Video":
            self.video_section.pack(fill="x", padx=24, pady=(4, 0))
        else:
            self.video_section.pack_forget()
            self.video_file_path = ""
            self.file_path_label.configure(text="No file selected", text_color="gray")

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All files", "*.*")]
        )
        if path:
            self.video_file_path = path
            # Show just the filename, not full path
            filename = path.split("/")[-1].split("\\")[-1]
            self.file_path_label.configure(text=f"✅  {filename}", text_color="#90ee90")

    def _start_detection(self):
        self.threat_target = self.threat_entry.get().strip()
        source_type = self.source_var.get()

        if not self.threat_target:
            self.error_label.configure(text="⚠ Please describe a threat to detect.")
            return

        if source_type == "Video":
            if not self.video_file_path:
                self.error_label.configure(text="⚠ Please select a video file.")
                return
            self.cap = cv2.VideoCapture(self.video_file_path)
        else:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            self.error_label.configure(text="⚠ Could not open camera / video file.")
            return

        self.detection_count = 0
        self.current_context = []
        self.last_llm_time = 0
        self.analysis_log = []
        self.paused = False
        self.user_seeking = False
        self.seek_var.set(0)
        self.running = True
        self.yolo_class_target = ""  # will be resolved by LLM

        # Get video metadata for seek bar
        if source_type == "Video":
            self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        else:
            self.total_frames = 0
            self.video_fps = 30.0

        self.show_detection_screen()
        threading.Thread(target=self.video_loop, daemon=True).start()
        threading.Thread(target=self.llm_loop, daemon=True).start()

    # ── Detection Screen ───────────────────────────────────────────────────────
    def show_detection_screen(self):
        for w in self.winfo_children():
            w.destroy()
        self.geometry("1320x820")

        # Top bar
        topbar = ctk.CTkFrame(self, height=52, corner_radius=0)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        ctk.CTkLabel(topbar, text="🛡️  AI Threat Detection  —  LIVE",
                     font=("Arial", 16, "bold")).pack(side="left", padx=16)
        self.target_badge = ctk.CTkLabel(
            topbar, text=f"🎯 Resolving threat target...",
            font=("Arial", 13), text_color="#ffca3a"
        )
        self.target_badge.pack(side="left", padx=12)
        ctk.CTkButton(topbar, text="⬅  New Session", width=130,
                      fg_color="transparent", border_width=1,
                      command=self._stop_and_reset).pack(side="right", padx=16, pady=8)

        # Body
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(expand=True, fill="both", padx=10, pady=10)
        body.grid_columnconfigure(0, weight=4)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        # Left: video
        vc = ctk.CTkFrame(body, corner_radius=12)
        vc.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.video_label = ctk.CTkLabel(vc, text="Starting feed...", font=("Arial", 14))
        self.video_label.pack(expand=True, fill="both")

        # Video controls (only shown in Video File mode)
        if self.source_var.get() == "Video":
            ctrl_bar = ctk.CTkFrame(vc, fg_color="#1a1a2e", corner_radius=0, height=90)
            ctrl_bar.pack(fill="x", side="bottom")
            ctrl_bar.pack_propagate(False)

            # Seek slider
            self.seek_slider = ctk.CTkSlider(
                ctrl_bar, from_=0, to=max(self.total_frames - 1, 1),
                variable=self.seek_var, number_of_steps=int(self.total_frames),
                button_color="#d62828", progress_color="#d62828"
            )
            self.seek_slider.pack(fill="x", padx=16, pady=(10, 2))
            self.seek_slider.bind("<ButtonPress-1>",   lambda e: self._on_seek_press())
            self.seek_slider.bind("<ButtonRelease-1>", lambda e: self._on_seek_release())

            # Buttons row
            btn_row = ctk.CTkFrame(ctrl_bar, fg_color="transparent")
            btn_row.pack(pady=(2, 6))

            ctk.CTkButton(btn_row, text="⏮", width=44, height=32,
                          fg_color="#2a2a3e", hover_color="#3a3a5e",
                          command=self._restart_video).pack(side="left", padx=4)
            ctk.CTkButton(btn_row, text="⏪ 10s", width=64, height=32,
                          fg_color="#2a2a3e", hover_color="#3a3a5e",
                          command=lambda: self._seek_relative(-10)).pack(side="left", padx=4)
            self.play_pause_btn = ctk.CTkButton(
                btn_row, text="⏸ Pause", width=90, height=32,
                fg_color="#d62828", hover_color="#a00000",
                command=self._toggle_pause)
            self.play_pause_btn.pack(side="left", padx=4)
            ctk.CTkButton(btn_row, text="10s ⏩", width=64, height=32,
                          fg_color="#2a2a3e", hover_color="#3a3a5e",
                          command=lambda: self._seek_relative(10)).pack(side="left", padx=4)
            self.time_label = ctk.CTkLabel(btn_row, text="0:00 / 0:00",
                                           font=("Arial", 12), text_color="gray")
            self.time_label.pack(side="left", padx=12)

        # Right: analytics
        right = ctk.CTkFrame(body, corner_radius=12, width=320)
        right.grid(row=0, column=1, sticky="nsew")
        right.pack_propagate(False)

        ctk.CTkLabel(right, text="📊  Analytics", font=("Arial", 17, "bold")).pack(pady=(14, 4))

        self.count_card = ctk.CTkFrame(right, fg_color="#002d14", corner_radius=10)
        self.count_card.pack(fill="x", padx=14, pady=4)
        ctk.CTkLabel(self.count_card, text="System Status",
                     font=("Arial", 11), text_color="gray").pack(pady=(6, 0))
        self.count_label = ctk.CTkLabel(self.count_card, text="✅ Scene Safe",
                                        font=("Arial", 28, "bold"), text_color="#00ff88")
        self.count_label.pack(pady=(0, 6))

        self.status_label = ctk.CTkLabel(right, text="🟡 Resolving threat with LLM...",
                                         font=("Arial", 12))
        self.status_label.pack(pady=4)

        ctk.CTkLabel(right, text="🧠  Scene Analysis Log",
                     font=("Arial", 14, "bold")).pack(anchor="w", padx=14, pady=(12, 2))
        ctk.CTkLabel(right, text="Newest entries at top  •  Scroll to see history",
                     font=("Arial", 10), text_color="gray").pack(anchor="w", padx=14)

        self.scene_textbox = ctk.CTkTextbox(right, wrap="word", font=("Arial", 12),
                                            corner_radius=8)
        self.scene_textbox.pack(fill="both", expand=True, padx=14, pady=(4, 14))
        self.scene_textbox.insert("0.0", "Waiting for LLM to map threat and analyze scene...")
        self.scene_textbox.configure(state="disabled")

    def _stop_and_reset(self):
        self.running = False
        time.sleep(0.15)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.show_setup_screen()

    # ── LLM: Resolve threat → YOLO class ──────────────────────────────────────
    def _resolve_threat_with_llm(self):
        """Map user's natural language threat to closest YOLO class name."""
        user_input = self.threat_target.lower()

        # Step 1: Strip common prefixes like "detect", "find", "show me"
        for prefix in ["detect ", "find ", "show me ", "look for ", "identify ", "flag "]:
            if user_input.startswith(prefix):
                user_input = user_input[len(prefix):].strip()

        # Step 2: Check synonym map first (fast, reliable)
        for keyword, yolo_class in SYNONYM_MAP.items():
            if keyword in user_input:
                return yolo_class

        # Step 3: Direct YOLO class name match
        for cls in YOLO_CLASSES:
            if cls.lower() in user_input or user_input in cls.lower():
                return cls

        # Step 4: Ask LLM as last resort
        classes_str = ", ".join(YOLO_CLASSES)
        prompt = (
            f"You are a computer vision assistant.\n"
            f"The user wants to detect: \"{user_input}\"\n"
            f"Available YOLO detection classes: {classes_str}\n\n"
            f"Return ONLY the single best matching class name from the list above, "
            f"exactly as written, nothing else. No explanation, no punctuation."
        )
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": self.llm_model_name, "prompt": prompt, "stream": False},
                timeout=30
            )
            if resp.status_code == 200:
                mapped = resp.json().get("response", "").strip().lower()
                # Validate exact match
                for cls in YOLO_CLASSES:
                    if cls.lower() == mapped:
                        return cls
                # Partial match
                for cls in YOLO_CLASSES:
                    if cls.lower() in mapped or mapped in cls.lower():
                        return cls
        except Exception:
            pass

        # Final fallback: return cleaned input
        return user_input

    # ── Video Control Helpers ──────────────────────────────────────────────────
    def _toggle_pause(self):
        self.paused = not self.paused
        def _u():
            try:
                if self.play_pause_btn and self.play_pause_btn.winfo_exists():
                    self.play_pause_btn.configure(
                        text="▶ Play" if self.paused else "⏸ Pause"
                    )
            except Exception:
                pass
        self._safe_after(_u)

    def _restart_video(self):
        if self.cap:
            with self.cap_lock:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.seek_var.set(0)
            self.paused = False
            def _u():
                try:
                    if self.play_pause_btn and self.play_pause_btn.winfo_exists():
                        self.play_pause_btn.configure(text="⏸ Pause")
                except Exception:
                    pass
            self._safe_after(_u)

    def _seek_relative(self, seconds):
        if self.cap:
            with self.cap_lock:
                cur = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                new = max(0, min(cur + seconds * self.video_fps, self.total_frames - 1))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, new)

    def _on_seek_press(self):
        self.user_seeking = True

    def _on_seek_release(self):
        if self.cap:
            target_frame = int(self.seek_var.get())
            with self.cap_lock:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        self.user_seeking = False

    def _fmt_time(self, frame_pos):
        total_secs = int(frame_pos / self.video_fps) if self.video_fps else 0
        dur_secs   = int(self.total_frames / self.video_fps) if self.video_fps else 0
        def _fmt(s):
            return f"{s // 60}:{s % 60:02d}"
        return f"{_fmt(total_secs)} / {_fmt(dur_secs)}"

    def _update_seek_ui(self, frame_pos):
        def _u():
            try:
                if not self.user_seeking and self.seek_slider and self.seek_slider.winfo_exists():
                    self.seek_var.set(frame_pos)
                if self.time_label and self.time_label.winfo_exists():
                    self.time_label.configure(text=self._fmt_time(frame_pos))
            except Exception:
                pass
        self._safe_after(_u)

    # ── Video Loop ─────────────────────────────────────────────────────────────
    def video_loop(self):
        source_type = self.source_var.get()
        while self.running:
            # Handle pause
            if self.paused:
                time.sleep(0.05)
                continue

            if not self.cap or not self.cap.isOpened():
                break
            
            with self.cap_lock:
                ret, frame = self.cap.read()
            
            if not ret:
                if source_type == "Video":
                    with self.cap_lock:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            # Update seek bar for video mode
            if source_type == "Video":
                with self.cap_lock:
                    cur_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                self._update_seek_ui(cur_frame)

            detected_classes = []

            # Run both models
            res_general = self.yolo_general(frame, verbose=False, conf=0.25)
            res_weapon  = self.yolo_weapon(frame, verbose=False, conf=0.6) # High threshold for custom model

            # Store person boxes for conflict checking
            person_boxes = []
            for r in res_general:
                for box in r.boxes:
                    if r.names[int(box.cls[0])].lower() == "person" and float(box.conf[0]) > 0.5:
                        person_boxes.append(box.xyxy[0].tolist())

            for results in [res_general, res_weapon]:
                is_weapon_model = (results == res_weapon)
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        class_name = r.names[cls_id].lower()
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Conflict resolution
                        if is_weapon_model:
                            is_false_alarm = False
                            for p_box in person_boxes:
                                if x1 > p_box[0]-10 and y1 > p_box[1]-10 and x2 < p_box[2]+10 and y2 < p_box[3]+10:
                                    is_false_alarm = True
                                    break
                            if is_false_alarm: continue

                        detected_classes.append(class_name)
                        is_threat = bool(self.yolo_class_target and
                                         self.yolo_class_target in class_name)
                        
                        color = (0, 0, 255) if is_threat else (0, 200, 80)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_threat else 1)
                        lbl = f"{'!! THREAT' if is_threat else class_name} {conf:.0%}"
                        cv2.putText(frame, lbl, (x1, max(y1 - 10, 14)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Update V-JEPA and Latest Frame for Vision Stream
            self.vjepa_engine.add_frame(frame)
            self.latest_frame = frame.copy()

            threat_found = bool(self.yolo_class_target and self.yolo_class_target in detected_classes)
            self._set_threat_status(threat_found)

            if threat_found:
                self.threat_override = self.threat_target # Signal to Vision loop

            self.current_context = detected_classes

            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                if self.video_label and self.video_label.winfo_exists():
                    lw = self.video_label.winfo_width()
                    lh = self.video_label.winfo_height()
                    if lw > 10 and lh > 10:
                        ar = img.width / img.height
                        if lw / lh > ar:
                            img = img.resize((int(lh * ar), lh))
                        else:
                            img = img.resize((lw, int(lw / ar)))
                ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
                self._set_video_frame(ctk_img)
            except Exception:
                pass

            time.sleep(0.03)

    # ── LLM & Vision Streams ───────────────────────────────────────────────────
    def llm_loop(self):
        """Step 1: Resolve user threat to YOLO class."""
        self._set_status("🟡 Asking LLM to map threat target...")
        self.yolo_class_target = self._resolve_threat_with_llm()

        mapped_display = self.yolo_class_target
        def _update_badge():
            try:
                if self.target_badge and self.target_badge.winfo_exists():
                    self.target_badge.configure(
                        text=f"🎯 Target: \"{self.threat_target}\"  →  YOLO: [{mapped_display}]"
                    )
            except Exception:
                pass
        self._safe_after(_update_badge)
        self._set_status("🟢 Monitoring...")

        self._append_analysis(
            f"✅ Threat mapped:\nUser: \"{self.threat_target}\"\nYOLO class: [{self.yolo_class_target}]"
        )

        # Launch the Stream 3: Continuous Vision AI
        threading.Thread(target=self.vision_loop, daemon=True).start()

    def vision_loop(self):
        """Stream 3: The Continuous Vision AI using Gemma 4."""
        import base64
        time.sleep(3) # Wait for buffer

        while self.running:
            try:
                if self.latest_frame is None or self.paused:
                    time.sleep(1)
                    continue

                # 1. Get Temporal Anomaly Data
                self.anomaly_info = self.vjepa_engine.compute_anomaly()
                
                # 2. Encode Frame
                _, buffer = cv2.imencode(".jpg", self.latest_frame)
                img_base64 = base64.b64encode(buffer).decode("utf-8")

                vjepa_ctx = f"V-JEPA Anomaly Score: {self.anomaly_info['score']:.2f} ({self.anomaly_info['label']})"
                
                if self.threat_override:
                    prompt = (
                        f"ALERT: The automated system suspects a '{self.threat_override}' is present.\n"
                        f"Context: {vjepa_ctx}.\n\n"
                        f"Instructions:\n"
                        f"1. LOOK CAREFULLY at the scene. Is there actually a {self.threat_override} or anything suspicious?\n"
                        f"2. If YES, start your response with 'Threat recorded: {self.threat_override}.' then describe the person's actions.\n"
                        f"3. If NO (false alarm), just describe the scene normally in 2 sentences WITHOUT using the 'Threat recorded' phrase.\n"
                        f"4. Be honest and accurate."
                    )
                    self.threat_override = None # Reset flag
                else:
                    prompt = (
                        f"Describe what is happening in this scene in 2-3 sentences.\n"
                        f"Context: {vjepa_ctx}.\n"
                        f"If the scene is safe, just describe the activity."
                    )

                self._set_status("🟡 Vision AI Thinking...")

                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.vision_model_name,
                        "prompt": prompt,
                        "images": [img_base64],
                        "stream": False
                    },
                    timeout=60
                )
                if resp.status_code == 200:
                    reply = resp.json().get("response", "").strip()
                    is_threat = "Threat recorded:" in reply or "Threat recorded" in reply
                    
                    if is_threat:
                        border = "🚨" * 12
                        log_entry = f"{border}\n🚨 THREAT DETECTED 🚨\n{border}\n\n{reply.upper()}\n\n{vjepa_ctx}"
                        self._append_analysis(log_entry, is_threat=True)
                    else:
                        log_entry = f"👁 Scene Insight:\n{reply}\n({vjepa_ctx})"
                        self._append_analysis(log_entry, is_threat=False)
                        
                    self._set_status("🟢 Monitoring...")
                else:
                    self._set_status("🔴 Vision API Error")

            except Exception as e:
                self._append_analysis(f"❌ Vision Loop Error:\n{str(e)}")
                self._set_status("🔴 Error in Vision Thread")
                time.sleep(2)

            time.sleep(8) # Vision cycle timing


if __name__ == "__main__":
    app = ThreatDetectionApp()
    app.mainloop()
