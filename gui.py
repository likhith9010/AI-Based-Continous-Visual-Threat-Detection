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
    # Weapons (YOLO has no gun class, knife is closest)
    "gun": "knife", "pistol": "knife", "weapon": "knife", "firearm": "knife",
    "rifle": "knife", "revolver": "knife", "sword": "knife",
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

        print("Loading YOLOv8 object detector...")
        self.yolo_model = YOLO("yolov8n.pt")
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

        # Widget refs (nulled before screen destroy so threads don't crash)
        self.video_label = None
        self.count_label = None
        self.status_label = None
        self.scene_textbox = None

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

    def _set_count(self, n):
        def _u():
            try:
                if self.count_label and self.count_label.winfo_exists():
                    self.count_label.configure(text=str(n))
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

    def _append_analysis(self, text):
        """Append a timestamped entry; cap log at 100, prune oldest 25 when full."""
        ts = datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}]\n{text}\n"
        self.analysis_log.append(entry)
        if len(self.analysis_log) > 100:
            self.analysis_log = self.analysis_log[25:]  # drop oldest 25

        def _u():
            try:
                if self.scene_textbox and self.scene_textbox.winfo_exists():
                    self.scene_textbox.configure(state="normal")
                    self.scene_textbox.delete("0.0", "end")
                    # Newest first
                    self.scene_textbox.insert("0.0", "\n─────────────────────\n".join(
                        reversed(self.analysis_log)
                    ))
                    self.scene_textbox.configure(state="disabled")
                    self.scene_textbox.see("0.0")  # scroll to top (newest)
            except Exception:
                pass
        self._safe_after(_u)

    # ── Setup Screen ───────────────────────────────────────────────────────────
    def show_setup_screen(self):
        self.video_label = None
        self.count_label = None
        self.status_label = None
        self.scene_textbox = None

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
        self.running = True
        self.yolo_class_target = ""  # will be resolved by LLM

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

        # Right: analytics
        right = ctk.CTkFrame(body, corner_radius=12, width=320)
        right.grid(row=0, column=1, sticky="nsew")
        right.pack_propagate(False)

        ctk.CTkLabel(right, text="📊  Analytics", font=("Arial", 17, "bold")).pack(pady=(14, 4))

        count_card = ctk.CTkFrame(right, fg_color="#2d0000", corner_radius=10)
        count_card.pack(fill="x", padx=14, pady=4)
        ctk.CTkLabel(count_card, text="Threat Detections",
                     font=("Arial", 11), text_color="gray").pack(pady=(6, 0))
        self.count_label = ctk.CTkLabel(count_card, text="0",
                                        font=("Arial", 40, "bold"), text_color="#ff4d4d")
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

    # ── Video Loop ─────────────────────────────────────────────────────────────
    def video_loop(self):
        source_type = self.source_var.get()
        while self.running:
            if not self.cap or not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            if not ret:
                if source_type == "Video":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            detected_classes = []

            # Only run YOLO after we have the mapped class (avoid false highlights)
            results = self.yolo_model(frame, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    class_name = self.yolo_model.names[cls_id].lower()
                    conf = float(box.conf[0])
                    detected_classes.append(class_name)

                    is_threat = bool(self.yolo_class_target and
                                     self.yolo_class_target in class_name)
                    color = (0, 0, 255) if is_threat else (0, 200, 80)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if is_threat else 1)
                    lbl = f"{'!! THREAT' if is_threat else class_name} {conf:.0%}"
                    cv2.putText(frame, lbl, (x1, max(y1 - 10, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if self.yolo_class_target and self.yolo_class_target in detected_classes:
                self.detection_count += 1
                self._set_count(self.detection_count)

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

    # ── LLM Loop ───────────────────────────────────────────────────────────────
    def llm_loop(self):
        # Step 1: Resolve user threat to YOLO class
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

        # Immediately log the mapping
        self._append_analysis(
            f"✅ Threat mapped:\nUser: \"{self.threat_target}\"\nYOLO class: [{self.yolo_class_target}]"
        )

        # Wait for video to start populating context
        time.sleep(3)

        # Step 2: Periodic scene analysis
        while self.running:
            now = time.time()
            if now - self.last_llm_time >= 8 and self.current_context:
                self.last_llm_time = now
                objects_str = ", ".join(sorted(set(self.current_context)))
                threat_present = self.yolo_class_target in self.current_context

                prompt = (
                    f"You are a concise security AI.\n"
                    f"Objects detected in frame: {objects_str}.\n"
                    f"Monitored threat: \"{self.threat_target}\" (YOLO class: {self.yolo_class_target}).\n"
                    f"Threat visible right now: {'YES' if threat_present else 'NO'}.\n\n"
                    f"In 2-3 sentences: describe the scene, state if the threat is present, "
                    f"and if YES — is this a real concern or a false alarm?"
                )

                self._set_status("🟡 Thinking...")

                try:
                    resp = requests.post(
                        "http://localhost:11434/api/generate",
                        json={"model": self.llm_model_name, "prompt": prompt, "stream": False},
                        timeout=60
                    )
                    if resp.status_code == 200:
                        reply = resp.json().get("response", "").strip()
                        label = "🚨 THREAT DETECTED" if threat_present else "✅ Scene Clear"
                        self._append_analysis(f"{label}\n\n{reply}")
                        self._set_status("🟢 Monitoring...")
                    else:
                        self._append_analysis(f"⚠ Ollama error: HTTP {resp.status_code}")
                        self._set_status("🔴 LLM Error")

                except requests.exceptions.ConnectionError:
                    self._append_analysis("❌ Cannot reach Ollama.\nMake sure it is running.")
                    self._set_status("🔴 Ollama offline")
                except requests.exceptions.Timeout:
                    self._append_analysis("⏱ LLM timed out. Will retry next cycle.")
                    self._set_status("🟠 Timeout")
                except Exception as e:
                    self._append_analysis(f"Unexpected error:\n{str(e)}")

            time.sleep(1)


if __name__ == "__main__":
    app = ThreatDetectionApp()
    app.mainloop()
