import sys
import os
import subprocess
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QTextEdit, 
                               QLineEdit, QFileDialog, QComboBox, QMessageBox)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QTextCursor, QTextCharFormat, QColor, QFont

from gesture_module import GestureProcessor
from voice_module import VoiceProcessor
from ocr_module import extract_text_from_canvas
from file_parser import parse_file

class CameraThread(QThread):
    change_pixmap = Signal(QImage)
    gesture_action = Signal(str)

    def __init__(self):
        super().__init__()
        self.running = False
        self.gesture_processor = GestureProcessor()
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.running = True
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, (640,480))
            if ret:
                processed_frame, action = self.gesture_processor.process_frame(frame)
                
                if action:
                    self.gesture_action.emit(action)
                
                # Convert to Qt format (OpenCV uses BGR, Qt expects RGB)
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = processed_frame_rgb.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(processed_frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_Qt_format.scaled(640, 480, Qt.KeepAspectRatio)
                self.change_pixmap.emit(p)
                
                # Sleep to limit FPS to ~30 and prevent the thread from starving the main GUI event loop
                QThread.msleep(20)
        
        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False
        self.wait()

    def get_canvas(self):
        return self.gesture_processor.get_canvas()

    def reset_canvas(self):
        self.gesture_processor.reset_canvas()

class OCRThread(QThread):
    finished = Signal(str, str) # text, error_msg

    def __init__(self, canvas_copy):
        super().__init__()
        self.canvas_copy = canvas_copy

    def run(self):
        text, error = extract_text_from_canvas(self.canvas_copy)
        self.finished.emit(text if text else "", error if error else "")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multimodal Intelligent Search Engine")
        self.setGeometry(100, 100, 1200, 800)
        
        self.current_file_path = ""
        self.current_text = ""
        
        self.current_match_indices = []
        self.current_match_pos = -1
        
        # Threads
        self.camera_thread = None
        
        # Voice Processor
        self.voice_processor = VoiceProcessor()
        self.voice_processor.command_detected.connect(self.handle_voice_command)
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        
        # Left Panel (Controls and Video)
        left_panel = QVBoxLayout()
        
        self.btn_load_file = QPushButton("Load Document")
        self.btn_load_file.clicked.connect(self.load_file)
        self.btn_load_file.setFixedHeight(40)
        left_panel.addWidget(self.btn_load_file)
        
        self.lbl_file_name = QLabel("No file selected")
        left_panel.addWidget(self.lbl_file_name)
        
        search_layout = QHBoxLayout()
        self.keyword_input = QLineEdit()
        self.keyword_input.setPlaceholderText("Enter keyword to search...")
        self.keyword_input.setFixedHeight(40)
        self.keyword_input.textChanged.connect(self.on_search_text_changed)
        search_layout.addWidget(self.keyword_input)
        
        self.combo_algo = QComboBox()
        self.combo_algo.addItems(["Naive", "KMP", "Rabin-Karp"])
        self.combo_algo.setFixedHeight(40)
        search_layout.addWidget(self.combo_algo)
        
        self.btn_search = QPushButton("Search")
        self.btn_search.clicked.connect(self.run_search)
        self.btn_search.setFixedHeight(40)
        search_layout.addWidget(self.btn_search)
        
        self.btn_prev = QPushButton("<")
        self.btn_prev.clicked.connect(self.prev_match)
        self.btn_prev.setFixedSize(40, 40)
        self.btn_prev.setStyleSheet("font-size: 16px; font-weight: bold;")
        search_layout.addWidget(self.btn_prev)
        
        self.btn_next = QPushButton(">")
        self.btn_next.clicked.connect(self.next_match)
        self.btn_next.setFixedSize(40, 40)
        self.btn_next.setStyleSheet("font-size: 16px; font-weight: bold;")
        search_layout.addWidget(self.btn_next)
        
        left_panel.addLayout(search_layout)
        
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: blue; font-weight: bold; font-size: 14px;")
        left_panel.addWidget(self.lbl_status)
        
        # Multi-modal buttons
        mm_layout = QHBoxLayout()
        self.btn_toggle_camera = QPushButton("Start Air Writing")
        self.btn_toggle_camera.clicked.connect(self.toggle_camera)
        self.btn_toggle_camera.setFixedHeight(50)
        self.btn_toggle_camera.setStyleSheet("background-color: #2c3e50; color: white;")
        mm_layout.addWidget(self.btn_toggle_camera)
        
        self.btn_toggle_voice = QPushButton("Start Voice Command")
        self.btn_toggle_voice.clicked.connect(self.toggle_voice)
        self.btn_toggle_voice.setFixedHeight(50)
        self.btn_toggle_voice.setStyleSheet("background-color: #e74c3c; color: white;")
        mm_layout.addWidget(self.btn_toggle_voice)
        
        left_panel.addLayout(mm_layout)
        
        # Video Label
        self.lbl_video = QLabel("Camera feed will appear here\n(Use Pinch gesture to write in air)")
        self.lbl_video.setFixedSize(640, 480)
        self.lbl_video.setStyleSheet("background-color: black; color: white; font-size: 16px;")
        self.lbl_video.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(self.lbl_video)
        
        # Instruction Label
        self.lbl_instructions = QLabel(
            "<b>🎯 Gesture Controls:</b><br>"
            "<b>☝️ Draw:</b> Point with index finger (others curled) to write.<br>"
            "<b>🖐️ Erase:</b> Open palm + swipe left/right to clear canvas.<br>"
            "<b>👍 Submit:</b> Thumbs-up to OCR &amp; search your writing.<br>"
            "<b>✌️ Pause:</b> Peace sign (V) to stop drawing.<br><br>"
            "<b>🎤 Voice Commands:</b><br>"
            "&nbsp;&nbsp;• <b>'done'</b> → OCR → fill search bar → auto search<br>"
            "&nbsp;&nbsp;• <b>'search'</b> → search with current bar text<br>"
            "&nbsp;&nbsp;• <b>'erase'</b> → clear canvas + search bar<br>"
            "&nbsp;&nbsp;• <b>'clear'</b> → clear search bar only"
        )
        self.lbl_instructions.setStyleSheet(
            "background-color: #1a1a2e; color: #e0e0e0; padding: 12px; "
            "font-size: 13px; border-radius: 8px; border: 1px solid #0f3460;"
        )
        left_panel.addWidget(self.lbl_instructions)
        
        # Fill remaining space
        left_panel.addStretch()
        
        # Right Panel (Text Editor)
        right_panel = QVBoxLayout()
        self.lbl_doc_title = QLabel("Document Viewer")
        self.lbl_doc_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        right_panel.addWidget(self.lbl_doc_title)
        
        self.text_editor = QTextEdit()
        self.text_editor.setReadOnly(True)
        self.text_editor.setStyleSheet("font-size: 14px; background-color: #f9f9f9; color: black;")
        right_panel.addWidget(self.text_editor)
        
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 1)
        
        
        central_widget.setLayout(main_layout)

    @Slot(str)
    def on_search_text_changed(self, text):
        # Clear matches and reset document view when user edits the search bar
        self.current_match_indices = []
        self.current_match_pos = -1
        
        if self.current_text:
            cursor = self.text_editor.textCursor()
            cursor.select(QTextCursor.Document)
            clear_fmt = QTextCharFormat()
            clear_fmt.setBackground(Qt.transparent)
            clear_fmt.setForeground(QColor("black"))
            clear_fmt.setFontWeight(QFont.Normal)
            cursor.setCharFormat(clear_fmt)
            cursor.clearSelection()
            self.text_editor.setTextCursor(cursor)
            
            if not text.strip():
                self.lbl_status.setText("Ready")
            else:
                self.run_search()

    @Slot(str)
    def handle_voice_command(self, cmd):
        print(f"Main thread handling voice cmd: {cmd}")

        if cmd == "search":
            # Search with whatever is currently in the search bar
            self.lbl_status.setText("🎤 Voice: Searching...")
            self.run_search()

        elif cmd == "done":
            # OCR the air writing → fill search bar → erase canvas → search
            self.lbl_status.setText("🎤 Voice: Processing air writing...")
            if self.camera_thread and self.camera_thread.running:
                self.process_ocr()
            else:
                self.lbl_status.setText("🎤 Voice: Camera not active — start Air Writing first.")

        elif cmd == "erase":
            # Clear both canvas AND search bar
            self.lbl_status.setText("🎤 Voice: Erased canvas & search bar")
            if self.camera_thread and self.camera_thread.running:
                self.camera_thread.reset_canvas()
            self.keyword_input.clear()
            # Reset document highlighting
            self._reset_document_highlighting()

        elif cmd == "clear":
            # Clear only the search bar
            self.lbl_status.setText("🎤 Voice: Cleared search bar")
            self.keyword_input.clear()
            self._reset_document_highlighting()

    def _reset_document_highlighting(self):
        """Remove all search highlights and reset match state."""
        self.current_match_indices = []
        self.current_match_pos = -1
        if self.current_text:
            cursor = self.text_editor.textCursor()
            cursor.select(QTextCursor.Document)
            clear_fmt = QTextCharFormat()
            clear_fmt.setBackground(Qt.transparent)
            clear_fmt.setForeground(QColor("black"))
            clear_fmt.setFontWeight(QFont.Normal)
            cursor.setCharFormat(clear_fmt)
            cursor.clearSelection()
            self.text_editor.setTextCursor(cursor)

    @Slot(str)
    def handle_gesture_action(self, action):
        if action == "swipe_left" or action == "swipe_right":
            self.lbl_status.setText("Gesture: Canvas Cleared")
            self.camera_thread.reset_canvas()
        elif action == "cancel":
            self.lbl_status.setText("Gesture: Canceled")
        elif action == "done":
            self.lbl_status.setText("Gesture: OCR Processing...")
            self.process_ocr()

    def process_ocr(self):
        if self.camera_thread and self.camera_thread.running:
            # Prevent starting a new thread if one is already running
            if hasattr(self, 'ocr_thread') and self.ocr_thread.isRunning():
                return
                
            # We copy the canvas explicitly so the drawing thread doesn't mutate it while Tesseract works
            canvas = self.camera_thread.get_canvas().copy()
            self.lbl_status.setText("OCR: Processing image in background...")
            
            self.ocr_thread = OCRThread(canvas)
            self.ocr_thread.finished.connect(self.on_ocr_finished)
            self.ocr_thread.start()

    @Slot(str, str)
    def on_ocr_finished(self, text, error_msg):
        if error_msg:
         QMessageBox.critical(self, "OCR Error", error_msg)
         self.lbl_status.setText("OCR: Failed due to error.")
        elif text:
            self.keyword_input.setText(text)
            self.lbl_status.setText(f"OCR Extracted: '{text}'. Searching...")
            self.run_search()

            # Clear canvas after OCR so next drawing is clean
            if self.camera_thread:
               self.camera_thread.reset_canvas()
            
        else:
          self.lbl_status.setText("OCR: No valid text found. Try writing again.")

    def toggle_camera(self):
        if self.camera_thread is None or not self.camera_thread.running:
            self.camera_thread = CameraThread()
            self.camera_thread.change_pixmap.connect(self.update_image)
            self.camera_thread.gesture_action.connect(self.handle_gesture_action)
            self.camera_thread.start()
            self.btn_toggle_camera.setText("Stop Air Writing")
            self.btn_toggle_camera.setStyleSheet("background-color: #27ae60; color: white;")
        else:
            self.camera_thread.stop()
            self.btn_toggle_camera.setText("Start Air Writing")
            self.btn_toggle_camera.setStyleSheet("background-color: #2c3e50; color: white;")
            self.lbl_video.clear()
            self.lbl_video.setText("Camera feed will appear here\n(Use Pinch gesture to write in air)")

    def toggle_voice(self):
        if self.voice_processor.stop_listening is None:
            self.voice_processor.start_listening()
            self.btn_toggle_voice.setText("Stop Voice Command")
            self.btn_toggle_voice.setStyleSheet("background-color: #27ae60; color: white;")
            self.lbl_status.setText("Voice Listening: ON")
        else:
            self.voice_processor.stop()
            self.btn_toggle_voice.setText("Start Voice Command")
            self.btn_toggle_voice.setStyleSheet("background-color: #e74c3c; color: white;")
            self.lbl_status.setText("Voice Listening: OFF")

    @Slot(QImage)
    def update_image(self, image):
        self.lbl_video.setPixmap(QPixmap.fromImage(image))

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Document", "", "Documents (*.txt *.pdf *.docx)")
        if file_path:
            self.current_file_path = file_path
            self.lbl_file_name.setText(os.path.basename(file_path))
            
            # parse
            self.current_text = parse_file(file_path)
            self.text_editor.setPlainText(self.current_text)
            self.lbl_status.setText(f"Loaded: {os.path.basename(file_path)}")

    def run_search(self):
        kw = self.keyword_input.text().strip()
        if not kw:
            QMessageBox.warning(self, "Warning", "Please enter a keyword.")
            return
        if not self.current_text:
            QMessageBox.warning(self, "Warning", "Please load a document first.")
            return
            
        algo_name = self.combo_algo.currentText()
        if algo_name == "Naive":
            algo_flag = "naive"
        elif algo_name == "KMP":
            algo_flag = "kmp"
        else:
            algo_flag = "rk"
            
        # Write text to temp file
        temp_path = "temp_search_text.txt"
        with open(temp_path, "w", encoding="utf-8", newline="") as f:
            f.write(self.current_text)
            
        backend_exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend.exe")
        if not os.path.exists(backend_exe):
            QMessageBox.critical(self, "Error", f"Backend executable not found at {backend_exe}")
            return
            
        cmd = [backend_exe, algo_flag, kw, temp_path]
        try:
            # We must use proper startupinfo to prevent terminal popup on Windows
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

            result = subprocess.run(cmd, capture_output=True, text=True, check=True, startupinfo=startupinfo)
            output = result.stdout.strip()
            
            if output:
                indices = [int(x) for x in output.split()]
            else:
                indices = []
                
            self.lbl_status.setText(f"Found {len(indices)} matches using {algo_name}.")
            self.highlight_matches(kw, indices)
            
        except subprocess.CalledProcessError as e:
            QMessageBox.critical(self, "Error", f"Backend error:\n{e.stderr}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def highlight_matches(self, kw, backend_indices):
        # 1. Reset state
        self.current_match_indices = []
        self.current_match_pos = -1
        
        # 2. Clear previous highlights
        cursor = self.text_editor.textCursor()
        cursor.select(QTextCursor.Document)
        clear_fmt = QTextCharFormat()
        clear_fmt.setBackground(Qt.transparent)
        clear_fmt.setForeground(QColor("black"))
        clear_fmt.setFontWeight(QFont.Normal)
        cursor.setCharFormat(clear_fmt)
        cursor.clearSelection()
        
        if not backend_indices:
            self.lbl_status.setText(f"No matches found for '{kw}'.")
            return
        
        # 3. Re-find all matches in Python to get correct character positions
        #    (C++ backend returns byte offsets which don't match Qt character
        #    positions when the text contains multi-byte UTF-8 characters)
        kw_lower = kw.lower()
        text_lower = self.current_text.lower()
        kw_len = len(kw)
        
        python_indices = []
        start = 0
        while True:
            pos = text_lower.find(kw_lower, start)
            if pos == -1:
                break
            python_indices.append(pos)
            start = pos + 1
            
        if not python_indices:
            self.lbl_status.setText(f"No matches found for '{kw}'.")
            return
        
        # 4. Apply highlighting using correct Python character positions
        fmt = QTextCharFormat()
        fmt.setBackground(QColor("#FFEB3B"))
        fmt.setForeground(QColor("black"))
        fmt.setFontWeight(QFont.Bold)
        
        self.text_editor.blockSignals(True)
        
        for idx in python_indices:
            cursor.setPosition(idx)
            cursor.setPosition(idx + kw_len, QTextCursor.KeepAnchor)
            cursor.setCharFormat(fmt)
        
        self.text_editor.blockSignals(False)
        
        self.current_match_indices = python_indices
        self.current_match_pos = 0

        # Scroll to first match
        if self.current_match_indices:
            self.scroll_to_match()

    def scroll_to_match(self):
        if not self.current_match_indices:
            return
            
        idx = self.current_match_indices[self.current_match_pos]
        kw_len = len(self.keyword_input.text().strip())
        
        cursor = self.text_editor.textCursor()
        cursor.setPosition(idx)
        cursor.setPosition(idx + kw_len, QTextCursor.KeepAnchor)
        self.text_editor.setTextCursor(cursor)
        self.text_editor.ensureCursorVisible()
        
        self.lbl_status.setText(f"Match {self.current_match_pos + 1} of {len(self.current_match_indices)}")

    def next_match(self):
        if self.current_match_indices:
            self.current_match_pos = (self.current_match_pos + 1) % len(self.current_match_indices)
            self.scroll_to_match()

    def prev_match(self):
        if self.current_match_indices:
            self.current_match_pos = (self.current_match_pos - 1 + len(self.current_match_indices)) % len(self.current_match_indices)
            self.scroll_to_match()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
