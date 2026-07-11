import speech_recognition as sr
from PySide6.QtCore import QObject, Signal
import time


class VoiceProcessor(QObject):
    command_detected = Signal(str)

    def __init__(self):
        super().__init__()
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.stop_listening = None

        # Tune recognizer for responsiveness
        self.recognizer.energy_threshold = 400
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.5

        # Debounce: prevent same command from firing twice within 1.5s
        self._last_cmd = ""
        self._last_cmd_time = 0
        self._debounce_secs = 1.5

        print("Adjusting for ambient noise... Please wait.")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1.5)
        print("Done adjusting.")

    def start_listening(self):
        if self.stop_listening is None:
            # Re-calibrate ambient noise each time
            print("Re-calibrating ambient noise...")
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

            print("Started background listening for voice commands...")
            self.stop_listening = self.recognizer.listen_in_background(
                self.microphone,
                self.audio_callback,
                phrase_time_limit=5
            )

    def stop(self):
        if self.stop_listening is not None:
            self.stop_listening(wait_for_stop=False)
            self.stop_listening = None
            print("Stopped listening.")

    def _emit_debounced(self, cmd):
        """Only emit if this command wasn't just fired."""
        now = time.time()
        if cmd == self._last_cmd and (now - self._last_cmd_time) < self._debounce_secs:
            print(f"  → Debounced (skipping duplicate '{cmd}')")
            return
        self._last_cmd = cmd
        self._last_cmd_time = now
        self.command_detected.emit(cmd)

    def audio_callback(self, recognizer, audio):
        try:
            text = recognizer.recognize_google(audio).lower().strip()
            print(f"Voice recognized: '{text}'")

            # ── Match commands with broad fuzzy aliases ──
            # Check each command group. Order matters — most specific first.

            # "erase" — clear canvas AND search bar
            if self._matches(text, [
                "erase", "erased", "erasing", "a race", "arrays",
                "race", "raise", "raised", "rase", "eras"
            ]):
                print("  → Command: ERASE")
                self._emit_debounced("erase")
                return

            # "done" — OCR + fill search bar + search
            if self._matches(text, [
                "done", "don", "dan", "dun", "ton", "down",
                "dawn", "run", "one", "gone", "bon",
                "finish", "finished", "submit"
            ]):
                print("  → Command: DONE")
                self._emit_debounced("done")
                return

            # "clear" — clear only the search bar
            if self._matches(text, [
                "clear", "cleared", "clearing", "clean",
                "claire", "clue", "gear"
            ]):
                print("  → Command: CLEAR")
                self._emit_debounced("clear")
                return

            # "search" — search with current bar text
            if self._matches(text, [
                "search", "searched", "searching", "surge",
                "such", "church", "sir", "serve"
            ]):
                print("  → Command: SEARCH")
                self._emit_debounced("search")
                return

            print(f"  → No command matched for: '{text}'")

        except sr.UnknownValueError:
            pass  # Unrecognized speech — ignore
        except sr.RequestError as e:
            print(f"Could not request results; {e}")

    @staticmethod
    def _matches(text, aliases):
        """Check if any alias appears as a word in the recognized text."""
        words = text.split()
        for alias in aliases:
            # Check both as substring and as whole word
            if alias in words or alias in text:
                return True
        return False
