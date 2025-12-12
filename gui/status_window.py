# gui/status_window.py
"""StatusDisplayWindow - AI音声とプロンプトを表示する別ウィンドウ"""

import tkinter as tk
from tkinter import ttk


class StatusDisplayWindow(tk.Toplevel):
    """A separate window to display AI speech and prompts."""
    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.title("AI Assistant Display")
        self.geometry("600x280")
        self.protocol("WM_DELETE_WINDOW", self._on_close_button)

        # This assumes the master is the Application instance
        if hasattr(master, 'toggle_status_window') and callable(master.toggle_status_window):
            self._toggle_main_app_visibility_method = master.toggle_status_window
        else:
            self._toggle_main_app_visibility_method = None

        self.ai_speech_var = tk.StringVar(value="AI: ")
        self.ai_speech_label = ttk.Label(
            self,
            textvariable=self.ai_speech_var,
            font=("Helvetica", 32, "bold"),
            wraplength=580,
            anchor="center",
            justify="center"
        )
        self.ai_speech_label.pack(padx=20, pady=(20, 10), fill=tk.BOTH, expand=True)

        self.prompt_message_var = tk.StringVar(value="")
        self.prompt_message_label = ttk.Label(
            self,
            textvariable=self.prompt_message_var,
            font=("Helvetica", 18),
            wraplength=580,
            anchor="center",
            justify="center",
            foreground="gray"
        )
        self.prompt_message_label.pack(padx=20, pady=(0, 20), fill=tk.X, expand=False)

    def _on_close_button(self):
        # When the status window's 'X' is clicked, toggle its visibility in the main app
        if self._toggle_main_app_visibility_method:
            self._toggle_main_app_visibility_method()
        else:
            self.withdraw()

    def set_ai_speech(self, text: str):
        if not self.winfo_exists(): return
        self.ai_speech_var.set(f"AI: {text}" if text else "AI: ")
        self.clear_prompt_message()
        self.update_idletasks()

    def clear_ai_speech_display(self):
        if not self.winfo_exists(): return
        self.ai_speech_var.set("AI: ")
        self.update_idletasks()

    def set_prompt_message(self, message: str):
        if not self.winfo_exists(): return
        self.prompt_message_var.set(message)
        self.update_idletasks()

    def clear_prompt_message(self):
        if not self.winfo_exists(): return
        self.prompt_message_var.set("")
        self.update_idletasks()
