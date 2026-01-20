# gui/status_window.py
"""StatusDisplayWindow - Amadeusスタイルの会話表示ウィンドウ"""

import tkinter as tk
from tkinter import ttk
import math


class StatusDisplayWindow(tk.Toplevel):
    """Amadeusスタイルの会話表示ウィンドウ"""

    # カラーテーマ
    COLORS = {
        'bg_dark': '#0a0f1e',
        'bg_panel': '#0d1117',
        'accent_blue': '#64b5f6',
        'accent_red': '#ff6b6b',
        'accent_green': '#69f0ae',
        'accent_yellow': '#ffd54f',
        'text_primary': '#c9d1d9',
        'text_secondary': '#8b949e',
        'border': '#30363d',
        'glow_idle': '#64b5f6',
        'glow_listening': '#ff6b6b',
        'glow_speaking': '#69f0ae',
        'glow_thinking': '#ffd54f',
    }

    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.title("HCS - AI Assistant")
        self.geometry("700x500")
        self.configure(bg=self.COLORS['bg_dark'])
        self.protocol("WM_DELETE_WINDOW", self._on_close_button)

        # 状態管理
        self.state = "idle"  # idle, listening, speaking, thinking
        self.animation_frame = 0
        self.pulse_direction = 1
        self.pulse_value = 0

        if hasattr(master, 'toggle_status_window') and callable(master.toggle_status_window):
            self._toggle_main_app_visibility_method = master.toggle_status_window
        else:
            self._toggle_main_app_visibility_method = None

        self._setup_ui()
        self._start_animation()

    def _setup_ui(self):
        """UIを構築"""
        # メインコンテナ
        main_frame = tk.Frame(self, bg=self.COLORS['bg_dark'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # 上部: タイトルとステータス
        header_frame = tk.Frame(main_frame, bg=self.COLORS['bg_dark'])
        header_frame.pack(fill=tk.X, pady=(0, 15))

        # タイトル
        title_label = tk.Label(
            header_frame,
            text="HCS",
            font=("Helvetica", 28, "bold"),
            fg=self.COLORS['accent_red'],
            bg=self.COLORS['bg_dark']
        )
        title_label.pack(side=tk.LEFT)

        subtitle_label = tk.Label(
            header_frame,
            text=" - Heart Rate Coupled Speech",
            font=("Helvetica", 14),
            fg=self.COLORS['text_secondary'],
            bg=self.COLORS['bg_dark']
        )
        subtitle_label.pack(side=tk.LEFT, pady=(10, 0))

        # ステータスインジケーター（右上）
        self.status_indicator = tk.Canvas(
            header_frame,
            width=120,
            height=30,
            bg=self.COLORS['bg_dark'],
            highlightthickness=0
        )
        self.status_indicator.pack(side=tk.RIGHT)

        # 中央: ビジュアライザーとAI発話
        center_frame = tk.Frame(main_frame, bg=self.COLORS['bg_dark'])
        center_frame.pack(fill=tk.BOTH, expand=True)

        # ビジュアライザー（円形のステータス表示）
        self.visualizer = tk.Canvas(
            center_frame,
            width=200,
            height=200,
            bg=self.COLORS['bg_dark'],
            highlightthickness=0
        )
        self.visualizer.pack(pady=10)

        # AI発話表示エリア
        speech_frame = tk.Frame(
            center_frame,
            bg=self.COLORS['bg_panel'],
            highlightbackground=self.COLORS['border'],
            highlightthickness=1
        )
        speech_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.ai_speech_var = tk.StringVar(value="")
        self.ai_speech_label = tk.Label(
            speech_frame,
            textvariable=self.ai_speech_var,
            font=("Helvetica", 24, "bold"),
            fg=self.COLORS['text_primary'],
            bg=self.COLORS['bg_panel'],
            wraplength=620,
            justify="center"
        )
        self.ai_speech_label.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

        # 下部: プロンプトメッセージ
        bottom_frame = tk.Frame(main_frame, bg=self.COLORS['bg_dark'])
        bottom_frame.pack(fill=tk.X, pady=(10, 0))

        self.prompt_message_var = tk.StringVar(value="")
        self.prompt_label = tk.Label(
            bottom_frame,
            textvariable=self.prompt_message_var,
            font=("Helvetica", 16),
            fg=self.COLORS['text_secondary'],
            bg=self.COLORS['bg_dark']
        )
        self.prompt_label.pack()

        # 心拍数表示（オプション）
        self.hr_var = tk.StringVar(value="")
        self.hr_label = tk.Label(
            bottom_frame,
            textvariable=self.hr_var,
            font=("Helvetica", 12),
            fg=self.COLORS['accent_red'],
            bg=self.COLORS['bg_dark']
        )
        self.hr_label.pack(pady=(5, 0))

    def _start_animation(self):
        """アニメーションを開始"""
        self._animate()

    def _animate(self):
        """アニメーションフレームを更新"""
        if not self.winfo_exists():
            return

        self.animation_frame += 1

        # パルスアニメーション
        self.pulse_value += 0.1 * self.pulse_direction
        if self.pulse_value >= 1.0:
            self.pulse_direction = -1
        elif self.pulse_value <= 0.0:
            self.pulse_direction = 1

        self._draw_visualizer()
        self._draw_status_indicator()

        # 50msごとに更新
        self.after(50, self._animate)

    def _draw_visualizer(self):
        """円形ビジュアライザーを描画"""
        canvas = self.visualizer
        canvas.delete("all")

        cx, cy = 100, 100
        base_radius = 60

        # 状態に応じた色
        state_colors = {
            'idle': self.COLORS['glow_idle'],
            'listening': self.COLORS['glow_listening'],
            'speaking': self.COLORS['glow_speaking'],
            'thinking': self.COLORS['glow_thinking'],
        }
        color = state_colors.get(self.state, self.COLORS['glow_idle'])

        # 外側のグロー効果
        for i in range(5, 0, -1):
            alpha = int(30 + self.pulse_value * 20)
            glow_radius = base_radius + i * 8 + self.pulse_value * 5
            # Tkinterはアルファをサポートしないので、色を薄くして擬似表現
            canvas.create_oval(
                cx - glow_radius, cy - glow_radius,
                cx + glow_radius, cy + glow_radius,
                outline=color, width=2
            )

        # メインの円
        canvas.create_oval(
            cx - base_radius, cy - base_radius,
            cx + base_radius, cy + base_radius,
            outline=color, width=3, fill=self.COLORS['bg_panel']
        )

        # 状態テキスト
        state_text = {
            'idle': 'STANDBY',
            'listening': '● 聞き取り中',
            'speaking': '♪ 発話中',
            'thinking': '◐ AI処理中',
        }
        canvas.create_text(
            cx, cy,
            text=state_text.get(self.state, 'STANDBY'),
            fill=color,
            font=("Helvetica", 14, "bold")
        )

        # リスニング時は波形を表示
        if self.state == 'listening':
            for i in range(8):
                angle = (self.animation_frame * 5 + i * 45) * math.pi / 180
                wave_height = 10 + 15 * abs(math.sin(angle + self.animation_frame * 0.2))
                x = cx + (base_radius + 20) * math.cos(i * math.pi / 4)
                y = cy + (base_radius + 20) * math.sin(i * math.pi / 4)
                canvas.create_line(
                    x, y - wave_height/2, x, y + wave_height/2,
                    fill=color, width=3
                )

        # スピーキング時は発話アニメーション
        elif self.state == 'speaking':
            for i in range(3):
                wave_offset = (self.animation_frame + i * 10) % 30
                canvas.create_arc(
                    cx - base_radius - 15 - wave_offset,
                    cy - base_radius - 15 - wave_offset,
                    cx + base_radius + 15 + wave_offset,
                    cy + base_radius + 15 + wave_offset,
                    start=45, extent=90, style=tk.ARC,
                    outline=color, width=2
                )
                canvas.create_arc(
                    cx - base_radius - 15 - wave_offset,
                    cy - base_radius - 15 - wave_offset,
                    cx + base_radius + 15 + wave_offset,
                    cy + base_radius + 15 + wave_offset,
                    start=225, extent=90, style=tk.ARC,
                    outline=color, width=2
                )

    def _draw_status_indicator(self):
        """右上のステータスインジケーターを描画"""
        canvas = self.status_indicator
        canvas.delete("all")

        state_colors = {
            'idle': self.COLORS['glow_idle'],
            'listening': self.COLORS['glow_listening'],
            'speaking': self.COLORS['glow_speaking'],
            'thinking': self.COLORS['glow_thinking'],
        }
        color = state_colors.get(self.state, self.COLORS['glow_idle'])

        # インジケーター点
        radius = 6 + self.pulse_value * 2
        canvas.create_oval(
            10 - radius, 15 - radius,
            10 + radius, 15 + radius,
            fill=color, outline=""
        )

        # ステータステキスト
        state_text = {
            'idle': 'STANDBY',
            'listening': 'LISTENING',
            'speaking': 'SPEAKING',
            'thinking': 'THINKING',
        }
        canvas.create_text(
            25, 15,
            text=state_text.get(self.state, 'STANDBY'),
            fill=color,
            font=("Menlo", 10),
            anchor="w"
        )

    def set_state(self, state: str):
        """状態を設定"""
        if state in ['idle', 'listening', 'speaking', 'thinking']:
            self.state = state

    def _on_close_button(self):
        if self._toggle_main_app_visibility_method:
            self._toggle_main_app_visibility_method()
        else:
            self.withdraw()

    def set_ai_speech(self, text: str):
        """AI発話テキストを設定"""
        if not self.winfo_exists():
            return
        self.ai_speech_var.set(text if text else "")
        self.set_state('speaking')
        self.update_idletasks()

    def clear_ai_speech_display(self):
        """AI発話表示をクリア"""
        if not self.winfo_exists():
            return
        self.ai_speech_var.set("")
        self.set_state('idle')
        self.update_idletasks()

    def set_prompt_message(self, message: str):
        """プロンプトメッセージを設定"""
        if not self.winfo_exists():
            return
        self.prompt_message_var.set(message)

        # メッセージに応じて状態を自動設定
        if "話してください" in message:
            self.set_state('idle')
        elif "聞き取り中" in message:
            self.set_state('listening')
        elif "AI 処理中" in message or "処理中" in message:
            self.set_state('thinking')
        elif "発話中" in message:
            self.set_state('speaking')

        self.update_idletasks()

    def clear_prompt_message(self):
        """プロンプトメッセージをクリア"""
        if not self.winfo_exists():
            return
        self.prompt_message_var.set("")
        self.update_idletasks()

    def set_heart_rate(self, hr: int):
        """心拍数を表示"""
        if not self.winfo_exists():
            return
        if hr and hr > 0:
            self.hr_var.set(f"♥ {hr} BPM")
        else:
            self.hr_var.set("")
        self.update_idletasks()
