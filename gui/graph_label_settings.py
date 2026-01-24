# gui/graph_label_settings.py
"""グラフラベル設定用の共通コンポーネント

グラフのタイトル、軸ラベル、条件名をGUIでカスタマイズ可能にするモジュール。
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional, List, Callable
from dataclasses import dataclass, field


@dataclass
class GraphLabelConfig:
    """グラフラベル設定を保持するデータクラス"""
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    condition_labels: Dict[str, str] = field(default_factory=dict)

    def get_condition_label(self, condition: str) -> str:
        """条件名のカスタムラベルを取得（未設定なら元の名前を返す）"""
        return self.condition_labels.get(condition, condition)

    def get_title(self, default: str) -> str:
        """タイトルを取得（未設定ならデフォルト値を返す）"""
        return self.title.strip() if self.title.strip() else default

    def get_x_label(self, default: str) -> str:
        """X軸ラベルを取得（未設定ならデフォルト値を返す）"""
        return self.x_label.strip() if self.x_label.strip() else default

    def get_y_label(self, default: str) -> str:
        """Y軸ラベルを取得（未設定ならデフォルト値を返す）"""
        return self.y_label.strip() if self.y_label.strip() else default


class ConditionLabelEntry(ttk.Frame):
    """条件名の横に配置するカスタムラベル入力欄"""

    def __init__(self, parent, condition_name: str, width: int = 12, **kwargs):
        super().__init__(parent, **kwargs)
        self.condition_name = condition_name
        self.label_var = tk.StringVar(value="")

        # 入力欄（プレースホルダーとして条件名を薄く表示）
        self.entry = ttk.Entry(self, textvariable=self.label_var, width=width)
        self.entry.pack(side=tk.LEFT, padx=(5, 0))

        # プレースホルダー動作
        self._setup_placeholder()

    def _setup_placeholder(self):
        """プレースホルダー表示の設定"""
        self.entry.insert(0, self.condition_name)
        self.entry.config(foreground='gray')
        self._is_placeholder = True

        self.entry.bind('<FocusIn>', self._on_focus_in)
        self.entry.bind('<FocusOut>', self._on_focus_out)

    def _on_focus_in(self, event):
        if self._is_placeholder:
            self.entry.delete(0, tk.END)
            self.entry.config(foreground='black')
            self._is_placeholder = False

    def _on_focus_out(self, event):
        if not self.entry.get().strip():
            self.entry.insert(0, self.condition_name)
            self.entry.config(foreground='gray')
            self._is_placeholder = True

    def get_label(self) -> str:
        """カスタムラベルを取得（未入力なら空文字）"""
        if self._is_placeholder:
            return ""
        return self.entry.get().strip()

    def set_label(self, label: str):
        """ラベルを設定"""
        self.entry.delete(0, tk.END)
        if label:
            self.entry.insert(0, label)
            self.entry.config(foreground='black')
            self._is_placeholder = False
        else:
            self.entry.insert(0, self.condition_name)
            self.entry.config(foreground='gray')
            self._is_placeholder = True


class GraphLabelSettingsFrame(ttk.LabelFrame):
    """グラフラベル設定用のフレーム（タイトル、軸ラベル）"""

    def __init__(self, parent, title: str = "グラフラベル設定（任意）", **kwargs):
        super().__init__(parent, text=title, **kwargs)

        self.title_var = tk.StringVar(value="")
        self.x_label_var = tk.StringVar(value="")
        self.y_label_var = tk.StringVar(value="")

        self._setup_ui()

    def _setup_ui(self):
        """UI構築"""
        # グラフタイトル
        title_frame = ttk.Frame(self)
        title_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(title_frame, text="グラフタイトル:", width=14).pack(side=tk.LEFT)
        ttk.Entry(title_frame, textvariable=self.title_var, width=40).pack(side=tk.LEFT, padx=5)
        ttk.Label(title_frame, text="(空欄でデフォルト)", foreground="gray").pack(side=tk.LEFT)

        # X軸ラベル
        x_frame = ttk.Frame(self)
        x_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(x_frame, text="X軸ラベル:", width=14).pack(side=tk.LEFT)
        ttk.Entry(x_frame, textvariable=self.x_label_var, width=25).pack(side=tk.LEFT, padx=5)

        # Y軸ラベル
        ttk.Label(x_frame, text="Y軸ラベル:", width=10).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Entry(x_frame, textvariable=self.y_label_var, width=25).pack(side=tk.LEFT, padx=5)

    def get_config(self) -> GraphLabelConfig:
        """現在の設定を取得"""
        return GraphLabelConfig(
            title=self.title_var.get(),
            x_label=self.x_label_var.get(),
            y_label=self.y_label_var.get()
        )

    def set_config(self, config: GraphLabelConfig):
        """設定を適用"""
        self.title_var.set(config.title)
        self.x_label_var.set(config.x_label)
        self.y_label_var.set(config.y_label)

    def clear(self):
        """設定をクリア"""
        self.title_var.set("")
        self.x_label_var.set("")
        self.y_label_var.set("")


class ConditionFilterWithLabels(ttk.LabelFrame):
    """条件フィルタ + カスタムラベル入力が一体になったフレーム"""

    def __init__(
        self,
        parent,
        conditions: List[str],
        colors: Dict[str, str],
        title: str = "条件フィルタ",
        label_prompt: str = "解析に含める条件を選択してください。",
        columns: int = 4,
        show_label_entry: bool = True,
        **kwargs
    ):
        super().__init__(parent, text=title, **kwargs)

        self.conditions = conditions
        self.colors = colors
        self.columns = columns
        self.show_label_entry = show_label_entry

        # 各条件のチェックボックス変数
        self.condition_vars: Dict[str, tk.BooleanVar] = {}
        # 各条件のカスタムラベル入力
        self.label_entries: Dict[str, ConditionLabelEntry] = {}

        self._setup_ui(label_prompt)

    def _setup_ui(self, label_prompt: str):
        """UI構築"""
        # 説明ラベル
        ttk.Label(self, text=label_prompt).pack(anchor="w", padx=5, pady=(5, 2))

        # 条件チェックボックス + ラベル入力のフレーム
        conditions_frame = ttk.Frame(self)
        conditions_frame.pack(fill=tk.X, padx=5, pady=5)

        for idx, condition in enumerate(self.conditions):
            row = idx // self.columns
            col = idx % self.columns

            # 各条件用のコンテナ
            cond_frame = ttk.Frame(conditions_frame)
            cond_frame.grid(row=row, column=col, sticky="w", padx=5, pady=2)

            # 色インジケーター
            color = self.colors.get(condition, "#999999")
            canvas = tk.Canvas(cond_frame, width=16, height=16, highlightthickness=0)
            canvas.create_rectangle(2, 2, 14, 14, fill=color, outline=color)
            canvas.pack(side=tk.LEFT)

            # チェックボックス
            var = tk.BooleanVar(value=True)
            self.condition_vars[condition] = var
            cb = ttk.Checkbutton(cond_frame, text=condition, variable=var)
            cb.pack(side=tk.LEFT, padx=(2, 0))

            # カスタムラベル入力欄
            if self.show_label_entry:
                label_entry = ConditionLabelEntry(cond_frame, condition, width=10)
                label_entry.pack(side=tk.LEFT)
                self.label_entries[condition] = label_entry

        # 全選択/解除ボタン
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=5, pady=(0, 5))
        ttk.Button(btn_frame, text="全て選択", command=self._select_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="全て解除", command=self._deselect_all).pack(side=tk.LEFT, padx=5)

        if self.show_label_entry:
            ttk.Button(btn_frame, text="ラベルをリセット", command=self._reset_labels).pack(side=tk.LEFT, padx=5)

    def _select_all(self):
        """全条件を選択"""
        for var in self.condition_vars.values():
            var.set(True)

    def _deselect_all(self):
        """全条件を解除"""
        for var in self.condition_vars.values():
            var.set(False)

    def _reset_labels(self):
        """ラベルをリセット"""
        for entry in self.label_entries.values():
            entry.set_label("")

    def get_selected_conditions(self) -> List[str]:
        """選択されている条件のリストを取得"""
        return [cond for cond, var in self.condition_vars.items() if var.get()]

    def get_condition_labels(self) -> Dict[str, str]:
        """カスタム条件ラベルの辞書を取得（設定されているもののみ）"""
        labels = {}
        for condition, entry in self.label_entries.items():
            label = entry.get_label()
            if label:
                labels[condition] = label
        return labels

    def get_label_config(self) -> GraphLabelConfig:
        """条件ラベル設定を含むConfigを取得"""
        return GraphLabelConfig(condition_labels=self.get_condition_labels())

    def set_condition_labels(self, labels: Dict[str, str]):
        """条件ラベルを設定"""
        for condition, entry in self.label_entries.items():
            entry.set_label(labels.get(condition, ""))


def apply_labels_to_axis(ax, config: GraphLabelConfig, default_title: str = "",
                         default_x: str = "", default_y: str = ""):
    """matplotlibのaxesにラベル設定を適用

    Args:
        ax: matplotlib axes
        config: GraphLabelConfig
        default_title: デフォルトのタイトル
        default_x: デフォルトのX軸ラベル
        default_y: デフォルトのY軸ラベル
    """
    ax.set_title(config.get_title(default_title))
    ax.set_xlabel(config.get_x_label(default_x))
    ax.set_ylabel(config.get_y_label(default_y))


def rename_conditions_in_dataframe(df, condition_col: str, labels: Dict[str, str]):
    """DataFrameの条件列をカスタムラベルで置換

    Args:
        df: pandas DataFrame
        condition_col: 条件列の名前
        labels: {元の条件名: カスタムラベル} の辞書

    Returns:
        条件名が置換されたDataFrameのコピー
    """
    if not labels:
        return df

    df_copy = df.copy()
    df_copy[condition_col] = df_copy[condition_col].replace(labels)
    return df_copy
