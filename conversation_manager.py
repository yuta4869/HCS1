# conversation_manager.py

import datetime
import logging
import time
import queue
from typing import Optional, List, Dict, Any

import config
from logger_utils import get_timestamped_log_path


class ConversationManager:
    """
    会話の履歴管理と、会話内容の記録（本文と表形式）を担当する簡易管理クラス。

    この版では
      - 音声入出力は AudioProcessor に任せる
      - ここでは会話履歴と記録だけを行う
      - 応答生成はひとまず「単純な固定応答」にしてあるので、
        実際の大規模言語モデルとの連携は後で書き換えてよい
    """

    def __init__(
        self,
        audio_processor: Any,
        hr_monitor: Any,
        h10_monitor: Any,
        app_ref: Any,
        log_queue_ref: Optional[queue.Queue] = None,
        max_history: int = 10,
    ) -> None:
        # 参照を保持（将来ここで使いたい場合に備えて）
        self.audio_processor = audio_processor
        self.hr_monitor = hr_monitor
        self.h10_monitor = h10_monitor
        self.app = app_ref

        # 会話履歴（system / user / assistant の辞書の列）
        self.max_history: int = max_history
        self.conversation_history: List[Dict[str, Any]] = []

        # ログ用
        self.log_queue: Optional[queue.Queue] = log_queue_ref
        self.log_filepath: Optional[str] = None
        self.csv_log_filepath: Optional[str] = None
        self.current_session_timestamp_for_csv: Optional[str] = None

        # 会話ループ用
        self._stop_conversation: bool = False

    # ------------------------------------------------------------------
    # 会話履歴まわり
    # ------------------------------------------------------------------
    def set_system_prompt(self, prompt: str) -> None:
        """system役の最初の文を履歴の先頭に入れる（既存のsystemは消す）。"""
        # 既存の system を取り除く
        self.conversation_history = [m for m in self.conversation_history if m.get("role") != "system"]
        # 先頭に追加
        self.conversation_history.insert(0, {"role": "system", "content": prompt})

    def add_message(
        self,
        role: str,
        content: str,
        start_time: Optional[datetime.datetime] = None,
        end_time: Optional[datetime.datetime] = None,
    ) -> None:
        """履歴に一件追加し、必要であればログにも書き出す。"""
        if not content:
            return

        msg: Dict[str, Any] = {"role": role, "content": content}
        if start_time is not None:
            msg["start_time"] = start_time
        if end_time is not None:
            msg["end_time"] = end_time

        # system は先頭、それ以外は末尾に追加
        if role == "system":
            self.set_system_prompt(content)
        else:
            self.conversation_history.append(msg)
            # 最大件数を超えたら古いものから削除（system は残す）
            non_system = [m for m in self.conversation_history if m.get("role") != "system"]
            while len(non_system) > self.max_history:
                # 最初の非systemを削除
                for i, m in enumerate(self.conversation_history):
                    if m.get("role") != "system":
                        del self.conversation_history[i]
                        break
                non_system = [m for m in self.conversation_history if m.get("role") != "system"]

        # ログにも書き出す
        self._log_message(role, content, start_time, end_time)

    def get_history_for_llm(self) -> List[Dict[str, str]]:
        """大規模言語モデルに渡す形（role/contentだけ）で履歴を返す。"""
        simple_list: List[Dict[str, str]] = []
        for m in self.conversation_history:
            simple_list.append({"role": m["role"], "content": m["content"]})
        return simple_list

    # ------------------------------------------------------------------
    # ログまわり
    # ------------------------------------------------------------------
    def _ensure_text_log_open(self) -> None:
        """本文ログ用の文章ファイルを開く。"""
        if self.log_filepath is not None:
            return
        # config.CONVERSATION_LOG_FILE_TEMPLATE を使ってファイル名を決める
        template = config.CONVERSATION_LOG_FILE_TEMPLATE
        filepath = get_timestamped_log_path(template)
        self.log_filepath = filepath
        try:
            with open(self.log_filepath, "a", encoding="utf-8") as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                f.write(f"=== Conversation log started at {timestamp} ===\n\n")
            print(f"Conversation text log: {self.log_filepath}")
        except Exception as e:
            print(f"Failed to open conversation text log: {e}")
            self.log_filepath = None

    def _ensure_csv_log_open(self) -> None:
        """表形式ログ用のCSVをロギングすれっどに依頼して用意する。"""
        if self.csv_log_filepath is not None:
            return
        if self.log_queue is None:
            return

        # セッション用の時間印
        self.current_session_timestamp_for_csv = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        template = config.CONVERSATION_CSV_LOG_FILE_TEMPLATE
        csv_path = get_timestamped_log_path(template, session_timestamp=self.current_session_timestamp_for_csv)
        self.csv_log_filepath = csv_path

        header = ["timestamp", "role", "content", "start_time", "end_time"]
        # LoggingThread への指示: ("add_handler", logger_name, filepath, header_list)
        try:
            self.log_queue.put(("add_handler", config.LOGGER_CONVERSATION_CSV, csv_path, header))
            print(f"Conversation CSV log: {csv_path}")
        except Exception as e:
            print(f"Failed to request CSV handler for conversation: {e}")
            self.csv_log_filepath = None
            self.current_session_timestamp_for_csv = None

    def _log_message(
        self,
        role: str,
        content: str,
        start_time: Optional[datetime.datetime],
        end_time: Optional[datetime.datetime],
    ) -> None:
        """一件の会話を文章ファイルおよびCSVに記録する。"""
        timestamp = datetime.datetime.now()

        # 文章ログ
        self._ensure_text_log_open()
        if self.log_filepath is not None:
            try:
                with open(self.log_filepath, "a", encoding="utf-8") as f:
                    ts = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
                    f.write(f"[{ts}] {role.upper()}: {content}\n")
                    if start_time or end_time:
                        st = start_time.strftime("%Y-%m-%d %H:%M:%S.%f") if start_time else ""
                        et = end_time.strftime("%Y-%m-%d %H:%M:%S.%f") if end_time else ""
                        f.write(f"    start={st}, end={et}\n")
                    f.write("\n")
            except Exception as e:
                print(f"Failed to write to conversation text log: {e}")

        # CSV ログ
        if self.log_queue is not None:
            self._ensure_csv_log_open()
            if self.csv_log_filepath is not None:
                ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
                st_str = start_time.strftime("%Y-%m-%d %H:%M:%S.%f") if start_time else ""
                et_str = end_time.strftime("%Y-%m-%d %H:%M:%S.%f") if end_time else ""
                payload = [ts_str, role, content, st_str, et_str]
                record = logging.LogRecord(
                    name=config.LOGGER_CONVERSATION_CSV,
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg="",
                    args=(payload,),
                    exc_info=None,
                )
                # CsvFileHandler は record.payload または args[0] を見る実装になっている
                record.payload = payload  # type: ignore[attr-defined]
                try:
                    self.log_queue.put(record)
                except Exception as e:
                    print(f"Failed to enqueue CSV conversation record: {e}")

    def close(self) -> None:
        """会話ログ用のCSVハンドラを閉じるようロギングすれっどに依頼する。"""
        if self.log_queue is None:
            return
        if self.csv_log_filepath is None:
            return
        try:
            self.log_queue.put(("remove_handler", config.LOGGER_CONVERSATION_CSV))
        except Exception as e:
            print(f"Failed to request removal of CSV handler: {e}")
        self.csv_log_filepath = None
        self.current_session_timestamp_for_csv = None

    # ------------------------------------------------------------------
    # 応答生成（簡易版）と会話ループ
    # ------------------------------------------------------------------
    def generate_reply(self, user_text: str, system_prompt: str) -> str:
        """
        利用中の大規模言語モデルとの結合は環境ごとに異なるはずなので、
        ここでは「単純な固定応答＋エコー」にしておく。

        必要に応じて、openai やローカル LLM への問い合わせ処理に置き換えてほしい。
        """
        # system プロンプトを更新（ログにも残る）
        if system_prompt:
            self.add_message("system", system_prompt)

        self.add_message("user", user_text)

        # ここでは単純な応答にしておく
        reply = f"あなたは「{user_text}」と言いました。今は心拍数連動の音声と動作確認用の簡易応答です。"

        self.add_message("assistant", reply)
        return reply

    def conversation_loop(self) -> None:
        """
        録音→文字起こし→応答生成→音声出力、をくり返す簡易会話ループ。
        Application 側から別すれっどで呼び出されることを想定。
        """
        self._stop_conversation = False
        if self.app is not None:
            try:
                self.app.set_status("会話ループを開始しました。話しかけてください。", "green")
            except Exception:
                pass

        while not self._stop_conversation:
            try:
                # 入力録音
                if self.app is not None:
                    self.app.set_status("録音待機中...", "orange")
                ok, rec_start, rec_end = self.audio_processor.record_audio()
                if not ok:
                    # 中断などを考慮して、少し待って続行
                    time.sleep(0.5)
                    continue

                # 文字起こし
                if self.app is not None:
                    self.app.set_status("音声認識中...", "orange")
                user_text = self.audio_processor.speech_to_text("input.wav")
                if not user_text:
                    if self.app is not None:
                        self.app.append_log("[System] 音声が認識できませんでした。")
                    continue

                if self.app is not None:
                    self.app.append_log(f"[User] {user_text}")

                # 応答生成
                sys_prompt = ""
                if self.app is not None and hasattr(self.app, "system_prompt"):
                    try:
                        sys_prompt = self.app.system_prompt.get("1.0", "end").strip()
                    except Exception:
                        sys_prompt = ""
                reply_text = self.generate_reply(user_text, sys_prompt)

                if self.app is not None:
                    self.app.append_log(f"[Assistant] {reply_text}")
                    self.app.set_status("応答を音声で再生中...", "green")

                # 音声出力
                self.audio_processor.text_to_speech(reply_text)

            except Exception as e:
                print(f"Error in conversation loop: {e}")
                if self.app is not None:
                    try:
                        self.app.set_status(f"会話ループ中にエラー: {e}", "red")
                    except Exception:
                        pass
                # 大きな例外があった場合はいったん抜ける
                break

        if self.app is not None:
            try:
                self.app.set_status("会話ループを終了しました。", "blue")
            except Exception:
                pass

    def stop_conversation(self) -> None:
        """会話ループ終了要求。"""
        self._stop_conversation = True
