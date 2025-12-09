# local_llm.py
"""
ローカルLLM (llama-cpp-python) を使用したテキスト生成モジュール
Japanese StableLM Instruct Gamma 7B モデルを使用

Mac (Metal) および Ubuntu (CUDA/CPU) に対応
"""

import os
import sys
from typing import List, Dict, Optional

import config


class LocalLLM:
    """llama-cpp-python を使用したローカルLLMラッパー"""

    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """モデルをロード"""
        try:
            from llama_cpp import Llama
        except ImportError as e:
            print(f"エラー: llama-cpp-python がインストールされていません。")
            print(f"インストール方法については setup_local_llm.md を参照してください。")
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Please run the appropriate installation command for your platform."
            ) from e

        model_path = config.LOCAL_LLM_MODEL_PATH

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"モデルファイルが見つかりません: {model_path}\n"
                f"setup_local_llm.md の手順に従ってモデルをダウンロードしてください。"
            )

        print(f"ローカルLLMモデルをロード中: {model_path}")
        print(f"  - コンテキスト長: {config.LOCAL_LLM_N_CTX}")
        print(f"  - GPUレイヤー数: {config.LOCAL_LLM_N_GPU_LAYERS}")

        try:
            self.model = Llama(
                model_path=model_path,
                n_ctx=config.LOCAL_LLM_N_CTX,
                n_gpu_layers=config.LOCAL_LLM_N_GPU_LAYERS,
                verbose=False,
            )
            print("ローカルLLMモデルのロード完了")
        except Exception as e:
            print(f"モデルロードエラー: {e}")
            raise

    def _format_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        メッセージリストをJapanese StableLM Instruct形式のプロンプトに変換

        Japanese StableLM Instruct のプロンプト形式:
        以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

        ### 指示:
        {system_prompt + user_message}

        ### 応答:
        """
        system_content = ""
        user_content = ""
        conversation_history = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                system_content = content
            elif role == "user":
                user_content = content
            elif role == "assistant":
                # 過去の会話履歴として保存
                conversation_history.append(f"ユーザー: {user_content}\nアシスタント: {content}")
                user_content = ""  # リセット

        # プロンプト構築
        prompt = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。\n\n"

        # システムプロンプトがあれば指示に含める
        instruction = system_content if system_content else "ユーザーの質問に丁寧に日本語で答えてください。"

        prompt += f"### 指示:\n{instruction}\n\n"

        # 会話履歴があれば入力として追加
        if conversation_history:
            prompt += f"### 入力:\n過去の会話:\n" + "\n".join(conversation_history) + f"\n\n現在のユーザーの発言: {user_content}\n\n"
        else:
            prompt += f"### 入力:\n{user_content}\n\n"

        prompt += "### 応答:\n"

        return prompt

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        メッセージリストからテキストを生成

        Args:
            messages: OpenAI API形式のメッセージリスト
                [{"role": "system", "content": "..."},
                 {"role": "user", "content": "..."}]

        Returns:
            生成されたテキスト
        """
        if self.model is None:
            raise RuntimeError("モデルがロードされていません")

        prompt = self._format_prompt(messages)

        try:
            output = self.model(
                prompt,
                max_tokens=config.LOCAL_LLM_MAX_TOKENS,
                temperature=config.LOCAL_LLM_TEMPERATURE,
                top_p=config.LOCAL_LLM_TOP_P,
                repeat_penalty=config.LOCAL_LLM_REPEAT_PENALTY,
                stop=["### 指示:", "### 入力:", "\n\n###"],  # 停止トークン
                echo=False,
            )

            generated_text = output["choices"][0]["text"].strip()

            # 繰り返しや余分な出力を除去
            # 「こんにちは」などの入力文が繰り返される場合は最初の応答のみを取り出す
            lines = generated_text.split('\n')
            clean_lines = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # 最初の完結した応答を取得（句点で終わるまで）
                clean_lines.append(line)
                if line.endswith('。') or line.endswith('！') or line.endswith('？'):
                    break

            generated_text = '\n'.join(clean_lines) if clean_lines else generated_text.split('\n')[0]
            generated_text = generated_text.strip()

            # 空の応答の場合
            if not generated_text:
                return "申し訳ありません、応答を生成できませんでした。"

            return generated_text

        except Exception as e:
            print(f"テキスト生成エラー: {e}")
            return f"エラーが発生しました: {str(e)}"

    def __del__(self):
        """デストラクタ - リソースの解放"""
        if self.model is not None:
            del self.model
            self.model = None


# シングルトンインスタンス（オプション）
_llm_instance: Optional[LocalLLM] = None


def get_local_llm() -> LocalLLM:
    """ローカルLLMのシングルトンインスタンスを取得"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LocalLLM()
    return _llm_instance
