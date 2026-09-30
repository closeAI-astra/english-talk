# 発音の受け入れ試験

本人が同じマイクで正しい発音と意図的に違う発音を録音し、ここに WAV を置きます。音声はリポジトリに含めません。

ファイル名は `分類__意図した英単語__correctまたはwrong__番号.wav` とします。例: `R_L__light__correct__01.wav`、`R_L__light__wrong__01.wav`。後者は「light」と言うつもりで「right」に近い音を出した録音です。使える分類は `R_L`、`TH`、`V_B`、`VOWEL`、`EPENTHESIS`、`F_H` です。

20組以上を正誤それぞれ録り、アプリのフォルダで `uv run evaluate_audio.py` を実行します。結果は `tests/audio_results.csv` に保存します。検出率70%以上、誤検出率10%以下は仕様書の目標値です。判定保留の録音は分母から除き、その数も報告します。
