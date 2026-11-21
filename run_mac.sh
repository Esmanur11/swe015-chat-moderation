#!/usr/bin/env bash
# One-command runner for macOS / Linux (PyCharm or VS Code terminal).
#
#   bash run_mac.sh          # 1) setup  2) quick test  3) full training + evaluation
#   bash run_mac.sh app      # start the chat app  ->  http://127.0.0.1:7860
#   bash run_mac.sh slides "Ad Soyad" "123456" "https://github.com/..." "https://youtu.be/..."
set -e
cd "$(dirname "$0")"

# ---- pick a Python 3.10 – 3.13 (PyTorch / gensim wheels); also look inside Anaconda / Homebrew / python.org installs
ok() { "$1" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; }
PY=""
for c in python3.12 python3.11 python3.13 python3.10 \
         "$HOME/anaconda3/bin/python" /opt/anaconda3/bin/python "$HOME/opt/anaconda3/bin/python" \
         "$HOME/miniconda3/bin/python" /opt/miniconda3/bin/python /opt/homebrew/bin/python3.12 \
         /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
         /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && ok "$c"; then PY="$c"; break; fi
done
# no suitable Python: let conda create one inside the project folder
if [ -z "$PY" ] && [ ! -x .pyenv312/bin/python ]; then
  CONDA=""
  for c in conda "$HOME/anaconda3/bin/conda" /opt/anaconda3/bin/conda "$HOME/opt/anaconda3/bin/conda" "$HOME/miniconda3/bin/conda" /opt/miniconda3/bin/conda; do
    if command -v "$c" >/dev/null 2>&1; then CONDA="$c"; break; fi
  done
  if [ -n "$CONDA" ]; then
    echo "▶ Uygun Python yok, conda ile Python 3.12 kuruluyor (2–3 dk)..."
    "$CONDA" create -y -q -p ./.pyenv312 python=3.12 >/dev/null
  fi
fi
[ -z "$PY" ] && [ -x .pyenv312/bin/python ] && PY="$PWD/.pyenv312/bin/python"
if [ -z "$PY" ]; then
  echo "❌ Python 3.10 – 3.13 bulunamadı. https://www.python.org/downloads/release/python-3129/ adresinden"
  echo "   'macOS 64-bit universal2 installer' indir, kur, sonra bu komutu tekrar çalıştır."
  exit 1
fi
echo "▶ Kullanılan Python: $PY ($("$PY" --version 2>&1))"

# ---- virtual environment (.venv) + packages, only the first time
if [ ! -f .venv-swe/.installed ]; then
  echo "▶ Kurulum: $($PY --version) ile .venv oluşturuluyor ve paketler yükleniyor (5–10 dk, bir kez)..."
  "$PY" -m venv .venv-swe
  .venv-swe/bin/python -m pip install -q --upgrade pip
  .venv-swe/bin/python -m pip install -q -r requirements.txt
  touch .venv-swe/.installed
fi
P=.venv-swe/bin/python
# keep the Mac awake while training
KEEP=""; command -v caffeinate >/dev/null 2>&1 && KEEP="caffeinate -i"

case "${1:-all}" in
  app)
    echo "▶ Uygulama açılıyor: tarayıcıda http://127.0.0.1:7860 adresine git (durdurmak için Ctrl+C)"
    $P app.py ;;
  slides)
    $P fill_slides.py --name "$2" --id "$3" --repo "$4" --video "$5"
    open presentation.pptx 2>/dev/null || true ;;
  quick)
    $P train_all.py --quick ;;
  all)
    echo "▶ 1/2 HIZLI TEST (3–5 dk)"
    $KEEP $P train_all.py --quick
    echo; echo "▶ 2/2 ASIL EĞİTİM (Apple Silicon Mac'te ~30–60 dk). Bilgisayarı kapatma, uykuya geçmesin."
    $KEEP $P train_all.py
    echo; echo "✅ Bitti. Sonuçlar: results/results.md"
    echo "   Sıradaki: bash run_mac.sh app" ;;
  *)
    echo "Kullanım: bash run_mac.sh [all|quick|app|slides]"; exit 1 ;;
esac
