#!/bin/bash
set -e
PROJECT="rag_bot"
VENV="$PROJECT/venv"

python3 -m venv $VENV
source $VENV/bin/activate
pip install --upgrade pip
pip install -r $PROJECT/requirements.txt

[[ -f "$PROJECT/.env" ]] || { echo "⚠️  Please create $PROJECT/.env with your API keys."; exit 1; }

EXCEL="$PROJECT/data/Issue Log.xlsx"
if [[ -f "$EXCEL" ]]; then
  python -m data_ingestion.ingest_excel
else
  echo "⚠️  No Excel found at $EXCEL; ingestion skipped."
fi

echo "✅ Setup done. Launch with:  streamlit run $PROJECT/ui/app.py"
