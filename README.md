# AI-Based Health Report Analyzer and Doctor Appointment System

Run:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Features: CBC analysis, ML anemia prediction (Normal/Mild/Moderate/Severe), thyroid analysis, BP analysis, doctor recommendation, appointment booking, notifications.

The anemia model is trained from `data/CBC_Datasets.csv` at startup. Doctor data and appointment slots are loaded from CSV files.
