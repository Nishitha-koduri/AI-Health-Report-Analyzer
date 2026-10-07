import streamlit as st
import pandas as pd
import numpy as np
import re, io, sqlite3, hashlib
from urllib.parse import quote_plus
from pathlib import Path
from datetime import date, datetime, timedelta

# Optional PDF reader
try:
    # pyrefly: ignore [missing-import]
    import fitz
except Exception:
    fitz = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE = Path(__file__).parent
DATA = BASE / "data"
DB = BASE / "healthcare_app.db"
CBC_FILE = DATA / "CBC_Datasets.csv"
DOCTORS_FILE = DATA / "doctors.csv"
SLOTS_FILE = DATA / "appointment_slots.csv"

st.set_page_config(page_title="AI Health Report Analyzer", page_icon="🩺", layout="wide")

st.markdown("""
<style>
.block-container {padding-top:1.2rem;}
.hero {padding:34px;border-radius:20px;background:linear-gradient(135deg,#eaf4ff,#f5f0ff);border:1px solid #cbd5e1;color:#102a43;}
.hero h1 {color:#102a43 !important;font-size:2.4rem;margin-bottom:10px;}
.hero p {color:#334e68 !important;font-size:1.1rem;line-height:1.6;}
.card {padding:20px;border:1px solid #cbd5e1;border-radius:16px;background:#fff;margin-bottom:14px;color:#102a43;}
.card h3,.card h4,.card p {color:#102a43 !important;}
.flow-card {padding:18px;border-radius:16px;background:#fff;border:1px solid #cbd5e1;min-height:145px;color:#102a43;box-shadow:0 2px 8px rgba(15,23,42,.05);}
.flow-card .num {font-size:1.4rem;font-weight:700;color:#2563eb;}
.flow-card h3 {color:#102a43 !important;margin:8px 0 6px 0;}
.flow-card p {color:#475569 !important;font-size:.95rem;}
.result {padding:14px;border-radius:16px;background:#f8fafc;border:1px solid #e2e8f0;}
.results-wrap .stMetric label {font-size:.82rem !important;}
.results-wrap .stMetric [data-testid="stMetricValue"] {font-size:1.15rem !important; line-height:1.2 !important;}
.results-wrap h1 {font-size:1.7rem !important;}
.results-wrap h2 {font-size:1.35rem !important;}
.results-wrap h3 {font-size:1.1rem !important;}
.small {color:#64748b;font-size:.9rem;}
.admin-banner {padding:14px 18px;border-radius:12px;background:#eef2ff;border:1px solid #c7d2fe;color:#1e1b4b;margin-bottom:16px;}
</style>
""", unsafe_allow_html=True)

# ---------------- DB ----------------
def db():
    con=sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS appointments (id INTEGER PRIMARY KEY AUTOINCREMENT, patient TEXT, doctor_id TEXT, doctor_name TEXT, specialization TEXT, appt_date TEXT, appt_time TEXT, mode TEXT, status TEXT, created_at TEXT)")
    con.execute("CREATE TABLE IF NOT EXISTS notifications (id INTEGER PRIMARY KEY AUTOINCREMENT, patient TEXT, message TEXT, created_at TEXT, read INTEGER DEFAULT 0)")
    con.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at TEXT)")
    con.commit(); return con

def hash_password(password):
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def register_patient(name, username, password):
    con=db()
    try:
        con.execute("INSERT INTO users(name,username,password_hash,created_at) VALUES(?,?,?,?)",(name.strip(),username.strip().lower(),hash_password(password),datetime.now().isoformat(timespec='seconds')))
        con.commit(); return True, 'Registration successful. You can now log in.'
    except sqlite3.IntegrityError:
        return False, 'That username already exists. Please choose another username.'
    finally:
        con.close()

def authenticate_patient(username, password):
    con=db(); row=con.execute("SELECT name,username FROM users WHERE username=? AND password_hash=?",(username.strip().lower(),hash_password(password))).fetchone(); con.close()
    return row

def notify(patient,msg):
    con=db(); con.execute("INSERT INTO notifications(patient,message,created_at) VALUES(?,?,?)",(patient,msg,datetime.now().isoformat(timespec='seconds'))); con.commit(); con.close()

def safe_patient(): return st.session_state.get('patient_name','Guest').strip() or 'Guest'

# ---------------- ML ----------------
@st.cache_resource
def train_anemia_model():
    df=pd.read_csv(CBC_FILE)
    features=['TRBC (in 10^6 /microL)','Hb (in gm/dL)','PCV (%)','MCV (in fL)','MCH (in pg)','MCHC (in gm/dL)','RDW (%)','SBP','DBP']
    X=df[features].apply(pd.to_numeric,errors='coerce')
    X=X.fillna(X.median(numeric_only=True))
    y=df['Severity of anemia (on the basis of Hb)'].astype(str).str.strip()
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
    model=RandomForestClassifier(n_estimators=350,class_weight='balanced',random_state=42,min_samples_leaf=2)
    model.fit(Xtr,ytr)
    acc=accuracy_score(yte,model.predict(Xte))
    return model,features,acc

MODEL, MODEL_FEATURES, MODEL_ACC = train_anemia_model()

def anemia_predict(values):
    row={f: values.get(f, np.nan) for f in MODEL_FEATURES}
    X=pd.DataFrame([row]).apply(pd.to_numeric,errors='coerce')
    # Use training medians for missing fields
    train=pd.read_csv(CBC_FILE)[MODEL_FEATURES].apply(pd.to_numeric,errors='coerce')
    X=X.fillna(train.median(numeric_only=True))
    pred=str(MODEL.predict(X)[0])
    probs=dict(zip(MODEL.classes_, MODEL.predict_proba(X)[0]))
    hb=values.get('Hb (in gm/dL)')
    # Safety/demo override for very low Hb; model remains the primary classifier.
    if hb is not None:
        try:
            hb=float(hb)
            if hb < 7: pred='Severe'
            elif hb < 10 and pred=='Normal': pred='Moderate'
            elif 10 <= hb < 11 and pred=='Normal': pred='Mild'
        except: pass
    return pred, probs

# ---------------- extraction ----------------
def pdf_text(upload):
    if fitz is None: return ''
    try:
        doc=fitz.open(stream=upload.read(),filetype='pdf')
        return '\n'.join(p.get_text() for p in doc)
    except Exception: return ''

def image_ocr(upload):
    # OCR is optional; app never crashes if pytesseract is unavailable.
    try:
        # pyrefly: ignore [missing-import]
        import pytesseract
        from PIL import Image
        img=Image.open(upload)
        return pytesseract.image_to_string(img)
    except Exception:
        return ''

def extract_value(text, patterns):
    num=r'([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d+)?)'
    for p in patterns:
        m=re.search(p+r'[^0-9\r\n]{0,40}'+num, text, re.I)
        if m:
            try:return float(m.group(1).replace(',',''))
            except:return None
    return None

def extract_report(text):
    specs={
      'Hb (in gm/dL)':[r'hemoglobin\s*(?:\(hb\))?',r'\bhgb\b',r'\bhb\b'],
      'TRBC (in 10^6 /microL)':[r'red blood cell(?:s)?(?: count)?',r'\brbc\b'],
      'PCV (%)':[r'packed cell volume',r'hematocrit',r'\bhct\b',r'\bpcv\b'],
      'MCV (in fL)':[r'mean corpuscular volume',r'\bmcv\b'],
      'MCH (in pg)':[r'mean corpuscular hemoglobin(?! concentration)',r'\bmch\b'],
      'MCHC (in gm/dL)':[r'mean corpuscular hemoglobin concentration',r'\bmchc\b'],
      'RDW (%)':[r'red cell distribution width',r'\brdw\b'],
      'SBP':[r'systolic blood pressure',r'\bsbp\b',r'blood pressure'],
      'DBP':[r'diastolic blood pressure',r'\bdbp\b'],
      'TSH':[r'thyroid stimulating hormone',r'\btsh\b'],
      'T3':[r'triiodothyronine',r'\bt3\b'],
      'T4':[r'thyroxine',r'\bt4\b'],
    }
    out={}
    for k,p in specs.items(): out[k]=extract_value(text,p)
    # BP often appears as 120/80; if individual SBP/DBP weren't found, parse it.
    if out['SBP'] is None or out['DBP'] is None:
        m=re.search(r'\b(?:bp|blood pressure)\b[^0-9]{0,20}(\d{2,3})\s*/\s*(\d{2,3})',text,re.I)
        if m:
            out['SBP']=out['SBP'] or float(m.group(1)); out['DBP']=out['DBP'] or float(m.group(2))
    return out

# ---------------- analysis ----------------
def thyroid_status(v):
    tsh,t3,t4=v.get('TSH'),v.get('T3'),v.get('T4')
    if tsh is None and t3 is None and t4 is None: return 'Not enough thyroid values', 'Enter/upload TSH, T3 or T4 to analyze thyroid.'
    # Broad adult reference ranges used only for project demonstration; lab ranges vary.
    if tsh is not None and tsh > 4.5: return 'Possible Hypothyroidism', 'TSH is above the common reference range. A clinician should interpret this with free T4 and symptoms.'
    if tsh is not None and tsh < 0.4: return 'Possible Hyperthyroidism', 'TSH is below the common reference range. A clinician should interpret this with T3/T4 and symptoms.'
    return 'Thyroid Values Within Common Range', 'No obvious TSH abnormality was detected from the entered values.'

def bp_status(sbp,dbp):
    if sbp is None or dbp is None:return 'Not enough BP values','Enter systolic and diastolic pressure.'
    if sbp>=180 or dbp>=120:return 'Very High BP Reading','This reading is very high. If symptoms such as chest pain, shortness of breath, weakness or confusion occur, seek urgent medical care.'
    if sbp>=140 or dbp>=90:return 'High BP Reading','The entered reading is high and should be discussed with a healthcare professional if repeated.'
    if sbp>=130 or dbp>=80:return 'Elevated BP Reading','The entered reading is above the normal range used by this demo.'
    if sbp<90 or dbp<60:return 'Low BP Reading','The entered reading is low; symptoms and repeated readings matter.'
    return 'BP Within Demo Range','The entered reading is within the range used by this project.'

def google_maps_search_url(query):
    return 'https://www.google.com/maps/search/?api=1&query=' + quote_plus(str(query))

def doctors_for(result):
    d=pd.read_csv(DOCTORS_FILE)
    r=result.lower()
    if 'thyroid' in r or 'hypo' in r or 'hyper' in r:
        q=d[d.specialization.str.contains('Endocrin',case=False,na=False)]
        if q.empty:q=d[d.specialization.str.contains('General Medicine|General Physician',case=False,na=False)]
    elif 'blood pressure' in r or 'bp' in r:
        q=d[d.specialization.str.contains('Cardio|General Medicine|Internal',case=False,na=False)]
    else:
        q=d[d.specialization.str.contains('General Medicine|Hematology',case=False,na=False)]
    return q if not q.empty else d[d.status.eq('Active')]

# ---------------- UI ----------------
if 'report_values' not in st.session_state: st.session_state.report_values={}
if 'analysis' not in st.session_state: st.session_state.analysis={}
if 'authenticated' not in st.session_state: st.session_state.authenticated=False
if 'role' not in st.session_state: st.session_state.role=None
if 'patient_username' not in st.session_state: st.session_state.patient_username=None
if 'patient_name' not in st.session_state: st.session_state.patient_name=''
if 'auth_page' not in st.session_state: st.session_state.auth_page='Patient Login'

# Initialize the local database. Patients must register their own account.
db().close()

# ---------------- Landing / authentication ----------------
def show_landing():
    st.markdown('<div class="hero"><h1>🩺 AI-Based Multi-Disease Health Report Analyzer</h1><p>Analyze blood reports, screen anemia, thyroid and blood pressure, get doctor recommendations, find nearby care on Google Maps, book appointments and receive notifications.</p></div>', unsafe_allow_html=True)
    st.markdown('### 🔄 How the system works')
    steps=[
        ('1','📄 Upload Report','Upload your PDF/image report and enter or verify the detected values.'),
        ('2','📊 Analyze Values','Analyze CBC, thyroid and blood-pressure parameters.'),
        ('3','🤖 Predict / Screen','Anemia: Normal/Mild/Moderate/Severe. Thyroid and BP are screened separately.'),
        ('4','👨‍⚕️ Recommend Doctor','Get a suitable medical specialty based on the result.'),
        ('5','🗺️ Google Maps','Find nearby doctors or hospitals using Google Maps.'),
        ('6','📅 Appointment + 🔔','Book an appointment and receive an in-app notification.')]
    cols=st.columns(3)
    for i,(n,title,desc) in enumerate(steps):
        cols[i%3].markdown(f'<div class="flow-card"><div class="num">{n}</div><h3>{title}</h3><p>{desc}</p></div>',unsafe_allow_html=True)
    st.markdown('### 🧪 Conditions covered')
    c1,c2,c3=st.columns(3)
    c1.markdown('<div class="card"><h3>🩸 Anemia</h3><p>CBC values such as Hb, RBC, MCV, MCH, MCHC and RDW.</p></div>',unsafe_allow_html=True)
    c2.markdown('<div class="card"><h3>🦋 Thyroid</h3><p>TSH, T3 and T4 values are screened using reference ranges.</p></div>',unsafe_allow_html=True)
    c3.markdown('<div class="card"><h3>❤️ Blood Pressure</h3><p>Systolic and diastolic readings are categorized for this demo.</p></div>',unsafe_allow_html=True)
    st.markdown('### 🔐 Choose how to continue')
    a,b=st.columns(2)
    with a:
        st.markdown('<div class="card"><h3>👤 Patient</h3><p>Login or create a patient account to upload reports, view results, book appointments and receive notifications.</p></div>',unsafe_allow_html=True)
        if st.button('👤 Patient Login / Register',use_container_width=True,type='primary'):
            st.session_state.auth_page='Patient Login'; st.rerun()
    with b:
        st.markdown('<div class="card"><h3>👨‍💼 Admin</h3><p>Administrator can view doctors, appointments and notifications and update appointment status.</p></div>',unsafe_allow_html=True)
        if st.button('👨‍💼 Admin Login',use_container_width=True):
            st.session_state.auth_page='Admin Login'; st.rerun()

def show_patient_auth():
    st.title('👤 Patient Portal')
    login_tab, register_tab = st.tabs(['🔐 Patient Login','📝 New Patient Registration'])
    with login_tab:
        st.subheader('Login to your patient account')
        u=st.text_input('Username',key='login_username')
        pw=st.text_input('Password',type='password',key='login_password')
        if st.button('🔐 Login',type='primary',use_container_width=True):
            row=authenticate_patient(u,pw)
            if row:
                st.session_state.authenticated=True; st.session_state.role='Patient'; st.session_state.patient_username=row[1]; st.session_state.patient_name=row[0]
                st.session_state.user_location='Warangal, Telangana'; st.success('Login successful.'); st.rerun()
            else: st.error('Invalid username or password.')
    with register_tab:
        st.subheader('Create a patient account')
        name=st.text_input('Full name',key='reg_name')
        u=st.text_input('Create username',key='reg_username')
        pw=st.text_input('Create password',type='password',key='reg_password')
        cpw=st.text_input('Confirm password',type='password',key='reg_confirm')
        if st.button('📝 Register',use_container_width=True):
            if not name.strip() or not u.strip() or not pw:
                st.error('Please fill all fields.')
            elif len(u.strip()) < 3:
                st.error('Username must contain at least 3 characters.')
            elif len(pw) < 6:
                st.error('Password must contain at least 6 characters.')
            elif pw != cpw:
                st.error('Passwords do not match.')
            else:
                ok,msg=register_patient(name,u,pw)
                if ok: st.success(msg)
                else: st.error(msg)
    st.divider()
    if st.button('← Back to Home'):
        st.session_state.auth_page='Home'; st.rerun()

def show_admin_login():
    st.title('👨‍💼 Admin Login')
    st.info('Demo administrator credentials: **admin / admin123**')
    u=st.text_input('Admin username',key='landing_admin_user')
    pw=st.text_input('Admin password',type='password',key='landing_admin_pass')
    if st.button('🔐 Admin Login',type='primary',use_container_width=True):
        if u=='admin' and pw=='admin123':
            st.session_state.authenticated=True; st.session_state.role='Admin'; st.rerun()
        else: st.error('Invalid admin credentials.')
    if st.button('← Back to Home'):
        st.session_state.auth_page='Home'; st.rerun()

# ---------------- Unauthenticated view ----------------
if not st.session_state.authenticated:
    if st.session_state.auth_page=='Patient Login': show_patient_auth()
    elif st.session_state.auth_page=='Admin Login': show_admin_login()
    else: show_landing()
    st.stop()

# ---------------- Authenticated sidebar ----------------
with st.sidebar:
    st.title('🩺 AI Health System')
    if st.session_state.role=='Patient':
        st.success(f'Logged in as: {st.session_state.patient_name}')
        location=st.text_input('Your city / location',st.session_state.get('user_location','Warangal, Telangana'))
        st.session_state.user_location=location
        page=st.radio('Navigate',['Home','Upload & Analyze','Results','Doctor Recommendation','Appointment','Notifications'])
    else:
        st.success('Logged in as: Administrator')
        page='Admin Dashboard'
    st.divider()
    if st.button('🚪 Logout',use_container_width=True):
        st.session_state.authenticated=False; st.session_state.role=None; st.session_state.patient_username=None; st.session_state.patient_name=''; st.session_state.auth_page='Home'; st.rerun()
    st.info('Demo/academic tool. It is not a medical diagnosis.')

if page=='Home':
    st.markdown('<div class="hero"><h1>Welcome, '+str(st.session_state.patient_name)+' 👋</h1><p>Your secure patient dashboard. Start by uploading your medical report.</p></div>',unsafe_allow_html=True)
    st.markdown('### 🔄 Your health analysis flow')
    steps=[('1','📄 Upload Report','Upload PDF/image and verify extracted values.'),('2','📊 Analyze','CBC, thyroid and BP analysis.'),('3','🤖 Results','View anemia, thyroid and BP results.'),('4','👨‍⚕️ Doctor','Get a suitable specialty.'),('5','🗺️ Maps','Find nearby hospitals/doctors.'),('6','📅 Appointment','Book and receive notification.')]
    cols=st.columns(3)
    for i,(n,t,d) in enumerate(steps): cols[i%3].markdown(f'<div class="flow-card"><div class="num">{n}</div><h3>{t}</h3><p>{d}</p></div>',unsafe_allow_html=True)
    st.markdown('### 🚀 Quick actions')
    q1,q2,q3=st.columns(3)
    if q1.button('📄 Upload Report',use_container_width=True): st.session_state.quick_page='Upload & Analyze'; st.rerun()
    if q2.button('📊 View Results',use_container_width=True): st.session_state.quick_page='Results'; st.rerun()
    if q3.button('📅 Book Appointment',use_container_width=True): st.session_state.quick_page='Appointment'; st.rerun()
    if 'quick_page' in st.session_state and st.session_state.quick_page:
        # Streamlit radio remains the authoritative navigation control; show the target as guidance.
        st.info(f'Use **{st.session_state.quick_page}** from the sidebar to continue.')

elif page=='Admin Dashboard':
    st.title('🛠️ Admin Dashboard')
    st.markdown('<div class="admin-banner"><b>Administrator view</b> — manage doctors, appointments, slots and notifications.</div>',unsafe_allow_html=True)
    tab1,tab2,tab3,tab4=st.tabs(['📊 Overview','👨‍⚕️ Doctors','📅 Appointments','🔔 Notifications'])
    docs=pd.read_csv(DOCTORS_FILE)
    with tab1:
        con=db(); apps=pd.read_sql_query('SELECT * FROM appointments ORDER BY id DESC',con); notes=pd.read_sql_query('SELECT * FROM notifications ORDER BY id DESC',con); con.close()
        a,b,c,d=st.columns(4); a.metric('Doctors',len(docs)); b.metric('Active Doctors',int((docs.status.str.lower()=='active').sum())); c.metric('Appointments',len(apps)); d.metric('Notifications',len(notes))
        st.subheader('Recent appointments')
        st.dataframe(apps.tail(10).drop(columns=['created_at'],errors='ignore'),use_container_width=True,hide_index=True)
    with tab2:
        st.dataframe(docs,use_container_width=True,hide_index=True)
        st.caption('Doctor records are loaded from data/doctors.csv. Edit that CSV to permanently change the doctor catalog.')
    with tab3:
        con=db(); apps=pd.read_sql_query('SELECT * FROM appointments ORDER BY id DESC',con); con.close()
        if apps.empty: st.info('No appointments yet.')
        else:
            st.dataframe(apps.drop(columns=['created_at'],errors='ignore'),use_container_width=True,hide_index=True)
            selected=st.selectbox('Appointment to update',apps.id.tolist())
            new_status=st.selectbox('New status',['Confirmed','Completed','Cancelled','Pending'])
            if st.button('Update appointment status'):
                con=db(); con.execute('UPDATE appointments SET status=? WHERE id=?',(new_status,selected)); con.commit(); con.close(); st.success('Appointment status updated.'); st.rerun()
    with tab4:
        con=db(); notes=pd.read_sql_query('SELECT * FROM notifications ORDER BY id DESC',con); con.close()
        if notes.empty: st.info('No notifications yet.')
        else: st.dataframe(notes,use_container_width=True,hide_index=True)

elif page=='Upload & Analyze':
    st.title('📄 Upload & Analyze Medical Report')
    up=st.file_uploader('Upload PDF or image',type=['pdf','png','jpg','jpeg'])
    text=''
    if up:
        if up.type=='application/pdf': text=pdf_text(up)
        else: text=image_ocr(up)
        if text: st.text_area('Extracted report text',text,height=180)
        else: st.warning('Automatic extraction could not read values from this file. You can enter the values manually below.')
    st.subheader('Enter / verify values')
    c=st.columns(4)
    fields=['Hb (in gm/dL)','TRBC (in 10^6 /microL)','PCV (%)','MCV (in fL)','MCH (in pg)','MCHC (in gm/dL)','RDW (%)','SBP','DBP','TSH','T3','T4']
    vals={}; existing=st.session_state.report_values; extracted_all=extract_report(text) if text else {}
    for i,f in enumerate(fields):
        extracted=extracted_all.get(f); default=existing.get(f,extracted)
        vals[f]=c[i%4].number_input(f,value=float(default) if default is not None else 0.0,format='%.2f',key='v_'+f)
    if st.button('🔎 Analyze Report',type='primary',use_container_width=True):
        clean={k:(None if float(v)==0 else float(v)) for k,v in vals.items()}
        anemia,probs=anemia_predict(clean); thyroid,thy_msg=thyroid_status(clean); bp,bp_msg=bp_status(clean.get('SBP'),clean.get('DBP'))
        st.session_state.report_values=clean
        st.session_state.analysis={'anemia':anemia,'probs':probs,'thyroid':thyroid,'thy_msg':thy_msg,'bp':bp,'bp_msg':bp_msg,'created':datetime.now().isoformat(timespec='seconds')}
        notify(safe_patient(),f'Report analyzed: Anemia={anemia}; Thyroid={thyroid}; BP={bp}.')
        st.success('Analysis completed. Open Results.')

elif page=='Results':
    st.title('📊 Results')
    a=st.session_state.analysis
    if not a: st.info('Upload and analyze a report first.'); st.stop()
    c1,c2,c3=st.columns(3); c1.metric('Anemia',a['anemia']); c2.metric('Thyroid',a['thyroid']); c3.metric('Blood Pressure',a['bp'])
    st.subheader('CBC values')
    st.dataframe(pd.DataFrame([{'Parameter':k,'Value':v if v is not None else 'Not entered'} for k,v in st.session_state.report_values.items() if k in MODEL_FEATURES]),use_container_width=True,hide_index=True)
    st.subheader('Anemia model confidence')
    p=pd.DataFrame({'Class':list(a['probs'].keys()),'Probability':[round(x*100,2) for x in a['probs'].values()]})
    st.dataframe(p,use_container_width=True,hide_index=True)
    st.caption(f'Model trained on the included CBC dataset; held-out accuracy during training: {MODEL_ACC*100:.1f}%.')
    st.info('These outputs are for an academic software demonstration and must not be used as a diagnosis or emergency decision.')

elif page=='Doctor Recommendation':
    st.title('👨‍⚕️ Doctor Recommendation')
    a=st.session_state.analysis
    if not a: st.info('Analyze a report first.'); st.stop()
    if a['thyroid'].startswith('Possible'): reason=a['thyroid']
    elif 'High BP' in a['bp'] or 'Very High' in a['bp']: reason=a['bp']
    elif a['anemia']!='Normal': reason=f'{a["anemia"]} Anemia'
    else: reason='General health review'
    st.success(f'Recommended specialty based on result: **{reason}**')
    docs=doctors_for(reason)
    st.dataframe(docs[['doctor_id','doctor_name','designation','specialization','experience_years','hospital_location','consultation_fee','consultation_mode']],use_container_width=True,hide_index=True)
    st.session_state.recommended_reason=reason
    st.subheader('📍 Find a nearby doctor / hospital')
    maps_query=f'{reason} doctor hospital near {st.session_state.get("user_location","")}'
    st.link_button('🗺️ Open Google Maps',google_maps_search_url(maps_query),use_container_width=True)

elif page=='Appointment':
    st.title('📅 Book Appointment')
    docs=pd.read_csv(DOCTORS_FILE); active=docs[docs.status.str.lower().eq('active')]
    default_reason=st.session_state.get('recommended_reason','General health review'); rec=doctors_for(default_reason); choices=rec if not rec.empty else active
    did=st.selectbox('Doctor',choices.doctor_id.tolist(),format_func=lambda x: f"{choices.loc[choices.doctor_id==x,'doctor_name'].iloc[0]} — {choices.loc[choices.doctor_id==x,'specialization'].iloc[0]}")
    drow=choices[choices.doctor_id==did].iloc[0]
    slots=pd.read_csv(SLOTS_FILE); slots=slots[(slots.doctor_id==did)&(slots.slot_status.str.lower().eq('available'))].copy(); slots['date']=pd.to_datetime(slots['date'],errors='coerce'); slots=slots[slots.date>=pd.Timestamp(date.today())]
    if slots.empty:
        appt_date=st.date_input('Appointment date',date.today()+timedelta(days=1)); appt_time=st.time_input('Appointment time')
    else:
        slot_id=st.selectbox('Available slot',slots.slot_id.tolist(),format_func=lambda x: f"{slots.loc[slots.slot_id==x,'date'].iloc[0].date()} — {slots.loc[slots.slot_id==x,'shift'].iloc[0]}")
        s=slots[slots.slot_id==slot_id].iloc[0]; appt_date=s.date.date(); appt_time=datetime.strptime('10:00','%H:%M').time(); st.write(f"Location: **{s.location}** | Shift: **{s['shift']}**")
    maps_query=f'{drow.doctor_name}, {drow.hospital_location}'
    st.link_button('🗺️ Open Hospital in Google Maps',google_maps_search_url(maps_query),use_container_width=True)
    mode=st.selectbox('Mode',['Offline','Online'])
    if st.button('✅ Confirm Appointment',type='primary'):
        con=db(); cur=con.execute("INSERT INTO appointments(patient,doctor_id,doctor_name,specialization,appt_date,appt_time,mode,status,created_at) VALUES(?,?,?,?,?,?,?,?,?)",(safe_patient(),did,drow.doctor_name,drow.specialization,str(appt_date),str(appt_time)[:5],mode,'Confirmed',datetime.now().isoformat(timespec='seconds'))); con.commit(); appt_id=cur.lastrowid; con.close()
        notify(safe_patient(),f'Appointment #{appt_id} confirmed with {drow.doctor_name} on {appt_date} at {str(appt_time)[:5]}.')
        st.success(f'Appointment confirmed! Appointment ID: APPT-{appt_id:05d}')

elif page=='Notifications':
    st.title('🔔 Notifications')
    con=db(); rows=pd.read_sql_query('SELECT * FROM notifications WHERE patient=? ORDER BY id DESC',con,params=(safe_patient(),)); con.close()
    if rows.empty: st.info('No notifications yet.')
    else:
        for _,r in rows.iterrows(): st.markdown(f'<div class="card"><b>🔔</b> {r.message}<div class="small">{r.created_at}</div></div>',unsafe_allow_html=True)
    st.subheader('Appointments')
    con=db(); apps=pd.read_sql_query('SELECT * FROM appointments WHERE patient=? ORDER BY id DESC',con,params=(safe_patient(),)); con.close()
    if apps.empty: st.info('No appointments booked yet.')
    else: st.dataframe(apps.drop(columns=['created_at']),use_container_width=True,hide_index=True)
