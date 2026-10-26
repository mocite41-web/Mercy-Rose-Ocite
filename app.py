import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title='Campus Facility Reservation Monitoring', page_icon='🏫', layout='wide')

REQUIRED = ['Reservation_ID','User_Type','Facility_Type','Days_In_Advance','Participants','Usage_Status','Actual_Attendance','Cancellation_Days_Before']
PREDICTORS = ['User_Type','Facility_Type','Days_In_Advance','Participants','Lead_Time_Category']

@st.cache_data
def clean_data(df):
    df = df.copy().drop_duplicates(subset=['Reservation_ID'])
    df['Days_In_Advance'] = pd.to_numeric(df['Days_In_Advance'], errors='coerce').clip(lower=0)
    df['Participants'] = pd.to_numeric(df['Participants'], errors='coerce')
    df['Actual_Attendance'] = pd.to_numeric(df['Actual_Attendance'], errors='coerce')
    df['Days_In_Advance'] = df['Days_In_Advance'].fillna(df['Days_In_Advance'].median())
    df['Participants'] = df['Participants'].fillna(df['Participants'].median())
    def lead(days):
        if days <= 2: return 'Last Minute'
        if days <= 7: return 'Short Lead'
        if days <= 14: return 'Medium Lead'
        return 'Early Booking'
    df['Lead_Time_Category'] = df['Days_In_Advance'].apply(lead)
    df['Reservation_Outcome'] = df['Usage_Status'].apply(lambda x: 'Completed' if str(x).strip().lower() == 'used' else 'Not Completed')
    df['Attendance_Ratio'] = np.where(df['Participants'].fillna(0).eq(0), 0.0, df['Actual_Attendance'].fillna(0) / df['Participants'])
    return df

@st.cache_resource
def train_model(df):
    X = df[PREDICTORS].copy()
    y = df['Reservation_Outcome'].copy()
    Xenc = pd.get_dummies(X, drop_first=True)
    Xtr, Xte, ytr, yte = train_test_split(Xenc, y, test_size=.20, random_state=42, stratify=y)
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    metrics = {
        'Accuracy': accuracy_score(yte,pred),
        'Precision': precision_score(yte,pred,pos_label='Not Completed',zero_division=0),
        'Recall': recall_score(yte,pred,pos_label='Not Completed',zero_division=0),
        'F1-Score': f1_score(yte,pred,pos_label='Not Completed',zero_division=0),
    }
    return model, Xenc.columns.tolist(), metrics, confusion_matrix(yte,pred), yte, pred

def prepare_input(row, columns):
    x = pd.DataFrame([row])
    x = pd.get_dummies(x, drop_first=False)
    return x.reindex(columns=columns, fill_value=0)

st.sidebar.title('SAD Predictive System')
page = st.sidebar.selectbox('Module', ['Home','Dataset','Data Analysis','Prediction','Dashboard','System Design'])
st.sidebar.caption('Campus Facility Reservation Monitoring')

# Bundled CSV is used by default; users can replace it from Dataset page.
default_path = 'facility_reservations (2).csv'
try:
    base = pd.read_csv(default_path)
except Exception:
    base = None

if 'raw_df' not in st.session_state and base is not None:
    st.session_state.raw_df = base

if page == 'Home':
    st.title('🏫 Campus Facility Reservation Monitoring')
    st.subheader('Cancellation / No-Show Prediction and Decision Support')
    st.write('This Streamlit application converts the midterm data-mining workflow into an interactive information system for facility administrators.')
    c1,c2,c3 = st.columns(3)
    c1.info('**Dataset**\nUpload and validate facility reservation records.')
    c2.info('**Prediction**\nEstimate Completed vs Not Completed outcomes.')
    c3.info('**Dashboard**\nReview KPIs and reservations requiring human follow-up.')
    st.warning('Predictions are decision-support outputs only. Final reservation decisions remain with an authorized human reviewer.')

elif page == 'Dataset':
    st.title('📁 Dataset Upload and Validation')
    up = st.file_uploader('Upload facility reservation CSV', type=['csv'])
    if up is not None:
        try:
            candidate = pd.read_csv(up)
            missing = [c for c in REQUIRED if c not in candidate.columns]
            if missing:
                st.error('Invalid schema. Missing columns: ' + ', '.join(missing))
            elif candidate.empty:
                st.error('The uploaded CSV is empty.')
            else:
                st.session_state.raw_df = candidate
                st.success('CSV uploaded and validated successfully.')
        except Exception as e:
            st.error(f'Unable to read CSV: {e}')
    if 'raw_df' in st.session_state:
        df = st.session_state.raw_df
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            st.error('Current dataset is missing: ' + ', '.join(missing))
        else:
            a,b,c,d = st.columns(4)
            a.metric('Rows', len(df)); b.metric('Columns', len(df.columns)); c.metric('Null Values', int(df.isna().sum().sum())); d.metric('Duplicate IDs', int(df.duplicated('Reservation_ID').sum()))
            st.subheader('Data Preview'); st.dataframe(df.head(20), use_container_width=True)
            st.subheader('Null Summary'); st.dataframe(df.isna().sum().rename('Null_Count').to_frame(), use_container_width=True)
            cleaned = clean_data(df)
            st.download_button('Download Cleaned CSV', cleaned.to_csv(index=False).encode(), 'cleaned_facility_reservations.csv', 'text/csv')
    else:
        st.info('Upload the facility_reservations.csv dataset to begin.')

elif 'raw_df' not in st.session_state:
    st.warning('Please upload a valid dataset in the Dataset module first.')
else:
    raw = st.session_state.raw_df
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        st.error('Dataset schema is invalid. Go to Dataset and upload a valid CSV.')
        st.stop()
    df = clean_data(raw)
    model, feature_cols, metrics, cm, ytest, ypred = train_model(df)

    if page == 'Data Analysis':
        st.title('📊 Exploratory Data Analysis')
        st.subheader('Descriptive Statistics'); st.dataframe(df.describe(include='all').T, use_container_width=True)
        st.subheader('Class Distribution'); st.bar_chart(df['Reservation_Outcome'].value_counts())
        c1,c2 = st.columns(2)
        with c1:
            st.subheader('Reservations by Facility Type'); st.bar_chart(df['Facility_Type'].value_counts())
        with c2:
            st.subheader('Reservations by User Type'); st.bar_chart(df['User_Type'].value_counts())
        st.subheader('Outcome by Lead-Time Category')
        lead = pd.crosstab(df['Lead_Time_Category'], df['Reservation_Outcome'], normalize='index')*100
        st.bar_chart(lead)
        st.subheader('Outcome Percentage by User Category')
        st.dataframe(pd.crosstab(df['User_Type'], df['Reservation_Outcome'], normalize='index')*100, use_container_width=True)

    elif page == 'Prediction':
        st.title('🔮 Reservation Outcome Prediction')
        st.caption('Uses the same pre-reservation predictors as the midterm decision-tree model.')
        c1,c2 = st.columns(2)
        with c1:
            user = st.selectbox('User Type', sorted(df['User_Type'].dropna().astype(str).unique()))
            facility = st.selectbox('Facility Type', sorted(df['Facility_Type'].dropna().astype(str).unique()))
        with c2:
            days = st.number_input('Days in Advance', min_value=0, max_value=365, value=7)
            participants = st.number_input('Participants', min_value=1, max_value=10000, value=30)
        if days <= 2: lead = 'Last Minute'
        elif days <= 7: lead = 'Short Lead'
        elif days <= 14: lead = 'Medium Lead'
        else: lead = 'Early Booking'
        st.write('Lead-Time Category:', lead)
        if st.button('Predict Reservation Outcome', type='primary'):
            row = {'User_Type':user,'Facility_Type':facility,'Days_In_Advance':days,'Participants':participants,'Lead_Time_Category':lead}
            xin = prepare_input(row, feature_cols)
            result = model.predict(xin)[0]
            st.success(f'Predicted Outcome: **{result}**') if result == 'Completed' else st.warning(f'Predicted Outcome: **{result}** — administrator review recommended.')
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(xin)[0]
                st.dataframe(pd.DataFrame({'Outcome':model.classes_, 'Probability':probs}), hide_index=True, use_container_width=True)
            st.info('This prediction supports review; it does not automatically approve, reject, or cancel a reservation.')

    elif page == 'Dashboard':
        st.title('📈 Decision-Support Dashboard')
        not_completed = (df['Reservation_Outcome']=='Not Completed').mean()*100
        top_facility = df['Facility_Type'].value_counts().idxmax()
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Total Reservations', len(df)); c2.metric('Non-Completion Rate', f'{not_completed:.1f}%'); c3.metric('Top Facility', top_facility); c4.metric('Model Accuracy', f"{metrics['Accuracy']:.1%}")
        st.subheader('Model Metrics')
        mc = st.columns(4)
        for col,(name,val) in zip(mc,metrics.items()): col.metric(name, f'{val:.3f}')
        st.subheader('Risk / Review Queue')
        queue = df[(df['Days_In_Advance'] < 3) | (df['User_Type'].isin(['External','Student']))].copy()
        st.dataframe(queue[['Reservation_ID','User_Type','Facility_Type','Days_In_Advance','Participants','Lead_Time_Category','Reservation_Outcome']].head(100), use_container_width=True)
        st.caption('Queue rules highlight reservations for human review; they are not automatic rejection rules.')
        st.subheader('Feature Importance')
        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        st.bar_chart(imp)
        st.subheader('Confusion Matrix'); st.dataframe(pd.DataFrame(cm, index=['Actual Completed','Actual Not Completed'], columns=['Pred Completed','Pred Not Completed']), use_container_width=True)

    elif page == 'System Design':
        st.title('🧩 SAD Integration')
        st.subheader('Actors')
        st.write('**Facility Administrator** — reviews risk information and records final decisions.  **Reservation User** — submits reservation information.  **Predictive System** — validates data and generates decision-support predictions.')
        st.subheader('Functional Requirements')
        reqs = pd.DataFrame([
            ['FR-DM-01','Flag reservations made with fewer than 3 days lead time and support a confirmation follow-up.'],
            ['FR-DM-02','Identify external/student reservations requiring additional administrator review for high-demand facilities.'],
            ['FR-DM-03','Provide a dashboard showing reservation risk and utilization information.'],
            ['FR-DM-04','Allow authorized users to upload and validate reservation CSV data.'],
            ['FR-DM-05','Accept pre-reservation inputs and display a predictive outcome for administrator review.'],
        ], columns=['ID','Requirement'])
        st.dataframe(reqs, hide_index=True, use_container_width=True)
        st.subheader('Non-Functional Requirements')
        st.markdown('- **Usability:** clear navigation and readable results.\n- **Reliability:** validate schema and invalid inputs before analysis.\n- **Security/Privacy:** use authorized or synthetic records and keep final decisions with authorized personnel.')
        st.subheader('Analytical Decision Workflow')
        st.code('Reservation Data → Validate/Clean → Feature Engineering → Decision Tree Model → Risk/Outcome Result → Administrator Review → Final Human Decision', language=None)
        st.subheader('Model Limitation')
        st.write('The model is trained from the uploaded historical dataset and may not generalize to different populations or policies. Predictions must be interpreted with operational context and human judgment.')
