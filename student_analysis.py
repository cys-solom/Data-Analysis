import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.cluster import KMeans
from textblob import TextBlob
import warnings
import logging
from io import BytesIO
from time import sleep

# إزالة أي استدعاء لمكتبات مدفوعة (مثل openai) لأننا سنستخدم بديل مجاني

# التحقق من توفر مكتبة ReportLab لتعريف المتغير REPORTLAB_AVAILABLE
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# إعدادات الصفحة
st.set_page_config(
    page_title="Student Performance & Career Dashboard",
    page_icon=":bar_chart:",
    layout="wide"
)

# CSS مخصص لتحسين المظهر
custom_css = """
<style>
body {
    background: linear-gradient(135deg, #121212, #1e1e1e);
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.header {
    padding: 1rem;
    background-color: #1e1e1e;
    text-align: center;
    border-bottom: 2px solid #66C2A5;
}
.header h1 {
    color: #66C2A5;
    margin: 0;
}
.header p {
    color: #ffffff;
    margin: 0;
    font-size: 16px;
}
.footer {
    text-align: center;
    padding: 0.5rem;
    color: #66C2A5;
    font-size: 14px;
}
.stMetric > div {
    background-color: #1e1e1e;
    border-radius: 10px;
    padding: 1rem;
}
.stButton > button {
    background-color: #66C2A5 !important;
    color: white !important;
    border-radius: 8px;
    font-weight: bold;
}
div.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# شريط التنقل العلوي
header_html = """
<div class="header">
    <h1>Student Performance & Career Dashboard</h1>
    <p>Prepared by - ISlam Ahmed Abdel-Fattah</p>
    <p>Specialization - Cybersecurity and Data Analytics</p>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# تعريف تذييل الصفحة
footer_html = """
<div class="footer">
    <p>Developed by ISlam Ahmed Abdel-Fattah - Cybersecurity and Data Analytics</p>
</div>
"""

# دعم لغتين مع النصوص المناسبة
lang = st.sidebar.selectbox("Choose Language / اختر اللغة", ["English", "العربية"])
if lang == "العربية":
    texts = {
        "title": "لوحة تحكم لأداء الطلاب ومسارهم المهني",
        "upload_data": "رفع بيانات الطلاب",
        "data_preview": "معاينة البيانات",
        "field_of_study": "Field of Study",
        "select_field": "اختر مجال الدراسة",
        "select_gender": "اختر الجنس",
        "select_age": "اختر نطاق العمر",
        "min_job_offers": "أقل عدد من عروض العمل",
        "total_records": "إجمالي عدد الطلاب",
        "avg_high_school_gpa": "متوسط معدل الثانوية",
        "avg_university_gpa": "متوسط معدل الجامعة",
        "univ_above_threshold": "نسبة الطلاب بمعدل الجامعة > 3.0",
        "filters": "المرشحات",
        "dashboard": "لوحة البيانات",
        "basic_plots": "الرسوم البيانية الأساسية",
        "advanced_plots": "الرسوم المتقدمة مع استنتاجات",
        "ml_model": "نموذج التعلم الآلي",
        "gpa_distribution": "توزيع معدل الثانوية",
        "uni_gpa_distribution": "توزيع معدل الجامعة",
        "major_distribution": "توزيع الطلبة حسب التخصص",
        "university_vs_salary": "تأثير معدل الجامعة على الراتب",
        "job_offers_by_field": "عروض العمل حسب المجال",
        "model_select": "اختر نوع النموذج",
        "random_forest": "الغابة العشوائية",
        "logistic_regression": "الانحدار اللوجستي",
        "model_accuracy": "دقة النموذج",
        "confusion_matrix": "مصفوفة الالتباس",
        "roc_curve": "منحنى ROC",
        "feature_importance": "أهمية الميزات",
        "download_data": "تحميل البيانات المعالجة",
        "download_pdf": "تحميل التقرير PDF",
        "info_pdf": "مكتبة ReportLab غير مثبتة. لا يمكن إنشاء تقرير PDF.",
        "new_prediction": "رفع بيانات جديدة للتنبؤ",
        "manual_prediction": "إدخال بيانات للتنبؤ يدويًا",
        "basic_plots_tab": "الرسوم الأساسية",
        "advanced_plots_tab": "الرسوم المتقدمة",
        "about": "حول الموقع",
        "corr_heatmap": "خريطة الارتباط",
        "sunburst_chart": "مخطط Sunburst",
        "projects_boxplot": "مخطط الصندوق للمشاريع",
        "age_bar": "توزيع العمر حسب الجنس",
        "sentiment_analysis": "تحليل المشاعر",
        "salary_prediction": "تنبؤ الراتب",
        "clustering": "التجميع",
        "ai_recommendations": "توصيات الذكاء الاصطناعي"
    }
else:
    texts = {
        "title": "Student Performance & Career Dashboard",
        "upload_data": "Upload Student Data",
        "data_preview": "Data Preview",
        "field_of_study": "Field of Study",
        "select_field": "Select Field of Study",
        "select_gender": "Select Gender",
        "select_age": "Select Age Range",
        "min_job_offers": "Minimum Job Offers",
        "total_records": "Total Number of Students",
        "avg_high_school_gpa": "AVG High School GPA",
        "avg_university_gpa": "AVG University GPA",
        "univ_above_threshold": "Percentage of Students with GPA > 3.0",
        "filters": "Filters",
        "dashboard": "Dashboard",
        "basic_plots": "Basic Plots",
        "advanced_plots": "Advanced Plots with Conclusions",
        "ml_model": "ML Model ( Beta )",
        "gpa_distribution": "High School GPA Distribution",
        "uni_gpa_distribution": "University GPA Distribution",
        "major_distribution": "Student Distribution by Major",
        "university_vs_salary": "University GPA vs. Starting Salary",
        "job_offers_by_field": "Job Offers by Field of Study",
        "model_select": "Select Model Type",
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression",
        "model_accuracy": "Model Accuracy",
        "confusion_matrix": "Confusion Matrix",
        "roc_curve": "ROC Curve",
        "feature_importance": "Feature Importance",
        "download_data": "Download Processed Data",
        "download_pdf": "Download PDF Report",
        "info_pdf": "ReportLab not installed. PDF report unavailable.",
        "new_prediction": "Upload New Data for Prediction",
        "manual_prediction": "Manual Prediction Input ( Beta )",
        "basic_plots_tab": "Basic Plots",
        "advanced_plots_tab": "Advanced Plots",
        "about": "About",
        "corr_heatmap": "Correlation Heatmap",
        "sunburst_chart": "Sunburst Chart",
        "projects_boxplot": "Projects Boxplot",
        "age_bar": "Age Distribution by Gender",
        "sentiment_analysis": "Sentiment Analysis",
        "salary_prediction": "Salary Prediction",
        "clustering": "Clustering",
        "ai_recommendations": "AI Recommendations (Beta)"
    }

# دالة لتحميل البيانات (تستخدم st.cache_data لتخزين النتيجة)
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format!")
            return None
    else:
        data = pd.read_csv("education_career_success.csv")
    return data

uploaded_file = st.file_uploader(f":file_folder: {texts['upload_data']}", type=["csv", "xlsx"])
df = load_data(uploaded_file)

# عرض معاينة البيانات
with st.expander(texts["data_preview"]):
    st.dataframe(df.head())

# ================== الشريط الجانبي للمرشحات ==================
st.sidebar.header(texts["filters"])
study_col = "Field of Study"
job_offers_col = "Job Offers"

if study_col in df.columns:
    selected_fields = st.sidebar.multiselect(texts["select_field"], df[study_col].unique())
    if selected_fields:
        df = df[df[study_col].isin(selected_fields)]
if "Gender" in df.columns:
    selected_gender = st.sidebar.multiselect(texts["select_gender"], df["Gender"].unique())
    if selected_gender:
        df = df[df["Gender"].isin(selected_gender)]
if "Age" in df.columns and pd.api.types.is_numeric_dtype(df["Age"]):
    min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.sidebar.slider(texts["select_age"], min_value=min_age, max_value=max_age, value=(min_age, max_age))
    df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]
if job_offers_col in df.columns and pd.api.types.is_numeric_dtype(df[job_offers_col]):
    min_offers = int(df[job_offers_col].min())
    max_offers = int(df[job_offers_col].max())
    selected_offers = st.sidebar.slider(texts["min_job_offers"], min_offers, max_offers, min_offers)
    df = df[df[job_offers_col] >= selected_offers]

# ================== إنشاء التبويبات بالترتيب الجديد ==================
tabs = st.tabs([
    texts["about"],               # تبويب "حول الموقع" في البداية
    texts["dashboard"],
    texts["basic_plots"],
    texts["advanced_plots"],
    texts["sentiment_analysis"],
    texts["salary_prediction"],
    texts["clustering"],
    texts["ml_model"],            # النماذج التجريبية في النهاية
    texts["manual_prediction"],
    texts["ai_recommendations"]
])

# ---------- تبويب حول الموقع (About) ----------
with tabs[0]:
    st.subheader(texts["about"])
    st.markdown("""
    ### About This Dashboard
    This dashboard provides an interactive analysis of student performance and career outcomes.
    
    **Features:**
    - Interactive filters (Field of Study, Gender, Age, Job Offers)
    - Basic and advanced visualizations with explanations and conclusions
    - ML model training with adjustable hyperparameters (Beta)
    - Manual input for predictions (Beta)
    - Data download and optional PDF report generation
    - AI Recommendations (Beta)
    
    **Developer:**
    Islam Ahmed Abdelfattah - Cyber Security
    
    Enjoy exploring the data and feel free to provide feedback!
    """)
    st.image("https://via.placeholder.com/1500x300?text=Welcome+to+the+Dashboard", use_column_width=True)

# ---------- تبويب لوحة البيانات (Dashboard) ----------
with tabs[1]:
    st.subheader(texts["dashboard"])
    total_students = df.shape[0]
    if "High School GPA" in df.columns and pd.api.types.is_numeric_dtype(df["High School GPA"]):
        avg_hs_gpa = round(df["High School GPA"].mean(), 2)
    else:
        avg_hs_gpa = "N/A"
    if "University - GPA" in df.columns and pd.api.types.is_numeric_dtype(df["University - GPA"]):
        avg_uni_gpa = round(df["University - GPA"].mean(), 2)
        above_threshold = df[df["University - GPA"] > 3.0].shape[0]
        perc_above = round((above_threshold / total_students) * 100, 2) if total_students > 0 else 0
    else:
        avg_uni_gpa = "N/A"
        perc_above = "N/A"
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(texts["total_records"], total_students)
    col2.metric(texts["avg_high_school_gpa"], avg_hs_gpa)
    col3.metric(texts["avg_university_gpa"], avg_uni_gpa)
    col4.metric(texts["univ_above_threshold"], f"{perc_above}%" if isinstance(perc_above, (int, float)) else perc_above)

# ---------- تبويب الرسوم البيانية الأساسية (Basic Plots) ----------
with tabs[2]:
    st.subheader(texts["basic_plots"])
    
    st.markdown("#### " + texts["gpa_distribution"])
    if "High School GPA" in df.columns and pd.api.types.is_numeric_dtype(df["High School GPA"]):
        gpa_hist = px.histogram(
            df, x="High School GPA", nbins=20,
            title="Distribution of High School GPA",
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        st.plotly_chart(gpa_hist, use_container_width=True)
        st.markdown("**Explanation:** This histogram shows the distribution of high school GPA among students.")
    else:
        st.warning("Column 'High School GPA' not found or not numeric.")
    
    st.markdown("#### " + texts["uni_gpa_distribution"])
    if "University - GPA" in df.columns and pd.api.types.is_numeric_dtype(df["University - GPA"]):
        uni_gpa_hist = px.histogram(
            df, x="University - GPA", nbins=20,
            title="Distribution of University GPA",
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        st.plotly_chart(uni_gpa_hist, use_container_width=True)
        st.markdown("**Explanation:** This chart represents the distribution of university GPA among the students.")
    else:
        st.warning("Column 'University - GPA' not found or not numeric.")
    
    st.markdown("#### " + texts["major_distribution"])
    if study_col in df.columns:
        major_dist = df[study_col].value_counts().reset_index()
        major_dist.columns = [study_col, "Count"]
        major_dist_fig = px.bar(
            major_dist, x=study_col, y="Count",
            title="Student Distribution by Major",
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        st.plotly_chart(major_dist_fig, use_container_width=True)
        st.markdown("**Explanation:** This bar chart shows how many students belong to each major.")
    else:
        st.info(f"Column '{study_col}' not found for major distribution chart.")
    
    st.markdown("#### " + texts["university_vs_salary"])
    if "University - GPA" in df.columns and "Starting Salary" in df.columns:
        if pd.api.types.is_numeric_dtype(df["University - GPA"]) and pd.api.types.is_numeric_dtype(df["Starting Salary"]):
            scatter_fig = px.scatter(
                df, x="University - GPA", y="Starting Salary",
                title="Impact of University GPA on Starting Salary",
                template="plotly_dark",
                color=study_col,
                size=job_offers_col,
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            st.plotly_chart(scatter_fig, use_container_width=True)
            st.markdown("**Explanation:** The scatter plot demonstrates how university GPA might influence the starting salary.")
        else:
            st.warning("'University - GPA' or 'Starting Salary' are not numeric.")
    else:
        st.warning("Required columns for the scatter plot not found.")
    
    st.markdown("#### " + texts["job_offers_by_field"])
    try:
        grouped = df.groupby(study_col, as_index=False).sum(numeric_only=True)
        if job_offers_col in grouped.columns:
            bar_fig = px.bar(
                grouped, x=study_col, y=job_offers_col,
                title="Job Offers by Field of Study",
                template="plotly_dark",
                text_auto=True,
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            st.plotly_chart(bar_fig, use_container_width=True)
            st.markdown("**Explanation:** This chart displays the total job offers received per field of study.")
        else:
            st.warning(f"Column '{job_offers_col}' not found in grouped data.")
    except Exception as e:
        st.warning("Error generating job offers chart: " + str(e))

# ---------- تبويب الرسوم البيانية المتقدمة (Advanced Plots) ----------
with tabs[3]:
    st.subheader(texts["advanced_plots"])
    st.markdown("#### " + texts["corr_heatmap"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="Correlation Heatmap",
            template="plotly_dark",
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("**Conclusion:** The heatmap reveals the strength of linear relationships among numeric variables. High correlation between variables may indicate redundancy.")
    else:
        st.warning("Not enough numeric columns for correlation heatmap.")
    
    st.markdown("---")
    st.markdown("#### " + texts["sunburst_chart"])
    if "Gender" in df.columns:
        sunburst_fig = px.sunburst(
            df,
            path=[study_col, "Gender"],
            title="Sunburst Chart: Major & Gender",
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        st.plotly_chart(sunburst_fig, use_container_width=True)
        st.markdown("**Conclusion:** The sunburst chart visualizes the hierarchical distribution of students by major and gender, highlighting dominant groups.")
    else:
        st.info("Column 'Gender' not found for sunburst chart.")
    
    st.markdown("---")
    st.markdown("#### " + texts["projects_boxplot"] + " (Advanced)")
    box_col = None
    if "Projects Completed" in df.columns and pd.api.types.is_numeric_dtype(df["Projects Completed"]):
        box_col = "Projects Completed"
    elif "Starting Salary" in df.columns and pd.api.types.is_numeric_dtype(df["Starting Salary"]):
        box_col = "Starting Salary"
    if box_col:
        box_fig = px.box(
            df, y=box_col, x=study_col,
            title=f"Box Plot of {box_col} by {study_col}",
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        st.plotly_chart(box_fig, use_container_width=True)
        st.markdown("**Conclusion:** The box plot helps in identifying outliers and the spread of the selected variable across different fields.")
    else:
        st.info("No numeric column available for an advanced box plot (Projects Completed or Starting Salary).")
    
    st.markdown("---")
    st.markdown("#### " + texts["age_bar"])
    if "Age" in df.columns and "Gender" in df.columns and pd.api.types.is_numeric_dtype(df["Age"]):
        age_bar_fig = px.histogram(
            df, x="Age", color="Gender", barmode="group",
            title="Age Distribution by Gender",
            template="plotly_dark",
            color_discrete_sequence=px.colors.sequential.Plasma
        )
        st.plotly_chart(age_bar_fig, use_container_width=True)
        st.markdown("**Conclusion:** The histogram displays the age distribution of students with a breakdown by gender, providing insights on age demographics.")
    else:
        st.info("Columns 'Age' or 'Gender' not found or not numeric for age distribution.")

# ---------- تبويب تحليل المشاعر (Sentiment Analysis) ----------
with tabs[4]:
    st.subheader(texts["sentiment_analysis"])
    if "Comments" in df.columns:
        df['Comments'] = df['Comments'].fillna("").astype(str)
        def get_sentiment(text):
            try:
                return TextBlob(text).sentiment.polarity
            except Exception:
                return 0
        df['Sentiment'] = df['Comments'].apply(get_sentiment)
        fig = px.histogram(df, x='Sentiment', title='تحليل المشاعر', template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Column 'Comments' not found for sentiment analysis.")

# ---------- تبويب تنبؤ الراتب (Salary Prediction) ----------
with tabs[5]:
    st.subheader(texts["salary_prediction"])
    # تنبيه بأن هذه الخاصية لا تعمل كما يجب
    if lang == "العربية":
        st.warning("تنبيه: تنبؤ الراتب لا يعمل كما يجب حاليًا.")
    else:
        st.warning("Note: Salary prediction is currently not working as expected.")
    if "University - GPA" in df.columns and "Starting Salary" in df.columns:
        X = df[["University - GPA"]]
        y = df["Starting Salary"]
        model_lr = LinearRegression()
        model_lr.fit(X, y)
        gpa = st.number_input("Enter GPA for Salary Prediction", min_value=0.0, max_value=4.0, value=3.0)
        salary_pred = model_lr.predict([[gpa]])
        st.write(f"Predicted Salary: ${salary_pred[0]:.2f}")
    else:
        st.warning("Required columns for salary prediction not found.")

# ---------- تبويب التجميع (Clustering) ----------
with tabs[6]:
    st.subheader(texts["clustering"])
    if "University - GPA" in df.columns and "Starting Salary" in df.columns:
        kmeans = KMeans(n_clusters=3)
        df['Cluster'] = kmeans.fit_predict(df[['University - GPA', 'Starting Salary']])
        fig = px.scatter(df, x='University - GPA', y='Starting Salary', color='Cluster', template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Required columns for clustering not found.")

# ---------- تبويب نموذج التعلم الآلي (ML Model) - Beta ----------
with tabs[7]:
    st.subheader(texts["ml_model"])
    if lang == "العربية":
        st.info("تنويه: هذه الميزة تجريبية وستتوفر قريباً باللغة العربية.")
    model_choice = st.selectbox(texts["model_select"], options=[texts["random_forest"], texts["logistic_regression"]])
    if model_choice == texts["random_forest"]:
        n_estimators = st.slider("Number of Trees (Random Forest)", min_value=50, max_value=200, value=100)
    else:
        max_iter = st.slider("Max Iterations (Logistic Regression)", min_value=100, max_value=500, value=200)
    
    if st.button("Train Model"):
        if "University - GPA" in df.columns and pd.api.types.is_numeric_dtype(df["University - GPA"]):
            df["Success_ML"] = df["University - GPA"].apply(lambda x: 1 if x > 3.0 else 0)
        else:
            st.error("Numeric column 'University - GPA' not found for ML.")
            st.stop()
        target_col = "Success_ML"
        features = []
        for col in ["High School GPA", "University - GPA", job_offers_col, "Starting Salary"]:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                features.append(col)
        if not features:
            st.error("Required numeric features not found in dataset for ML model.")
        else:
            X = pd.get_dummies(df[features])
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            if model_choice == texts["random_forest"]:
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            else:
                model = LogisticRegression(max_iter=max_iter)
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            st.write(f"{texts['model_accuracy']}: ", round(acc, 3))
            
            unique_labels = np.unique(y_test)
            if len(unique_labels) == 2:
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.2f})'))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Chance', line=dict(dash='dash')))
                fig_roc.update_layout(
                    title=texts["roc_curve"],
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            
            cm = confusion_matrix(y_test, predictions)
            fig_cm = px.imshow(cm, text_auto=True, title=texts["confusion_matrix"], template="plotly_dark")
            st.plotly_chart(fig_cm, use_container_width=True)
            
            if model_choice == texts["random_forest"]:
                importance = model.feature_importances_
                importance_df = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values(by="Importance", ascending=False)
                fig_importance = px.bar(
                    importance_df, x="Feature", y="Importance",
                    title=texts["feature_importance"],
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.sequential.Plasma
                )
                st.plotly_chart(fig_importance, use_container_width=True)

# ---------- تبويب الإدخال اليدوي للتنبؤ (Manual Prediction) - Beta ----------
with tabs[8]:
    st.subheader(texts["manual_prediction"])
    if lang == "العربية":
        st.info("تنويه: هذه الميزة تجريبية وستتوفر قريباً باللغة العربية.")
    with st.form("manual_pred_form"):
        manual_inputs = {}
        for col in ["High School GPA", "University - GPA", job_offers_col, "Starting Salary"]:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                manual_inputs[col] = st.number_input(f"Enter value for {col}", min_value=0.0, max_value=1000.0, value=50.0, step=0.1)
        submitted = st.form_submit_button("Predict")
    if submitted:
        if 'model' in locals() and 'features' in locals() and features:
            manual_df = pd.DataFrame([manual_inputs])
            X_manual = pd.get_dummies(manual_df[features])
            X_manual = X_manual.reindex(columns=X.columns, fill_value=0)
            manual_prediction = model.predict(X_manual)[0]
            st.success(f"Prediction: {'Successful' if manual_prediction == 1 else 'Unsuccessful'}")
        else:
            st.warning("Please train an ML model first.")

# ---------- تبويب توصيات الذكاء الاصطناعي (AI Recommendations) - Beta ----------
with tabs[9]:
    st.subheader(texts["ai_recommendations"])
    if lang == "العربية":
        st.info("تنويه: هذه الميزة تجريبية وستتوفر قريباً باللغة العربية.")
    prompt = st.text_area("Enter your question for AI recommendations")
    if st.button("Get Recommendations"):
        if prompt.strip() == "":
            st.warning("Please enter a prompt for AI recommendations.")
        else:
            with st.spinner("Generating recommendation..."):
                try:
                    from transformers import pipeline
                    # استخدام نموذج GPT-2 مع معلمات محسنة للتوليد
                    generator = pipeline("text-generation", model="gpt2", tokenizer="gpt2")
                    response = generator(prompt, max_length=150, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.95)
                    st.write(response[0]['generated_text'])
                except Exception as e:
                    st.error("Error generating recommendations: " + str(e))

# زر تحميل البيانات المعالجة
csv_data = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label=texts["download_data"],
    data=csv_data,
    file_name="Processed_Student_Data.csv",
    mime="text/csv"
)

# توليد تقرير PDF (اختياري)
if REPORTLAB_AVAILABLE:
    def generate_pdf_report(data):
        buffer = BytesIO()
        try:
            c = canvas.Canvas(buffer, pagesize=letter)
            width, height = letter
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, height - 50, texts["title"])
            c.setFont("Helvetica", 12)
            c.drawString(50, height - 80, f"{texts['total_records']}: {data.shape[0]}")
            if "High School GPA" in data.columns and pd.api.types.is_numeric_dtype(data["High School GPA"]):
                c.drawString(50, height - 100, f"{texts['avg_high_school_gpa']}: {round(data['High School GPA'].mean(),2)}")
            if "Job Offers" in data.columns and pd.api.types.is_numeric_dtype(data["Job Offers"]):
                c.drawString(50, height - 120, f"{texts['min_job_offers']}: {int(data['Job Offers'].sum())}")
            c.showPage()
            c.save()
            buffer.seek(0)
            return buffer
        except Exception as e:
            logging.error(f"Error generating PDF: {e}")
            return None

    pdf_buffer = generate_pdf_report(df)
    if pdf_buffer:
        st.download_button(
            label=texts["download_pdf"],
            data=pdf_buffer,
            file_name="Student_Report.pdf",
            mime="application/pdf"
        )
else:
    st.info(texts["info_pdf"])

# تذييل الصفحة
st.markdown(footer_html, unsafe_allow_html=True)
