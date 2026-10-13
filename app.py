import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from xgboost import XGBClassifier
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Student Placement Predictor",
    layout="wide"
)


# --------------------------------------------------
# CUSTOM UI STYLING
# --------------------------------------------------
st.markdown("""
<style>
/* Main application background */
.stApp {
    background: linear-gradient(135deg, #0b1120 0%, #111827 100%);
    color: #f1f5f9;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

/* Main headings */
h1 {
    color: #f8fafc !important;
    font-weight: 750 !important;
    letter-spacing: -0.8px;
}

h2, h3 {
    color: #e2e8f0 !important;
    font-weight: 650 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0b1220;
    border-right: 1px solid #263449;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] p {
    color: #e2e8f0;
}

/* Sidebar navigation */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: #141e30;
    border: 1px solid #263449;
    padding: 10px 14px;
    border-radius: 10px;
    margin-bottom: 8px;
    transition: all 0.2s ease;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #1e3048;
    border-color: #38bdf8;
}

/* Form containers and bordered sections */
div[data-testid="stForm"],
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(17, 30, 49, 0.72);
    border: 1px solid #29394f;
    border-radius: 16px;
    padding: 18px;
}

/* Input fields */
div[data-baseweb="input"] input,
div[data-baseweb="select"] > div,
div[data-baseweb="textarea"] textarea {
    background-color: #182235;
    color: #f8fafc;
    border-radius: 9px;
}

/* Number input buttons */
div[data-testid="stNumberInput"] button {
    background: #24334a;
    color: #f8fafc;
    border-radius: 7px;
}

/* Sliders */
div[data-testid="stSlider"] {
    padding-top: 5px;
    padding-bottom: 5px;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #0284c7, #0d9488);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    font-weight: 650;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    border: 1px solid #67e8f9;
    color: white;
    transform: translateY(-1px);
}

/* Metrics */
div[data-testid="stMetric"] {
    background: #141f32;
    border: 1px solid #2a3b53;
    border-radius: 14px;
    padding: 18px;
}

div[data-testid="stMetricLabel"] {
    color: #94a3b8;
}

div[data-testid="stMetricValue"] {
    color: #67e8f9;
    font-weight: 750;
}

/* Data tables */
div[data-testid="stDataFrame"] {
    border: 1px solid #29394f;
    border-radius: 12px;
    overflow: hidden;
}

/* Status messages */
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* Progress bar */
div[data-testid="stProgressBar"] > div {
    background-color: #263449;
}

div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #0284c7, #2dd4bf);
}

/* Captions */
.stCaption {
    color: #94a3b8 !important;
}

/* Horizontal separators */
hr {
    border-color: #29394f;
}

/* Keep layout responsive on smaller screens */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)


MODEL_DIR = "placement_model"

# --------------------------------------------------
# LOAD SAVED MODELS
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(
        os.path.join(MODEL_DIR, "preprocessor.pkl")
    )

    best_name = joblib.load(
        os.path.join(MODEL_DIR, "best_model_name.pkl")
    )

    features = joblib.load(
        os.path.join(MODEL_DIR, "feature_names.pkl")
    )

    models = {
        "Logistic Regression": joblib.load(
            os.path.join(MODEL_DIR, "logistic_regression.pkl")
        ),
        "Decision Tree": joblib.load(
            os.path.join(MODEL_DIR, "decision_tree.pkl")
        ),
        "Random Forest": joblib.load(
            os.path.join(MODEL_DIR, "random_forest.pkl")
        )
    }

    # Load XGBoost using its native model format
    xgb_model = XGBClassifier()
    xgb_model.load_model(
        os.path.join(MODEL_DIR, "xgboost_model.json")
    )
    models["XGBoost"] = xgb_model

    return preprocessor, models, best_name, features


try:
    preprocessor, models, best_name, feature_names = load_artifacts()
except Exception as e:
    st.error(
        "Could not load the model files. Check that the required "
        "model files are inside the placement_model folder."
    )
    st.exception(e)
    st.stop()

# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------
def predict_student(student, model_name):
    model = models[model_name]

    # Preserve the feature order used during training
    student_df = pd.DataFrame(
        [student],
        columns=feature_names
    )

    transformed = preprocessor.transform(student_df)
    prediction = int(model.predict(transformed)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed)[0]
        classes = list(model.classes_)
        probability = float(probabilities[classes.index(1)])

    return prediction, probability


def generate_prediction_report(
    student,
    selected_model,
    prediction,
    probability,
    comparison_results=None
):
    """Generate a downloadable PDF placement prediction report."""

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch
    )

    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(
        "Student Placement Prediction Report",
        styles["Title"]
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "Generated by the Student Placement Prediction System",
        styles["Normal"]
    ))
    story.append(Spacer(1, 18))

    # Prediction summary
    story.append(Paragraph("Prediction Summary", styles["Heading2"]))

    status = "Placed" if prediction == 1 else "Not Placed"
    probability_text = (
        f"{probability * 100:.2f}%"
        if probability is not None
        else "Not available"
    )

    summary_data = [
        ["Item", "Result"],
        ["Selected ML Model", selected_model],
        ["Predicted Placement Status", status],
        ["Estimated Placement Probability", probability_text]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[2.7 * inch, 3.7 * inch]
    )
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#164E63")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))

    story.append(summary_table)
    story.append(Spacer(1, 20))

    # Student inputs
    story.append(Paragraph("Student Details", styles["Heading2"]))

    input_labels = {
        "CGPA": "CGPA",
        "Internships": "Number of Internships",
        "Projects": "Number of Projects",
        "Workshops/Certifications": "Workshops / Certifications",
        "AptitudeTestScore": "Aptitude Test Score",
        "SoftSkillsRating": "Soft Skills Rating",
        "SSC_Marks": "SSC Marks (%)",
        "HSC_Marks": "HSC Marks (%)",
        "ExtracurricularActivities": "Extracurricular Activities",
        "PlacementTraining": "Placement Training"
    }

    input_data = [["Parameter", "Entered Value"]]

    for key, label in input_labels.items():
        input_data.append([label, str(student.get(key, "N/A"))])

    input_table = Table(
        input_data,
        colWidths=[2.7 * inch, 3.7 * inch]
    )
    input_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#164E63")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.whitesmoke]),
        ("PADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))

    story.append(input_table)
    story.append(Spacer(1, 20))

    # All-model comparison, when available
    if comparison_results:
        story.append(Paragraph(
            "ML Model Comparison", styles["Heading2"]
        ))

        comparison_data = [[
            "Model", "Predicted Status", "Probability"
        ]]

        for result in comparison_results:
            model_probability = result.get(
                "Placement Probability (%)"
            )
            probability_label = (
                f"{model_probability:.2f}%"
                if model_probability is not None
                else "N/A"
            )

            comparison_data.append([
                result["Model"],
                result["Predicted Status"],
                probability_label
            ])

        comparison_table = Table(
            comparison_data,
            colWidths=[2.3 * inch, 2.0 * inch, 2.1 * inch],
            repeatRows=1
        )
        comparison_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#164E63")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.whitesmoke]),
            ("PADDING", (0, 0), (-1, -1), 7),
            ("VALIGN", (0, 0), (-1, -1), "TOP")
        ]))

        story.append(comparison_table)
        story.append(Spacer(1, 18))

    story.append(Paragraph(
        "Important Disclaimer", styles["Heading2"]
    ))
    story.append(Paragraph(
        "This report contains machine-learning estimates based on "
        "the supplied inputs and the training data. The estimated "
        "probability is not a guarantee of actual placement. Employer "
        "eligibility criteria, interview performance, market conditions "
        "and other factors can affect placement outcomes.",
        styles["BodyText"]
    ))

    doc.build(story)
    buffer.seek(0)

    return buffer.getvalue()



# --------------------------------------------------
# SIDEBAR NAVIGATION (DASHBOARD REMOVED)
# --------------------------------------------------
st.sidebar.title("Placement Predictor")

page = st.sidebar.radio(
    "Navigate",
    [
        "Placement Prediction",
        "What-if Analysis",
        "Model Comparison",
        "Feature Importance"
    ]
)

#st.sidebar.markdown("---")
#st.sidebar.caption("Machine Learning Mini Project")

# --------------------------------------------------
# PLACEMENT PREDICTION
# --------------------------------------------------
if page == "Placement Prediction":
    st.title("Student Placement Prediction")
    st.write("Enter the student's details below.")


    with st.form("student_form"):
        col1, col2 = st.columns(2)

        with col1:
            cgpa = st.number_input(
                "CGPA",
                min_value=0.0,
                max_value=10.0,
                value=7.5,
                step=0.1,
                format="%.2f"
            )

            internships = st.number_input(
                "Number of Internships", 0, 20, 1
            )

            projects = st.number_input(
                "Number of Projects", 0, 30, 3
            )

            workshops = st.number_input(
                "Workshops / Certifications", 0, 30, 2
            )

            aptitude = st.number_input(
                "Aptitude Test Score", 
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=0.5
            )

        with col2:
            soft_skills = st.number_input(
                "Soft Skills Rating",
                min_value=0.0,
                max_value=10.0,
                value=4.0,
                step=0.5
            )

            extracurricular = st.selectbox(
                "Extracurricular Activities", ["Yes", "No"]
            )

            training = st.selectbox(
                "Placement Training", ["Yes", "No"]
            )

            ssc = st.number_input(
                "SSC Marks (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=0.5
            )

            hsc = st.number_input(
                "HSC Marks (%)",
                min_value=0.0,
                max_value=100.0,
                value=75.0,
                step=0.5
            )

        selected_model = st.selectbox(
            "Choose ML Algorithm",
            list(models.keys()),
            index=list(models.keys()).index(best_name)
        )

        submitted = st.form_submit_button(
            "Predict Placement",
            type="primary"
        )


    if submitted:
        student = {
            "CGPA": cgpa,
            "Internships": internships,
            "Projects": projects,
            "Workshops/Certifications": workshops,
            "AptitudeTestScore": aptitude,
            "SoftSkillsRating": soft_skills,
            "ExtracurricularActivities": extracurricular,
            "PlacementTraining": training,
            "SSC_Marks": ssc,
            "HSC_Marks": hsc
        }

        try:
            prediction, probability = predict_student(
                student, selected_model
            )

            st.markdown("### Prediction Result")

            if prediction == 1:
                st.success("Prediction: Placed 🎉")
            else:
                st.warning("Prediction: Not Placed")

            if probability is not None:
                st.metric(
                    "Estimated Placement Probability",
                    f"{probability:.1%}"
                )
                st.progress(
                    min(1.0, max(0.0, probability))
                )

            st.caption(
                "This probability is model-estimated and is not a "
                "guarantee of placement."
            )

            
            # --------------------------------------------------
            # COMPARE ALL FOUR MODELS FOR THE SAME STUDENT
            # --------------------------------------------------
            st.divider()
            st.subheader("Compare All ML Models")
            st.write(
                "Each trained model predicts the placement outcome "
                "using the same student details."
            )

            comparison_results = []

            for model_name, model_object in models.items():
                try:
                    transformed_student = preprocessor.transform(
                        pd.DataFrame([student], columns=feature_names)
                    )

                    model_prediction = int(
                        model_object.predict(transformed_student)[0]
                    )

                    model_probability = None

                    if hasattr(model_object, "predict_proba"):
                        probabilities = model_object.predict_proba(
                            transformed_student
                        )[0]

                        classes = list(model_object.classes_)

                        if 1 in classes:
                            model_probability = float(
                                probabilities[classes.index(1)] * 100
                            )

                    comparison_results.append({
                        "Model": model_name,
                        "Predicted Status": (
                            "Placed" if model_prediction == 1
                            else "Not Placed"
                        ),
                        "Placement Probability (%)": (
                            round(model_probability, 2)
                            if model_probability is not None
                            else None
                        )
                    })

                except Exception as model_error:
                    st.warning(
                        f"{model_name} could not make a prediction: "
                        f"{model_error}"
                    )

            if comparison_results:
                results_df = pd.DataFrame(comparison_results)

                # Summary cards
                placed_count = sum(
                    result["Predicted Status"] == "Placed"
                    for result in comparison_results
                )

                not_placed_count = len(comparison_results) - placed_count

                card1, card2, card3 = st.columns(3)

                card1.metric(
                    "Models Compared",
                    len(comparison_results)
                )

                card2.metric(
                    "Predicted Placed",
                    placed_count
                )

                card3.metric(
                    "Predicted Not Placed",
                    not_placed_count
                )

                # Detailed comparison table
                st.markdown("#### Model Prediction Results")

                st.dataframe(
                    results_df,
                    use_container_width=True,
                    hide_index=True
                )

                # Compare estimated probabilities
                probability_df = results_df.dropna(
                    subset=["Placement Probability (%)"]
                )

                if not probability_df.empty:
                    st.markdown("#### Estimated Probability Comparison")

                    fig_compare, ax_compare = plt.subplots(figsize=(9, 4))

                    bars = ax_compare.bar(
                        probability_df["Model"],
                        probability_df["Placement Probability (%)"]
                    )

                    ax_compare.set_ylim(0, 100)
                    ax_compare.set_ylabel("Estimated probability (%)")
                    ax_compare.set_title(
                        "Placement Probability by ML Model"
                    )
                    ax_compare.tick_params(
                        axis="x",
                        rotation=15
                    )
                    ax_compare.grid(axis="y", alpha=0.25)

                    ax_compare.bar_label(
                        bars,
                        fmt="%.1f%%",
                        padding=3
                    )

                    fig_compare.tight_layout()
                    st.pyplot(fig_compare)
                    plt.close(fig_compare)

                st.caption(
                    "Probabilities are estimates from each model and "
                    "may not be directly comparable if the models are "
                    "not equally well calibrated. Different models may "
                    "predict different outcomes for the same student."
                )

                
            # --------------------------------------------------
            # DOWNLOADABLE PDF REPORT
            # --------------------------------------------------
            st.divider()
            st.subheader("Download Prediction Report")

            report_pdf = generate_prediction_report(
                student=student,
                selected_model=selected_model,
                prediction=prediction,
                probability=probability,
                comparison_results=comparison_results
            )

            st.download_button(
                label="⬇️ Download PDF Report",
                data=report_pdf,
                file_name="student_placement_report.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )




            # --------------------------------------------------
            # STUDENT PARAMETER ANALYSIS
            # --------------------------------------------------
            st.divider()
            st.subheader("Student Parameter Analysis")
            st.write(
                "Explore the values entered for this student's "
                "academic performance and experience."
            )

            score_parameters = {
                "CGPA (out of 10)": cgpa * 10,
                "SSC Marks (%)": ssc,
                "HSC Marks (%)": hsc,
                "Aptitude Score (%)": aptitude,
                "Soft Skills (out of 10)": soft_skills * 10
            }

            count_parameters = {
                "Internships": internships,
                "Projects": projects,
                "Workshops / Certifications": workshops
            }

            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.markdown("#### Academic & Skill Scores")

                fig1, ax1 = plt.subplots(figsize=(7, 4))
                bars = ax1.barh(
                    list(score_parameters.keys()),
                    list(score_parameters.values())
                )

                ax1.set_xlim(0, 100)
                ax1.set_xlabel("Score (normalized to 0–100)")
                ax1.set_title("Student Scores")
                ax1.invert_yaxis()
                ax1.grid(axis="x", alpha=0.25)

                for bar, value in zip(
                    bars, score_parameters.values()
                ):
                    ax1.text(
                        min(value + 1, 96),
                        bar.get_y() + bar.get_height() / 2,
                        f"{value:.1f}",
                        va="center"
                    )

                fig1.tight_layout()
                st.pyplot(fig1)
                plt.close(fig1)

            with chart_col2:
                st.markdown("#### Experience & Activities")

                fig2, ax2 = plt.subplots(figsize=(7, 4))
                bars = ax2.barh(
                    list(count_parameters.keys()),
                    list(count_parameters.values())
                )

                ax2.set_xlabel("Number")
                ax2.set_title("Experience and Activities")
                ax2.invert_yaxis()
                ax2.grid(axis="x", alpha=0.25)

                for bar, value in zip(
                    bars, count_parameters.values()
                ):
                    ax2.text(
                        bar.get_width(),
                        bar.get_y() + bar.get_height() / 2,
                        f" {value}",
                        va="center"
                    )

                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)

            # --------------------------------------------------
            # MODEL FEATURE IMPORTANCE
            # --------------------------------------------------
            st.divider()
            st.subheader("Why Did the Model Make This Prediction?")
            st.write(
                "The chart below shows the features the selected "
                "model relies on most strongly. This describes "
                "model-level importance, not a guaranteed explanation "
                "of this individual prediction."
            )

            selected_model_object = models[selected_model]

            try:
                if hasattr(
                    selected_model_object, "feature_importances_"
                ):
                    importance_values = (
                        selected_model_object.feature_importances_
                    )
                    importance_title = (
                        f"Feature Importance — {selected_model}"
                    )

                elif hasattr(selected_model_object, "coef_"):
                    importance_values = np.abs(
                        selected_model_object.coef_[0]
                    )
                    importance_title = (
                        f"Absolute Coefficients — {selected_model}"
                    )

                else:
                    st.info(
                        "Feature importance is not available "
                        "for this model."
                    )
                    importance_values = None

                if importance_values is not None:
                    encoded_names = (
                        preprocessor.get_feature_names_out()
                    )

                    if len(encoded_names) != len(importance_values):
                        st.warning(
                            "Feature names and importance values "
                            "do not match, so the chart cannot be shown."
                        )
                    else:
                        importance_df = pd.DataFrame({
                            "Feature": encoded_names,
                            "Importance": importance_values
                        }).sort_values(
                            "Importance", ascending=False
                        ).head(12)

                        fig3, ax3 = plt.subplots(figsize=(10, 5))
                        ax3.barh(
                            importance_df["Feature"][::-1],
                            importance_df["Importance"][::-1]
                        )
                        ax3.set_title(importance_title)
                        ax3.set_xlabel(
                            "Importance / Absolute Coefficient"
                        )
                        ax3.grid(axis="x", alpha=0.25)
                        fig3.tight_layout()
                        st.pyplot(fig3)
                        plt.close(fig3)

            except Exception as chart_error:
                st.warning(
                    f"Could not display feature importance: "
                    f"{chart_error}"
                )


            # --------------------------------------------------
            # PERSONALIZED IMPROVEMENT SUGGESTIONS
            # --------------------------------------------------
            st.divider()
            st.subheader("Personalized Improvement Suggestions")

            st.write(
                "Based on the details you entered, here are some "
                "practical areas to focus on for placement preparation."
            )

            suggestions = []

            # CGPA
            if cgpa < 7.0:
                suggestions.append({
                    "area": "Academic Performance",
                    "message": (
                        "Your CGPA is below 7.0. Focus on strengthening "
                        "your core subjects, revising concepts regularly, "
                        "and preparing for upcoming examinations."
                    )
                })
            elif cgpa < 8.0:
                suggestions.append({
                    "area": "Academic Performance",
                    "message": (
                        "Keep improving your CGPA where possible and "
                        "maintain consistency in your academic performance."
                    )
                })

            # Internships
            if internships == 0:
                suggestions.append({
                    "area": "Internship Experience",
                    "message": (
                        "You have not entered any internships. Consider "
                        "applying for relevant internships, virtual "
                        "internships, or practical industry projects."
                    )
                })

            # Projects
            if projects < 2:
                suggestions.append({
                    "area": "Projects",
                    "message": (
                        "Consider building at least one or two practical "
                        "projects that demonstrate your skills. Keep the "
                        "code organized and prepare to explain your work."
                    )
                })

            # Workshops and certifications
            if workshops < 2:
                suggestions.append({
                    "area": "Certifications and Workshops",
                    "message": (
                        "Consider completing relevant certifications or "
                        "workshops that match your target roles. Prioritize "
                        "hands-on learning over collecting certificates."
                    )
                })

            # Aptitude
            if aptitude < 60:
                suggestions.append({
                    "area": "Aptitude Preparation",
                    "message": (
                        "Practice quantitative aptitude, logical reasoning, "
                        "and verbal ability regularly. Take timed mock tests "
                        "to identify areas that need improvement."
                    )
                })

            # Soft skills: input scale is 0–10 in this app
            if soft_skills < 5.0:
                suggestions.append({
                    "area": "Soft Skills",
                    "message": (
                        "Practice communication, group discussions, "
                        "presentation skills, and mock interviews. Ask for "
                        "feedback to identify areas for improvement."
                    )
                })

            # SSC and HSC marks
            if ssc < 60:
                suggestions.append({
                    "area": "SSC Academic Record",
                    "message": (
                        "Some employers apply academic eligibility criteria. "
                        "Check each company's requirements and strengthen "
                        "your current skills and interview preparation."
                    )
                })

            if hsc < 60:
                suggestions.append({
                    "area": "HSC Academic Record",
                    "message": (
                        "Some employers apply HSC eligibility criteria. "
                        "Check the eligibility rules for your target "
                        "companies and focus on your current preparation."
                    )
                })

            # Placement training
            if training == "No":
                suggestions.append({
                    "area": "Placement Training",
                    "message": (
                        "Consider joining placement training or following "
                        "a structured preparation plan covering aptitude, "
                        "resumes, technical fundamentals, and interviews."
                    )
                })

            # Extracurricular activities
            if extracurricular == "No":
                suggestions.append({
                    "area": "Communication and Teamwork",
                    "message": (
                        "If opportunities are available, consider joining "
                        "a club, volunteering, or participating in team "
                        "activities to practise collaboration and leadership."
                    )
                })

            if suggestions:
               

                for item in suggestions:
                    with st.container(border=True):
                        st.markdown(f"#### {item['area']}")
                        st.write(item["message"])

            else:
                st.success(
                    "Your entered values meet the basic thresholds used "
                    "by this suggestion system. Keep practising, building "
                    "skills, and preparing for interviews."
                )

            st.caption(
                "These are general preparation suggestions, not guarantees "
                "of placement or proof that changing a particular input "
                "will increase the model's predicted probability."
            )


        except Exception as e:
            st.error("Prediction failed. Check the input values.")
            st.exception(e)

# --------------------------------------------------
# WHAT-IF ANALYSIS
# --------------------------------------------------
elif page == "What-if Analysis":
    st.title("What-if Analysis")
    st.write(
        "Explore how the selected model's estimate changes when "
        "CGPA changes and other inputs remain fixed."
    )

    col1, col2 = st.columns(2)

    with col1:
        base_cgpa = st.number_input(
            "Starting CGPA",
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.1,
            format="%.2f"
        )

        internships = st.number_input(
            "Internships", 0, 20, 1, key="wf_internships"
        )

        projects = st.number_input(
            "Projects", 0, 30, 3, key="wf_projects"
        )

        workshops = st.number_input(
            "Workshops / Certifications",
            0, 30, 2, key="wf_workshops"
        )

        aptitude = st.number_input(
            "Aptitude Score", 
             min_value=0.0,
             max_value=100.0,
             value=75.0,
             step=0.5, 
             key="wf_aptitude"
        )

    with col2:
        soft_skills = st.number_input(
            "Soft Skills Rating",
            min_value=0.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
            key="wf_soft"
        )

        extracurricular = st.selectbox(
            "Extracurricular Activities",
            ["Yes", "No"],
            key="wf_extra"
        )

        training = st.selectbox(
            "Placement Training",
            ["Yes", "No"],
            key="wf_training"
        )

        ssc = st.number_input(
            "SSC Marks (%)",
            min_value=0.0,
            max_value=100.0,
            value=80.0,
            step=0.5,
            key="wf_ssc"
        )

        hsc = st.number_input(
            "HSC Marks (%)",
            min_value=0.0,
            max_value=100.0,
            value=75.0,
            step=0.5,
            key="wf_hsc"
        )


    if st.button("Run What-if Analysis", type="primary"):
        test_cgpas = np.linspace(
            max(0.0, base_cgpa - 2),
            min(10.0, base_cgpa + 2),
            25
        )

        probabilities = []

        for value in test_cgpas:
            student = {
                "CGPA": float(value),
                "Internships": internships,
                "Projects": projects,
                "Workshops/Certifications": workshops,
                "AptitudeTestScore": aptitude,
                "SoftSkillsRating": soft_skills,
                "ExtracurricularActivities": extracurricular,
                "PlacementTraining": training,
                "SSC_Marks": ssc,
                "HSC_Marks": hsc
            }

            _, probability = predict_student(student, best_name)

            probabilities.append(
                np.nan if probability is None else probability * 100
            )

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(test_cgpas, probabilities, marker=".")
        ax.axvline(
            base_cgpa,
            linestyle="--",
            label="Starting CGPA"
        )
        ax.set_xlabel("CGPA")
        ax.set_ylabel("Estimated placement probability (%)")
        ax.set_title(f"What-if Analysis — {best_name}")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Only CGPA changes in this analysis. Results depend on "
            "the model and dataset; they are not causal estimates."
        )

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------
elif page == "Model Comparison":
    st.title("ML Model Comparison")

    # Scores from the evaluation performed in Colab
    comparison = pd.DataFrame([
        {
            "Model": "Logistic Regression",
            "Accuracy": 79.19,
            "Precision": 75.82,
            "Recall": 73.41,
            "F1-Score": 74.60
        },
        {
            "Model": "XGBoost",
            "Accuracy": 79.61,
            "Precision": 77.48,
            "Recall": 71.91,
            "F1-Score": 74.59
        },
        {
            "Model": "Random Forest",
            "Accuracy": 78.98,
            "Precision": 77.11,
            "Recall": 70.40,
            "F1-Score": 73.60
        },
        {
            "Model": "Decision Tree",
            "Accuracy": 75.23,
            "Precision": 70.79,
            "Recall": 68.90,
            "F1-Score": 69.83
        }
    ])

    st.dataframe(
        comparison.style.format({
            "Accuracy": "{:.2f}%",
            "Precision": "{:.2f}%",
            "Recall": "{:.2f}%",
            "F1-Score": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    metric = st.selectbox(
        "Choose metric",
        ["Accuracy", "Precision", "Recall", "F1-Score"]
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(comparison["Model"], comparison[metric])
    ax.set_title(f"Model Comparison — {metric}")
    ax.set_ylabel("Score (%)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "These metrics are from the current test split. "
        "Logistic Regression has the highest F1-score, while "
        "XGBoost has the highest accuracy."
    )


    # --------------------------------------------------
    # CONFUSION MATRIX AND ROC CURVES
    # --------------------------------------------------
    st.divider()
    st.header("Model Evaluation on Held-Out Test Data")

    evaluation_file = "placement_evaluation_results.csv"

    if not os.path.exists(evaluation_file):
        st.warning(
            "Evaluation CSV not found. Place "
            "placement_evaluation_results.csv in the same "
            "folder as app.py."
        )
    else:
        evaluation_data = pd.read_csv(evaluation_file)

        model_options = {
            "Logistic Regression": "logistic_regression",
            "Decision Tree": "decision_tree",
            "Random Forest": "random_forest",
            "XGBoost": "xgboost"
        }

        actual = evaluation_data["Actual"].astype(int)

        selected_eval_model = st.selectbox(
            "Select model to inspect",
            list(model_options.keys()),
            key="evaluation_model"
        )

        model_key = model_options[selected_eval_model]
        prediction_column = f"{model_key}_prediction"
        probability_column = f"{model_key}_probability"

        if (
            prediction_column not in evaluation_data.columns
            or probability_column not in evaluation_data.columns
        ):
            st.error(
                f"Required columns for {selected_eval_model} "
                "are missing from the evaluation CSV."
            )
        else:
            predicted = evaluation_data[
                prediction_column
            ].astype(int)

            # Confusion matrix
            st.subheader("Confusion Matrix")

            cm = confusion_matrix(
                actual,
                predicted,
                labels=[0, 1]
            )

            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))

            image_cm = ax_cm.imshow(cm, interpolation="nearest")

            ax_cm.set_title(
                f"Confusion Matrix — {selected_eval_model}"
            )
            ax_cm.set_xlabel("Predicted Label")
            ax_cm.set_ylabel("Actual Label")

            ax_cm.set_xticks([0, 1])
            ax_cm.set_xticklabels(["Not Placed", "Placed"])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_yticklabels(["Not Placed", "Placed"])

            for i in range(2):
                for j in range(2):
                    ax_cm.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        fontsize=13
                    )

            fig_cm.colorbar(image_cm, ax=ax_cm)
            fig_cm.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

            st.caption(
                "Rows represent actual outcomes; columns represent "
                "predicted outcomes. The diagonal cells are correct "
                "predictions, while the off-diagonal cells are errors."
            )

            # ROC curve and AUC
            st.subheader("ROC Curve")

            probabilities = evaluation_data[
                probability_column
            ].astype(float)

            fpr, tpr, thresholds = roc_curve(
                actual,
                probabilities
            )

            auc_score = roc_auc_score(
                actual,
                probabilities
            )

            fig_roc, ax_roc = plt.subplots(figsize=(7, 5))

            ax_roc.plot(
                fpr,
                tpr,
                label=f"{selected_eval_model} (AUC = {auc_score:.3f})"
            )

            ax_roc.plot(
                [0, 1],
                [0, 1],
                linestyle="--",
                label="Random classifier"
            )

            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title(
                f"ROC Curve — {selected_eval_model}"
            )
            ax_roc.legend(loc="lower right")
            ax_roc.grid(True, alpha=0.25)

            fig_roc.tight_layout()
            st.pyplot(fig_roc)
            plt.close(fig_roc)

            st.metric("ROC AUC Score", f"{auc_score:.3f}")

            st.caption(
                "AUC measures how well the model ranks placed students "
                "above not-placed students across thresholds. Higher is "
                "generally better; 0.5 indicates random ranking."
            )


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------
elif page == "Feature Importance":
    st.title("Feature Importance")
    st.write(
        "Explore which encoded input features the selected model "
        "uses most strongly."
    )

    selected = st.selectbox(
        "Choose model",
        list(models.keys()),
        index=list(models.keys()).index(best_name)
    )

    model = models[selected]

    try:
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
            title = "Model Feature Importance"

        elif hasattr(model, "coef_"):
            values = np.abs(model.coef_[0])
            title = "Absolute Logistic Regression Coefficients"

        else:
            st.info(
                "This model does not expose feature importance "
                "in the expected format."
            )
            st.stop()

        names = preprocessor.get_feature_names_out()

        importance = pd.DataFrame({
            "Feature": names,
            "Importance": values
        }).sort_values(
            "Importance", ascending=False
        ).head(12)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(
            importance["Feature"][::-1],
            importance["Importance"][::-1]
        )
        ax.set_title(f"{title} — {selected}")
        ax.set_xlabel("Importance / Absolute Coefficient")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "One-hot encoded categories may appear as separate "
            "features. Absolute coefficients show strength, not "
            "whether a feature increases or decreases the prediction."
        )

    except Exception as e:
        st.error("Could not calculate feature importance.")
        st.exception(e)
