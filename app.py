import streamlit as st
import os
from openai import PrescriptionOCR
from PIL import Image
import time
from fpdf import FPDF
import requests
import json
from datetime import datetime
import pandas as pd

# Set page config and custom CSS
st.set_page_config(
    page_title="MedScan Pro - Prescription Analysis",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 5px;
            height: 3em;
        }
        .reportBlock {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .warning {
            color: #ff4b4b;
            font-weight: bold;
        }
        .success {
            color: #00c853;
            font-weight: bold;
        }
        h1 {
            color: #1f77b4;
        }
        h2 {
            color: #2c3e50;
        }
    </style>
""", unsafe_allow_html=True)

class PDFReport:
    def __init__(self):
        self.pdf = FPDF()
        self.pdf.add_page()
        self.pdf.set_font('Arial', 'B', 16)
        
    def generate_report(self, medications, validation_results):
        # Add header
        self.pdf.cell(0, 10, 'Prescription Analysis Report', 0, 1, 'C')
        self.pdf.line(10, 30, 200, 30)
        self.pdf.ln(10)
        
        # Add date
        self.pdf.set_font('Arial', '', 10)
        self.pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
        
        # Add medications
        self.pdf.set_font('Arial', 'B', 12)
        self.pdf.cell(0, 10, 'Prescribed Medications:', 0, 1)
        
        self.pdf.set_font('Arial', '', 10)
        for med in medications:
            self.pdf.cell(0, 10, f"Medication: {med['medication']}", 0, 1)
            self.pdf.cell(0, 10, f"Dosage: {med['dosage']}", 0, 1)
            self.pdf.cell(0, 10, f"Frequency: {med['frequency']}", 0, 1)
            self.pdf.ln(5)
            
        return self.pdf

def get_rxcui(medication_name):
    """Get RxCUI for a medication name using RxNav API"""
    base_url = "https://rxnav.nlm.nih.gov/REST/rxcui.json"
    params = {"name": medication_name}
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "idGroup" in data and "rxnormId" in data["idGroup"]:
            return data["idGroup"]["rxnormId"][0]
        return None
    except Exception as e:
        st.warning(f"Error fetching RxCUI for {medication_name}: {str(e)}")
        return None

def get_drug_interactions(rxcuis):
    """Get drug interactions using RxNav Interaction API"""
    if not rxcuis or len(rxcuis) < 2:
        return []
    
    base_url = "https://rxnav.nlm.nih.gov/REST/interaction/list.json"
    rxcui_list = "+".join(rxcuis)
    
    try:
        response = requests.get(f"{base_url}?rxcuis={rxcui_list}")
        response.raise_for_status()
        data = response.json()
        
        interactions = []
        if "fullInteractionTypeGroup" in data:
            for group in data["fullInteractionTypeGroup"]:
                if "fullInteractionType" in group:
                    for interaction in group["fullInteractionType"]:
                        if "interactionPair" in interaction:
                            for pair in interaction["interactionPair"]:
                                interactions.append({
                                    "medications": [
                                        interaction["minConcept"][0]["name"],
                                        interaction["minConcept"][1]["name"]
                                    ],
                                    "severity": pair.get("severity", "N/A"),
                                    "description": pair.get("description", "No description available"),
                                    "source": "RxNav Drug Interaction API"
                                })
        return interactions
    except Exception as e:
        st.warning(f"Error checking drug interactions: {str(e)}")
        return []

def check_drug_interactions(medications):
    """
    Check for drug interactions using RxNav API
    """
    # Get RxCUIs for all medications
    med_rxcuis = {}
    for med in medications:
        med_name = med['medication'].lower()
        rxcui = get_rxcui(med_name)
        if rxcui:
            med_rxcuis[med_name] = rxcui
    
    if len(med_rxcuis) < 2:
        return []
    
    # Get interactions using RxNav API
    interactions = get_drug_interactions(list(med_rxcuis.values()))
    
    # Add severity levels based on the description
    for interaction in interactions:
        if "contraindicated" in interaction["description"].lower():
            interaction["severity"] = "High"
        elif any(word in interaction["description"].lower() for word in ["serious", "severe", "significant"]):
            interaction["severity"] = "Moderate to High"
        else:
            interaction["severity"] = "Moderate"
    
    return interactions

# Update the tab display code to show more detailed interaction information
def display_interaction_details(interaction):
    """Display detailed interaction information in a formatted way"""
    severity_colors = {
        "High": "🔴",
        "Moderate to High": "🟠",
        "Moderate": "🟡",
        "N/A": "⚪"
    }
    
    st.markdown(f"""
    ### {severity_colors[interaction['severity']]} Interaction Detected
    
    **Between**: {' and '.join(interaction['medications'])}
    
    **Severity**: {interaction['severity']}
    
    **Details**: {interaction['description']}
    
    **Source**: {interaction['source']}
    """)
    
    # Add recommendations based on severity
    if interaction['severity'] == "High":
        st.error("🚨 Consult healthcare provider immediately before taking these medications together.")
    elif interaction['severity'] == "Moderate to High":
        st.warning("⚠️ Discuss with healthcare provider before combining these medications.")
    else:
        st.info("ℹ️ Monitor for potential interactions and report any unusual effects to your healthcare provider.")

def create_upload_folder():
    upload_folder = 'static/uploads'
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    return upload_folder

def get_medication_advice(medication_name, dosage, frequency):
    """Generate general advice for medications"""
    advice = [
        f"📌 Take {medication_name} exactly as prescribed.",
        f"⏰ Follow the frequency: {frequency}",
        "🍽️ Unless specified otherwise, take medication with food to avoid stomach upset.",
        "❗ If you experience any unusual side effects, contact your healthcare provider.",
        "📝 Keep a record of when you take your medication.",
        "🔄 Don't stop taking medication without consulting your doctor.",
        "🏥 Store in a cool, dry place away from direct sunlight."
    ]
    return advice

def main():
    # Header with logo and title
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("🏥 MedScan Pro")
        st.subheader("Professional Prescription Analysis System")
    
    # Sidebar with advanced options
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", use_container_width=True)
        st.header("Settings & Information")
        
       
        
        # Advanced options
        st.subheader("Advanced Options")
        enable_interactions = st.checkbox("Check Drug Interactions", value=True)
        enable_pdf = st.checkbox("Generate PDF Report", value=True)
        
        st.markdown("---")
        st.markdown("""
        ### About MedScan Pro
        Professional prescription analysis tool with:
        - OCR Technology
        - Drug Interaction Checking
        - PDF Report Generation
        - AI-Powered Validation
        """)

    # Main content
    uploaded_file = st.file_uploader("📄 Upload Prescription Image", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        tabs = st.tabs(["Analysis Results", "Drug Interactions", "Report Generation"])
        
        with tabs[0]:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(uploaded_file, caption="Uploaded Prescription", use_container_width=True)
                
            with col2:
                with st.spinner("🔍 Analyzing prescription..."):
                    # Process prescription
                    upload_folder = create_upload_folder()
                    temp_path = os.path.join(upload_folder, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    ocr_reader = PrescriptionOCR()
                    results = ocr_reader.read_prescription(temp_path)

                    if isinstance(results, dict):
                        st.success("✅ Analysis Complete!")
                        
                        # Display medications in a clean format
                        for med in results['extracted_medications']:
                            with st.expander(f"💊 {med['medication']}", expanded=True):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Dosage", med['dosage'])
                                with col2:
                                    st.metric("Frequency", med['frequency'])
                                
                                st.markdown("### 📝 Instructions")
                                advice = get_medication_advice(
                                    med['medication'],
                                    med['dosage'],
                                    med['frequency']
                                )
                                for tip in advice:
                                    st.write(tip)

        with tabs[1]:
            if enable_interactions and isinstance(results, dict):
                st.subheader("💊 Drug Interaction Analysis")
                
                with st.spinner("Checking drug interactions..."):
                    interactions = check_drug_interactions(results['extracted_medications'])
                    
                    if interactions:
                        st.warning(f"Found {len(interactions)} potential drug interaction(s)")
                        
                        # Group interactions by severity
                        severity_order = ["High", "Moderate to High", "Moderate", "N/A"]
                        grouped_interactions = {severity: [] for severity in severity_order}
                        
                        for interaction in interactions:
                            grouped_interactions[interaction['severity']].append(interaction)
                        
                        # Display interactions grouped by severity
                        for severity in severity_order:
                            if grouped_interactions[severity]:
                                st.markdown(f"### {severity} Risk Interactions")
                                for interaction in grouped_interactions[severity]:
                                    with st.expander(f"🔄 {' + '.join(interaction['medications'])}", expanded=True):
                                        display_interaction_details(interaction)
                    else:
                        st.success("✅ No significant drug interactions detected in RxNav database")
                        
                    st.info("ℹ️ Note: Always verify interactions with your healthcare provider and pharmacist.")

        with tabs[2]:
            if enable_pdf and isinstance(results, dict):
                st.subheader("📑 Report Generation")
                
                if st.button("Generate PDF Report"):
                    pdf_report = PDFReport()
                    pdf = pdf_report.generate_report(
                        results['extracted_medications'],
                        results['validation_results']
                    )
                    
                    # Save PDF
                    pdf_path = "prescription_report.pdf"
                    pdf.output(pdf_path)
                    
                    with open(pdf_path, "rb") as f:
                        st.download_button(
                            label="Download Report",
                            data=f,
                            file_name="prescription_report.pdf",
                            mime="application/pdf"
                        )

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>🏥 MedScan Pro - Professional Prescription Analysis System</p>
            <p style='color: #666; font-size: 0.8em;'>
                ⚠️ This tool is for educational purposes only. Always consult your healthcare provider.
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 