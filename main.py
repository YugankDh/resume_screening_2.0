import streamlit as st
import html
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
import pdfplumber
from typing import List
import os

load_dotenv()

def load_secret(name):
    try:
        if name in st.secrets:
            os.environ[name] = st.secrets[name]   
    except Exception:
        pass  

load_secret("GROQ_API_KEY")

st.set_page_config(page_title="ResumeMatch AI", page_icon="🎯", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 760px;
    }

    /* Hero */
    .hero {
        text-align: center;
        margin-bottom: 1.75rem;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        color: #6b7280;
        font-size: 1rem;
        max-width: 540px;
        margin: 0 auto;
    }

    /* Info note */
    .info-note {
        background: #f5f3ff;
        border-left: 3px solid #6366f1;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #4b5563;
        margin-bottom: 1.75rem;
        line-height: 1.5;
    }

    /* Score card */
    .score-card {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border-radius: 18px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin: 1rem 0 1.5rem 0;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.25);
        animation: fadeUp 0.4s ease both;
    }
    .score-card .score-value {
        font-size: 3.8rem;
        font-weight: 700;
        line-height: 1;
    }
    .score-card .score-value span {
        font-size: 1.4rem;
        font-weight: 500;
        opacity: 0.85;
        margin-left: 2px;
    }
    .score-card .score-label {
        font-size: 0.75rem;
        opacity: 0.9;
        margin-top: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
    }

    /* Result section cards */
    .result-section {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        animation: fadeUp 0.4s ease both;
    }
    .result-section h3 {
        font-size: 0.95rem;
        font-weight: 600;
        margin: 0 0 0.75rem 0;
        color: #111827;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .result-section p {
        color: #374151;
        font-size: 0.93rem;
        line-height: 1.65;
        margin: 0;
        white-space: pre-wrap;
    }

    /* Bullet list */
    .result-section ul {
        margin: 0;
        padding-left: 1.15rem;
        list-style: disc;
    }
    .result-section li {
        color: #374151;
        font-size: 0.93rem;
        line-height: 1.6;
        margin-bottom: 0.45rem;
        padding-left: 0.15rem;
    }
    .result-section li:last-child {
        margin-bottom: 0;
    }
    .result-section li::marker {
        color: #6366f1;
        font-size: 0.9em;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        padding: 0.65rem 1rem;
        font-weight: 600;
        font-size: 0.95rem;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white;
        border: none;
        transition: opacity 0.15s ease;
    }
    .stButton > button:hover {
        opacity: 0.9;
        color: white;
    }

    /* Inputs */
    .stTextArea textarea, .stFileUploader section {
        border-radius: 10px !important;
    }

    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #e5e7eb;
    }

    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(6px); }
        to   { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_model():
    llm = ChatGroq(
        model="qwen/qwen3.8-27b",
        max_tokens=700
    )
    return llm

model = load_model()



st.markdown("""
<div class="hero">
    <h1>🎯 ResumeMatch AI</h1>
    <p>Upload your resume and paste a job description to get an instant AI-powered match score.</p>
</div>
""", unsafe_allow_html=True)


class Resume(BaseModel):
    summary: str = Field(description="2-3 sentence overall summary of what is present and what is missing")
    missing_skills: List[str] = Field(description="3-5 short points, each one a missing skill")
    suggestion: List[str] = Field(description="3-5 short points, each one a concrete rewriting suggestion")
    match_score: int = Field(description="overall match score from 0 to 100")

pydantic_parser = PydanticOutputParser(pydantic_object=Resume)


def get_score(job_description, resume_details, model, parser):
    template = PromptTemplate(
        template="compare the given job description and the resume details and give your honest review as a senior person in that field and if you have multiple points format them accordingly, it dosent need to be a paragraph \n job_description={job_description} \n resume_details = {resume_details}\n {format_instruction}",
        input_variables=['job_description', 'resume_details'],
        partial_variables={'format_instruction': parser.get_format_instructions()}
    )

    chain = template | model | parser
    result = chain.invoke({'job_description': job_description, 'resume_details': resume_details})
    return result



uploaded_file = st.file_uploader(
    label="Upload your PDF file",
    type=["pdf"]
)

job_description = st.text_area(
    "Paste the Job Description",
    height=250
)



def bullet_list(items):
    return "".join(f"<li>{html.escape(str(item))}</li>" for item in items)



if uploaded_file is not None:

    if uploaded_file.type == "application/pdf":
        if st.button("Analyze"):

            with pdfplumber.open(uploaded_file) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text(layout=True)
                    if page_text:
                        text += page_text + '\n\n'

            with st.spinner("Analyzing your resume..."):
                computed = get_score(job_description, text, model, pydantic_parser)

            st.markdown("---")

            # Score card
            st.markdown(f"""
            <div class="score-card">
                <div class="score-value">{computed.match_score}<span>/100</span></div>
                <div class="score-label">Match Score</div>
            </div>
            """, unsafe_allow_html=True)

            # Summary
            st.markdown(f"""
            <div class="result-section">
                <h3>📋 Summary</h3>
                <p>{html.escape(computed.summary)}</p>
            </div>
            """, unsafe_allow_html=True)

            # Missing skills (list)
            st.markdown(f"""
            <div class="result-section">
                <h3>🧩 Missing Skills</h3>
                <ul>{bullet_list(computed.missing_skills)}</ul>
            </div>
            """, unsafe_allow_html=True)

            # Suggestions (list)
            st.markdown(f"""
            <div class="result-section">
                <h3>✍️ Suggestions</h3>
                <ul>{bullet_list(computed.suggestion)}</ul>
            </div>
            """, unsafe_allow_html=True)