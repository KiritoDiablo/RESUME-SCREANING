import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    job_description_vector = vectors[0].reshape(1, -1)  # Reshaping to 2D array
    resumes_vector = vectors[1:]
    cosine_similarities = cosine_similarity(job_description_vector, resumes_vector).flatten()
    return cosine_similarities

st.title("AI Resume Screening & Ranking System")
job_description = st.text_area("Enter the job description")

st.header("Upload your resumes")
uploaded_files = st.file_uploader("Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.subheader("Ranking Resumes")
    
    resumes = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)
        
    scores = rank_resumes(job_description, resumes)
    
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)
    st.write(results)
