import streamlit as st
import pytesseract
import cv2
from PIL import Image
import numpy as np
import re
import pandas as pd
from wordcloud import WordCloud
import spacy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2 

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


@st.cache_resource
def load_spacy_model():
    return spacy.load('en_core_web_sm')

# This will be cached, so the model is loaded only once
nlp = load_spacy_model()


# # Load the NLP model for NER
# nlp = spacy.load("en_core_web_sm")

# Load skills dataset
skills_path = r'C:\Users\User\OneDrive\Desktop\Project SmartCVAnalyzer\streamlit-react-project\allskillandnonskill\Technology Skills.xlsx'
job_descriptions_path = "hf://datasets/jacob-hugging-face/job-descriptions/training_data.csv"

@st.cache_data
def load_skills_dataset(file_path):
    return pd.read_excel(file_path)

@st.cache_data
def load_job_descriptions():
    return pd.read_csv(job_descriptions_path)

skills_dataset = load_skills_dataset(skills_path)
job_descriptions_df = load_job_descriptions()

# Define functions for skill extraction
def is_resume_text(text):
    required_sections = [
        "experience", "education", "skills", "projects", 
        "summary", "work history", "contact", "certifications"
    ]
    profile_phrases = [
        "highly motivated", "results-oriented", "strong background in", 
        "proven track record", "professional with experience in", 
        "goal-driven", "self-starter", "proficient in"
    ]
    has_section_headers = any(re.search(rf"\b{section}\b", text, re.IGNORECASE) for section in required_sections)
    has_profile_phrases = any(re.search(rf"\b{phrase}\b", text, re.IGNORECASE) for phrase in profile_phrases)
    contains_dates = bool(re.search(r'\b(19|20)\d{2}\b', text))
    contains_bullets = bool(re.search(r'•|\*|-', text))
    match_count = sum([has_section_headers, has_profile_phrases, contains_dates, contains_bullets])
    return match_count >= 3

def extract_skills_tfidf(docs):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(docs)
    return vectorizer.get_feature_names_out()

def extract_skills_regex(text):
    pattern = r'\b(proficient in|experience with|knowledge of)\b\s([a-zA-Z\s]+)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [match[1].strip() for match in matches]

def extract_skills_ner(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ in ['PRODUCT', 'ORG', 'LANGUAGE']]

def extract_skills(docs):
    all_skills = []
    for doc in docs:
        tfidf_skills = extract_skills_tfidf([doc])
        regex_skills = extract_skills_regex(doc)
        ner_skills = extract_skills_ner(doc)
        combined_skills = set(tfidf_skills).union(set(regex_skills)).union(set(ner_skills))
        all_skills.extend(combined_skills)
    return all_skills

def analyze_resume_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    text = pytesseract.image_to_string(thresh)
    skills = extract_skills([text])
    normalized_skills = set(skill.lower() for skill in skills)
    relevant_skills = [skill for skill in normalized_skills if skill in map(str.lower, skills_dataset['Example'].dropna())]
    return text, relevant_skills

def analyze_resume_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            lines = page_text.splitlines()
            formatted_lines = [line.strip() for line in lines if line.strip()]
            page_text = "\n".join(formatted_lines)
            text += page_text + "\n\n"
    text = text.strip()

    skills = extract_skills([text])
    normalized_skills = set(skill.lower() for skill in skills)
    relevant_skills = [skill for skill in normalized_skills if skill in map(str.lower, skills_dataset['Example'].dropna())]

    return text, relevant_skills

import json

# Skill Recommendation based on job descriptions and relevant resume skills
def recommend_job_based_on_skills(relevant_skills):
    job_descriptions = job_descriptions_df['job_description'].tolist()

    # Combine skills from job descriptions into one list for each JD
    job_skills = []
    for jd in job_descriptions:
        skills_in_jd = extract_skills([jd])
        job_skills.append(' '.join(skills_in_jd))

    # Vectorize the resume skills and job description skills using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    resume_vector = vectorizer.fit_transform([' '.join(relevant_skills)])
    job_vector = vectorizer.transform(job_skills)

    # Calculate cosine similarity between resume and job descriptions
    cosine_similarities = cosine_similarity(resume_vector, job_vector).flatten()

    # Get the index of the most similar job description
    most_similar_index = cosine_similarities.argmax()

    # Get the corresponding job title, company name, and additional skills
    recommended_job = job_descriptions_df.iloc[most_similar_index]

    # Check if 'model_response' is a string and parse it if needed
    if isinstance(recommended_job['model_response'], str):
        model_response = json.loads(recommended_job['model_response'])  # Parse the JSON string
    else:
        model_response = recommended_job['model_response']

    # Safely access 'Required Skills' and split by commas
    additional_skills = set(map(str.lower, model_response.get('Required Skills', '').split(', ')))
    missing_skills = list(additional_skills - set(relevant_skills))[:6]


    return recommended_job['company_name'], recommended_job['position_title'], cosine_similarities[most_similar_index], missing_skills, recommended_job['model_response']


# Streamlit UI
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
        font-family: Arial, sans-serif;
    }
    .header {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        text-align: center;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 1.5rem;
        text-align: center;
        margin-bottom: 10px;
    }
    .info {
        margin-bottom: 20px;
        padding: 10px;
        background-color: white;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        color: blue;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='header'><h2>Resume Analyzer with Skill Analysis</h2></div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your resume (image or PDF)", type=['jpg', 'jpeg', 'png', 'pdf'])

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        st.info("PDF uploaded!")
        text, relevant_skills = analyze_resume_from_pdf(uploaded_file)
    else:
        st.info("Image uploaded!")
        image = Image.open(uploaded_file)
        image = np.array(image)
        st.image(image, caption='Uploaded Resume', use_container_width=True)
        text, relevant_skills = analyze_resume_from_image(image)

    if is_resume_text(text):
        st.subheader("Extracted Text:")
        st.markdown(f"<div class='info'><pre>{text}</pre></div>", unsafe_allow_html=True)

        st.subheader("Relevant Skills Found:")
        if relevant_skills:
            st.markdown("<ul>", unsafe_allow_html=True)
            for skill in relevant_skills:
                st.markdown(f"<li>{skill}</li>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)
        else:
            st.write("No relevant skills were detected in the resume.")

        # Recommend the most relevant job based on skills
        company, position, similarity_score, missing_skills, jd = recommend_job_based_on_skills(relevant_skills)
        # Display job recommendation with styling
        st.markdown(f"""
            <div style="border: 2px solid #f0f0f0; padding: 20px; margin: 20px 0; 
                        border-radius: 10px; background: linear-gradient(to right, #4c4c4c, #333333); 
                        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);">
                <h2 style="color: white; font-size: 24px; margin-bottom: 10px;">{position}</h2>
                <h4 style="color: #ffcc00; font-size: 18px; margin-top: 5px;">Company: {company}</h4>
                <p style="font-size: 16px; line-height: 1.5; color: #dcdcdc;">Job Role Requirement:</p>
                <p style="font-size: 14px; line-height: 1.5; color: white;">{jd}</p>
            </div>
        """, unsafe_allow_html=True)
        
        
        # Display the Resume Score with a progress bar
        st.subheader("Resume Skill based Ranking")
        progress = int(similarity_score * 100)  # Convert to percentage
        st.progress(progress)  # Display the progress bar  

        # Explanation under the slider
        st.markdown(f"""
        <div style="text-align: center; font-size: 14px; color: #555; margin-top: 10px;">
            The recommended job matches <b>{similarity_score:.2f}%</b> of your skills!
        </div>
        """, unsafe_allow_html=True)



        # Display skill recommendations professionally
        if missing_skills:
            st.subheader("Skill Recommendations")
            st.markdown("""
                <style>
                    .skill-box {
                        background-color: #f9f9f9;
                        padding: 15px;
                        border: 1px solid #ccc;
                        border-radius: 10px;
                        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                        margin-top: 10px;
                    }
                    .skill-title {
                        color: #2c3e50;
                        font-size: 18px;
                        font-weight: bold;
                        margin-bottom: 10px;
                    }
                    .skill-list {
                        list-style-type: disc;
                        padding-left: 20px;
                        color: #555;
                        font-size: 14px;
                    }
                    .skill-list li {
                        margin-bottom: 5px;
                    }
                </style>
            """, unsafe_allow_html=True)

            st.markdown("""
                <div class="skill-box">
                    <div class="skill-title">Consider Adding the following skills in Your Resume:</div>
                    <ul class="skill-list">
            """, unsafe_allow_html=True)

            for skill in missing_skills:
                st.markdown(f"<li>{skill}</li>", unsafe_allow_html=True)

            st.markdown("</ul></div>", unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background-color: #dff0d8; padding: 15px; border-radius: 10px; 
                            border: 1px solid #d6e9c6; color: #3c763d; margin-top: 10px;">
                    <b>Congratulations!</b> You already have all the skills required for the recommended job.
                </div>
            """, unsafe_allow_html=True)






        # Dataset-based Analysis
        st.subheader("Analysis Based on Skills Dataset")
    
    
        import seaborn as sns
        import matplotlib.pyplot as plt
        import streamlit as st
        
        def get_skill_heatmap(resume_skills, skills_dataset):
            # Filter the dataset for the resume's skills
            relevant_skills_df = skills_dataset[skills_dataset['Example'].str.lower().isin(resume_skills)]
            
            # Create a pivot table with Commodity Title vs Example (skills)
            heatmap_data = relevant_skills_df.pivot_table(index='Commodity Title', columns='Example', aggfunc='size', fill_value=0)
            
            return heatmap_data
        
        # Get skill heatmap data
        heatmap_data = get_skill_heatmap(relevant_skills, skills_dataset)
        
        # Plot heatmap
        st.write("*Skill Heatmap by Commodity Title and Skills*")
        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the size to make it more readable
        sns.heatmap(heatmap_data, annot=True, fmt='d', cmap="YlGnBu", ax=ax, vmin=0, vmax=heatmap_data.values.max())
        ax.set_xlabel('Skills (Example)')
        ax.set_ylabel('Commodity Title')
        st.pyplot(fig)
    
    
    

        # 1. Adjusted Job Title Distribution
        st.write("*Job Title Distribution (Filtered by Relevant Skills)*")
        filtered_job_titles = skills_dataset[skills_dataset['Example'].str.lower().isin(relevant_skills)]['Title'].value_counts().nlargest(10)
        if not filtered_job_titles.empty:
            fig1, ax1 = plt.subplots(figsize=(8, 6))  # Increase figure size
            sns.barplot(y=filtered_job_titles.index, x=filtered_job_titles.values, palette="viridis", ax=ax1)
            ax1.set_xlabel("Count")
            ax1.set_ylabel("Job Title")
            st.pyplot(fig1)
        else:
            st.write("No matching job titles found for extracted skills.")
        
        
         # 2. Hot Technology Distribution
        st.write("*Hot Technology Distribution*")
        hot_tech_counts = skills_dataset['Hot Technology'].value_counts()
        fig2, ax2 = plt.subplots(figsize=(3, 3))  # Set the figure size to 6x6 inches
        hot_tech_counts.plot(kind='pie', autopct='%1.1f%%', labels=["No", "Yes"], colors=["lightblue", "green"], ax=ax2)
        ax2.set_ylabel('')
        st.pyplot(fig2)
    
    
    
        # 3. Relevant Skills by Commodity Title
        st.write("*Relevant Skills by Commodity Title*")
        relevant_skills_df = skills_dataset[skills_dataset['Example'].str.lower().isin(relevant_skills)]
        if not relevant_skills_df.empty:
            commodity_counts = relevant_skills_df['Commodity Title'].value_counts()
            fig3, ax3 = plt.subplots()
            sns.barplot(y=commodity_counts.index, x=commodity_counts.values, palette="magma", ax=ax3)
            ax3.set_xlabel("Count")
            ax3.set_ylabel("Commodity Title")
            st.pyplot(fig3)
        else:
            st.write("No matching commodity titles found for extracted skills.")
    
    
        st.write("*Most Common Skills which can be gained*")
        example_counts = skills_dataset['Example'].value_counts().nlargest(10)
        fig4, ax4 = plt.subplots()
        sns.barplot(y=example_counts.index, x=example_counts.values, palette="coolwarm", ax=ax4)
        ax4.set_xlabel("Count")
        ax4.set_ylabel("Skill Example")
        st.pyplot(fig4)

    else:
        st.warning("The uploaded document does not seem to be a resume. Please try again.")











