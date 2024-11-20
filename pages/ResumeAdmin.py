import streamlit as st
import nltk
import spacy
import pandas as pd
import base64, random
import time, datetime
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io, random
from streamlit_tags import st_tags
from collections import Counter
from PIL import Image
import PyPDF2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import yt_dlp
import plotly.express as px
import urllib.parse
import re
import os 
from sklearn.metrics import jaccard_score
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


st.set_page_config(
    page_title="Smart Resume Analyzer",
    page_icon="📊",
)

# Define the path for the CSV file
csv_file_path = 'user_data.csv'
admin_csv_file_path = 'admin_data.csv'

@st.cache_resource
def download_stopwords():
    nltk.download('stopwords')
    return nltk.corpus.stopwords.words('english')

# This will be cached, so it's downloaded only once
stopwords = download_stopwords()


@st.cache_resource
def load_spacy_model():
    return spacy.load('en_core_web_sm')

# This will be cached, so the model is loaded only once
nlp = load_spacy_model()


# Caching dataset loading to avoid reloading each time
@st.cache_data
def load_dataset():
    return pd.read_csv("hf://datasets/jacob-hugging-face/job-descriptions/training_data.csv")


# Load your skills dataset
@st.cache_data
def load_skills_dataset(file_path):
    df = pd.read_excel(file_path)
    # Assuming 'Example' column contains the skills we want
    return df['Example'].dropna().tolist()  # This generates the array of skill



def fetch_yt_video(link):
    ydl_opts = {
        'format': 'best',
        'quiet': False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(link, download=False)
        video_title = info_dict.get('title', None)
        return video_title


def get_table_download_link(df, filename, text):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    # href = f'<a href="data:file/csv;base64,{b64}">Download Report</a>'
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to check if the content is a resume
def is_resume(text):
    # Early keyword check for quick filtering
    keywords = [
        "experience", "education", "skills", "certification", 
        "work history", "projects", "references", "summary"
    ]
    
    keyword_count = sum(bool(re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE)) for keyword in keywords)

    if keyword_count < 3:  # Quick exit for non-resumes
        return False

    # Limit text length for NER
    text_to_analyze = text[:2000]  # Limit to the first 2000 characters
    doc = nlp(text_to_analyze)
    
    relevant_entities = sum(1 for ent in doc.ents if ent.label_ in {'PERSON', 'ORG', 'GPE'})

    # Check if we found significant named entities
    return relevant_entities >= 2  # Adjust threshold as needed

def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            print(page)
        text = fake_file_handle.getvalue()

    # close open handles
    converter.close()
    fake_file_handle.close()
    return text



def display_best_matched_job(position_title, company_name, job_description):
    st.markdown(f"""
        <div style="border: 2px solid #f0f0f0; padding: 20px; margin: 20px 0; 
                    border-radius: 10px; box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);">
            <h2 style="color: white; font-size: 24px; margin-bottom: 10px;">{position_title}</h2>
            <h4 style="color: #8e44ad; font-size: 18px; margin-top: 5px;">Company: {company_name}</h4>
            <p style="font-size: 16px; line-height: 1.5; color: white;">{job_description}</p>
        </div>
    """, unsafe_allow_html=True)

# Skill Extraction Functions (same as before)
def extract_skills_tfidf(docs):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names_out()
    return feature_names

def extract_skills_regex(text):
    pattern = r'\b(proficient in|experience with|knowledge of)\b\s([a-zA-Z\s]+)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    skills = [match[1].strip() for match in matches]
    return skills

def extract_skills_ner(text):
    doc = nlp(text)  # Assuming you have an NLP model defined
    skills = [ent.text for ent in doc.ents if ent.label_ in ['PRODUCT', 'ORG', 'LANGUAGE']]
    return skills

# Combined Skill Extraction Function
def extract_skills(docs):
    all_skills = []
    for doc in docs:
        tfidf_skills = extract_skills_tfidf([doc])
        regex_skills = extract_skills_regex(doc)
        ner_skills = extract_skills_ner(doc)
        combined_skills = set(tfidf_skills).union(set(regex_skills)).union(set(ner_skills))
        all_skills.extend(combined_skills)
    return all_skills

# Function to Recommend Skills (Top N skills)
def recommend_skills(documents, top_n=10):
    extracted_skills = extract_skills(documents)
    skill_counter = Counter(extracted_skills)
    most_common_skills = skill_counter.most_common(top_n)
    recommendations = {skill: count for skill, count in most_common_skills}
    return recommendations

def skill_recommendation(df, resume_data, skills_dataset):
    all_skills = load_skills_dataset(skills_path)
    normalized_all_skills = {skill.lower() for skill in all_skills}

    # Combine job descriptions into a single text for matching
    job_descriptions = df['model_response'] + ' ' + df['position_title']
    
    st.subheader("*Skills Recommendation💡*")
    
    if 'skills' in resume_data and isinstance(resume_data['skills'], list):
        user_skills = resume_data['skills']
        normalized_user_skills = {skill.lower() for skill in user_skills}
        keywords = st_tags(label='### Skills that you have',
                           text='See our skills recommendation',
                           value=user_skills, key='1')
    else:
        st.error("No skills found in the resume data. Please provide a valid list of skills.")
        return

    user_skills_string = ' '.join(user_skills)

    # Vectorization and similarity calculation
    vectorizer = TfidfVectorizer()
    job_descriptions_vec = vectorizer.fit_transform(job_descriptions)
    user_skills_vec = vectorizer.transform([user_skills_string])
    similarities = cosine_similarity(user_skills_vec, job_descriptions_vec)

    best_match_index = similarities.argsort()[0][-1]
    best_match_job = df.iloc[best_match_index]
    required_skills = extract_skills([best_match_job['model_response']])

    missing_skills = set(required_skills) - normalized_user_skills
    valid_missing_skills = [skill for skill in missing_skills if skill in normalized_all_skills]

    st.success(f"** Our analysis suggests you are looking for jobs in the field of: {best_match_job['position_title']} **")
    


    display_best_matched_job(best_match_job['position_title'], best_match_job['company_name'], best_match_job['model_response'])
    if valid_missing_skills:
        recommended_keywords = st_tags(
            label='### Recommended missing skills for you.',
            text='Recommended skills based on the job description',
            value=valid_missing_skills, 
            key='2'
        )
        st.markdown(
        '''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to your resume will boost🚀 your chances of getting a Job💼</h4>''',
        unsafe_allow_html=True)
    else:
        st.success("No recommended missing skills found. You are good fit for this role")

    
    
    return best_match_job, valid_missing_skills

def extract_education_from_job_description(job_description):
    education_levels = ["BSc", "MSc", "PhD", "Bachelor", "Master", "Doctorate", "Associate", "Diploma"]
    education_pattern = r'(' + '|'.join(education_levels) + ')'
    match = re.search(education_pattern, job_description, re.IGNORECASE)
    return match.group(1) if match else ""

    
#     return job_skills
def extract_skills_from_job_description(job_description):
    # Process the job description with spaCy
    doc = nlp(job_description)

    # Extract named entities and filter for skills or qualifications
    job_skills = []
    for ent in doc.ents:
        print(f"Entity: {ent.text}, Label: {ent.label_}")  # Debugging statement
        if ent.label_ in ["ORG", "PRODUCT", "WORK_OF_ART", "SKILL"]:
            job_skills.append(ent.text)

    return job_skills

def calculate_resume_score(job_description, candidate_skills, candidate_experience, candidate_education, df):
    # Step 1: Combine job descriptions into a single text for matching
    all_skills = load_skills_dataset(skills_path)
    normalized_all_skills = {skill.lower() for skill in all_skills}

    job_descriptions = df['model_response'] + ' ' + df['position_title']

    # Step 2: Normalize candidate skills
    normalized_candidate_skills = {skill.lower() for skill in candidate_skills}

    # Vectorization and similarity calculation
    vectorizer = TfidfVectorizer()
    job_descriptions_vec = vectorizer.fit_transform(job_descriptions)
    
    # Convert candidate skills to a single string for vectorization
    candidate_skills_string = ' '.join(normalized_candidate_skills)
    candidate_skills_vec = vectorizer.transform([candidate_skills_string])

    # Calculate cosine similarity
    similarities = cosine_similarity(candidate_skills_vec, job_descriptions_vec)

    # Find the best match job index
    best_match_index = similarities.argsort()[0][-1]
    best_match_job = df.iloc[best_match_index]

    # Step 3: Extract required skills from the best matching job description
    required_skills = extract_skills([best_match_job['model_response']])
    required_skills = [skill for skill in required_skills if skill in normalized_all_skills]

    # Step 4: Compare candidate's skills with required skills to find similar skills
    similar_skills = set(normalized_candidate_skills).intersection(set(required_skills))
    skill_match_score = len(similar_skills) / len(set(required_skills)) if required_skills else 0
    
    # Step 5: Extract experience requirement (implement this function as needed)
    # experience_required = extract_experience_from_job_description(best_match_job['model_response'])
    # experience_score = min(candidate_experience / experience_required, 1.0) if experience_required else 0
    
    # Step 6: Compare education level
    education_required = extract_education_from_job_description(best_match_job['model_response'])
    education_match_score = 1 if any(edu in education_required for edu in candidate_education) else 0

    # Step 7: Weighted score calculation
    final_score = (0.9 * skill_match_score) + (0.1 * education_match_score)

    return final_score, similar_skills


# Function to determine candidate level based on general NLP patterns
def determine_candidate_level(experience_text=None, education_text=None, leadership_text=None, skills_text=None):
    # Initialize a score for candidate level determination
    candidate_score = 0

    # If any of the inputs are None or empty, treat them as empty strings
    experience_text = ' '.join(experience_text) if isinstance(experience_text, list) else (experience_text or "")
    education_text = ' '.join(education_text) if isinstance(education_text, list) else (education_text or "")
    leadership_text = ' '.join(leadership_text) if isinstance(leadership_text, list) else (leadership_text or "")
    skills_text = ' '.join(skills_text) if isinstance(skills_text, list) else (skills_text or "")

    # Convert texts to lowercase for easier matching
    experience_text = experience_text.lower()
    education_text = education_text.lower()
    leadership_text = leadership_text.lower()
    skills_text = skills_text.lower()

    # Pattern to find mentions of years of experience
    if experience_text:
        experience_pattern = r"(\d+)\s*(years|yrs|year)\s*(of experience|experience)"
        experience_match = re.search(experience_pattern, experience_text)

        if experience_match:
            years_of_experience = int(experience_match.group(1))  # Extract the number of years
            if years_of_experience >= 5:
                candidate_score += 3  # Senior or Experienced
            elif 1 <= years_of_experience < 5:
                candidate_score += 2  # Intermediate
            elif years_of_experience < 1:
                candidate_score += 1  # Fresher

    # Education Level Check (only if education_text is provided)
    if education_text:
        education_pattern = r"(phd|doctorate|master's|bachelor's|associate's|degree)"
        education_match = re.search(education_pattern, education_text)
        if education_match:
            education_level = education_match.group(0)
            if "phd" in education_level or "doctorate" in education_level:
                candidate_score += 3  # Higher education level
            elif "master" in education_level:
                candidate_score += 2
            elif "bachelor" in education_level:
                candidate_score += 1

    # Detect managerial or leadership roles (only if leadership_text is provided)
    if leadership_text:
        leadership_roles = r"(lead|manager|director|coordinator|supervisor|chief)"
        if re.search(leadership_roles, leadership_text):
            candidate_score += 3  # Higher responsibility roles

    # Skills diversity (only if skills_text is provided)
    if skills_text:
        skills_section = re.findall(r"(\w+)", skills_text)
        unique_skills = set(skills_section)
        if len(unique_skills) > 15:  # A threshold for skill diversity
            candidate_score += 2  # Intermediate to advanced level based on skill diversity

    # Determine the candidate's level based on the accumulated score
    if candidate_score >= 7:
        return "Experienced"
    elif 4 <= candidate_score < 7:
        return "Intermediate"
    elif 1 <= candidate_score < 4:
        return "Fresher"
    else:
        return "Unknown"  # No clear indicators found


def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf">'
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)



def insert_data(name, email, res_score, timestamp, no_of_pages, reco_field, 
                cand_level, skills, recommended_skills):
    # Check if the CSV file exists; if it does, load it into the DataFrame
    if os.path.exists(csv_file_path):
        user_data_df = pd.read_csv(csv_file_path)
    else:
        # Initialize a new DataFrame if the CSV doesn't exist
        user_data_df = pd.DataFrame(columns=["Name", "Email", "Resume Score", 
                                              "Timestamp", "Number of Pages", 
                                              "Recommended Field", "Candidate Level", 
                                              "Skills", "Recommended Skills"])
    
    # Create a new record as a DataFrame
    new_record = pd.DataFrame({
        "Name": [name],
        "Email": [email],
        "Resume Score": [res_score],
        "Timestamp": [timestamp],
        "Number of Pages": [no_of_pages],
        "Recommended Field": [reco_field],
        "Candidate Level": [cand_level],
        "Skills": [skills],
        "Recommended Skills": [recommended_skills]
    })
    
    # Concatenate the new record with the existing DataFrame
    user_data_df = pd.concat([user_data_df, new_record], ignore_index=True)
    
    # Save the updated DataFrame back to the CSV file
    user_data_df.to_csv(csv_file_path, index=False)


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()



# st.title("Smart Resume Analyser")
# st.sidebar.markdown("# Choose User")
# activities = ["Normal User", "Admin"]
# choice = st.sidebar.selectbox("Choose among the given options:", activities)
# Load the image and get base64 string
img_path = './Logo/resume.png'
base64_img = get_base64_image(img_path)
# Display the image in the center with adjusted width using HTML and CSS
st.markdown(
    f"""
    <div style="display: flex; justify-content: center; align-items: center;">
        <img src="data:image/png;base64,{base64_img}" width="800" height="auto">
    </div>
    """,
    unsafe_allow_html=True
)


# import re

# degree_keywords = ['bachelor', 'master', 'phd', 'b\.sc', 'm\.sc', 'm\.tech', 'b\.tech', 'mba', 'm\.a', 'm\.eng', 'b\.eng']

# def extract_degree(education_text_list):
#     # Check if the input is None
#     if education_text_list is None:
#         return []  # Return an empty list if input is None

#     degrees_found = []  # To store extracted degrees

#     # Compile the regex pattern for efficiency
#     degree_pattern = r'\b(?:' + '|'.join(degree_keywords) + r')\b'
    
#     for education_text in education_text_list:
#         # Lowercase the text for uniformity
#         education_text = education_text.lower()
        
#         # Extract degree if present in the text
#         degree_match = re.findall(degree_pattern, education_text)
        
#         if degree_match:
#             # If matches found, return the first relevant degree, capitalize it
#             degrees_found.append(degree_match[0].capitalize())
    
#     return degrees_found  # Return the list of found degrees


# if choice == 'Normal User':
#     # st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Upload your resume, and get smart recommendation based on it."</h4>''',
#     #             unsafe_allow_html=True)
#     pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
#     if pdf_file is not None:
#         # with st.spinner('Uploading your Resume....'):
#         #     time.sleep(4)
        
#         save_image_path = './Uploaded_Resumes/' + pdf_file.name
#         with open(save_image_path, "wb") as f:
#                 f.write(pdf_file.getbuffer())
        

#         # After extracting data from the resume
#         resume_data = ResumeParser(save_image_path).get_extracted_data()
#         resume_text = pdf_reader(save_image_path)

#         # Check if the document is a valid resume
#         if is_resume(resume_text):
#             st.success("This appears to be a valid resume.")

#             # Display the resume PDF file
#             show_pdf(save_image_path)

#             # If resume data is extracted
#             if resume_data:
#                 ## Display Resume Analysis
#                 st.header("*Resume Analysis*")
#                 st.success(f"Hello, {resume_data.get('name', 'Candidate')}")

#                 st.subheader("*Your Basic Information*")
#                 # Use try-except to ensure smooth display even if some data is missing
#                 try:
#                     # Display basic information
#                     st.text(f"Name: {resume_data.get('name', 'Not Available')}")
#                     st.text(f"Email: {resume_data.get('email', 'Not Available')}")
#                     st.text(f"Contact: {resume_data.get('mobile_number', 'Not Available')}")
#                     st.text(f"Resume Pages: {str(resume_data.get('no_of_pages', 'Not Available'))}")

#                     # LinkedIn, GitHub, and Experience - handling missing data
#                     linkedin = resume_data.get('linkedin', '')
#                     github = resume_data.get('github', '')
#                     # Conditionally display LinkedIn and GitHub links
#                     if linkedin:
#                         st.subheader("LinkedIn:")
#                         st.text(linkedin)

#                     if github:
#                         st.subheader("GitHub:")
#                         st.text(github)

                    
#                 except Exception as e:
#                     st.error(f"Some information is missing or could not be retrieved: {e}")

#                 # Display a horizontal rule for better organization
#                 st.markdown("---")


#                 # Safely extract experience, education, and skills
#                 experience = resume_data.get('experience', '')
#                 # Define common degree keywords to filter relevant information
                

#                 education = resume_data.get('degree', '')
#                 education = extract_degree(education)
#                 skills = resume_data.get('skills', '')
                

#                 cand_level = determine_candidate_level(experience, education, "", skills)

#                 # Display the candidate level with appropriate messages
#                 if cand_level == "Fresher":
#                     st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>You are looking for a Fresher level job.</h4>''', unsafe_allow_html=True)
#                 elif cand_level == "Intermediate":
#                     st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''', unsafe_allow_html=True)
#                 elif cand_level == "Experienced":
#                     st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level in your domain!</h4>''', unsafe_allow_html=True)
#                 else:
#                     st.markdown('''<h4 style='text-align: left; color: #000;'>Unable to determine your experience level.</h4>''', unsafe_allow_html=True)



#                 # Example usage within Streamlit
#                 df = load_dataset()  # Caching the dataset load

#                 skills_path = r'C:\Users\User\OneDrive\Desktop\streamlit-react-project\allskillandnonskill\Technology Skills.xlsx'
#                 # Call the main skill recommendation function
#                 best_matched, recommended_skills = skill_recommendation(df, resume_data, skills_path)

               
#                 # Input Job Description
#                 job_description = best_matched['model_response']
#                 # st.title('company demand')
#                 # st.text(job_description)
#                 # Input Candidate Info
#                 candidate_skills_list = skills
#                 candidate_experience = 0
#                 candidate_education = education
#                 # st.title('Skills of user')
#                 # st.text(candidate_skills_list)
#                 # Convert the comma-separated skills into a list
#                 # candidate_skills_list = [skill.strip() for skill in candidate_skills.split(",")]

#                 # Calculate resume score on button click
#                 # if st.button("Calculate Resume Score"):
#                 score, matched_skills = calculate_resume_score(job_description, candidate_skills_list, candidate_experience, candidate_education, df)

#                 # Display the Resume Score with a progress bar
#                 st.subheader("Resume Skill based Ranking")
#                 progress = int(score * 100)  # Convert to percentage
#                 st.progress(progress)  # Display the progress bar

#                 # Show the Resume Score as text
#                 st.markdown(f"  Skill Analysis  :   {score * 100:.2f}%")

#                 # Section for matched skills
#                 if matched_skills:
#                     # st.subheader("Matched Skills:")
#                     # st.write(", ".join(matched_skills))
#                     pass
#                 else:
#                     st.warning("No matched skills found. Please review your skills and job description.")

                

#                 # ## Insert into table
#                 ts = time.time()
#                 cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
#                 cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
#                 timestamp = str(cur_date + '_' + cur_time)
                

#                 st.balloons()
#                 insert_data(resume_data['name'], resume_data['email'], str(score*100), timestamp,
#                             str(resume_data['no_of_pages']), best_matched['position_title'], cand_level, str(resume_data['skills']),
#                             str(recommended_skills))
                
#             else:
#                 st.error('Something went wrong..')
#         else:
#             st.error("The uploaded document does not appear to be a Resume. Please ensure you have uploaded the correct file.")
    
    
# Display the Admin Mode heading
st.markdown('<div class="subheader-style">Admin Mode</div>', unsafe_allow_html=True)
# Initialize session state for logged_in and state_changed if not already set
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'state_changed' not in st.session_state:
    st.session_state.state_changed = False
# Check if the user is already logged in
if st.session_state.logged_in:
    st.success("Welcome Admin")
    # Display user data and admin-specific features
    st.header("User's👨‍💻 Data")
    # Load the real data from the CSV file (for user data)
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
    else:
        st.error("No user data found.")
        df = pd.DataFrame()  # Empty DataFrame as fallback
    # Display the DataFrame in a professional manner
    if not df.empty:
        st.subheader("Candidate Profiles")
        for index, row in df.iterrows():
            with st.expander(f"Profile of {row['Name']}", expanded=False):
                # Create the subject and body content with the admin details
                subject = urllib.parse.quote(f"Opportunity at {st.session_state.admin_post} - {row['Recommended Field']} Role")
                body = urllib.parse.quote(f"""
                Dear {row['Name']},
                I hope this email finds you well. We came across your profile and believe you could be a strong candidate for our {row['Recommended Field']} role at {st.session_state.admin_post}. If you're interested in exploring this opportunity further, please feel free to get in touch.
                Best regards,  
                {st.session_state.admin_username}  
                [Hiring Manager / HR]
                {st.session_state.admin_post}  
                """)
                # Displaying each piece of candidate data in a well-structured format
                st.write(f"**Name:** {row['Name']}")
                st.write(f"**Email:** [Send Email](mailto:{row['Email']}?subject={subject}&body={body})")
                st.write(f"**Score:** {row['Resume Score']}")
                st.write(f"**Timestamp:** {row['Timestamp']}")
                st.write(f"**Number of Pages:** {row['Number of Pages']}")
                st.write(f"**Suitable Position Title:** {row['Recommended Field']}")
                st.write(f"**Candidate Level:** {row['Candidate Level']}")
                st.write(f"**Skills:** {row['Skills']}")
                st.write(f"**Recommended Skills to add  :** {row['Recommended Skills']}")
    # Button to display analysis charts
    if st.button('Show Data Analysis'):
        # Pie chart for Recommended Field Distribution based on count of people
        field_counts = df['Recommended Field'].value_counts().reset_index()
        field_counts.columns = ['Recommended Field', 'Count']
        fig_pie = px.pie(field_counts, values='Count', names='Recommended Field', title='Distribution of People in Recommended Fields')
        st.plotly_chart(fig_pie)
    # Add a button to display the DataFrame
    if st.button('Display DataFrame'):
        # Only show the DataFrame when the button is clicked
        if not df.empty:
            st.dataframe(df)
        else:
            st.write("No data to display.")
    # Button to Log Out with callback function
    if st.button('Logout'):
        st.session_state.logged_in = False
        st.session_state.state_changed = not st.session_state.state_changed  # Toggle state_changed to force rerender
else:
    # Admin Login Page
    st.subheader("Please Log In to Access Admin Features")
    ad_user = st.text_input("Username")
    ad_password = st.text_input("Password", type='password')
    # Load admin credentials from CSV if it exists
    if os.path.exists(admin_csv_file_path):
        admin_df = pd.read_csv(admin_csv_file_path)
    else:
        admin_df = pd.DataFrame(columns=['Username', 'Post', 'Password'])  # Empty DataFrame for admins
    # Button for Login
    if st.button('Login'):
        if not admin_df.empty:
            admin_entry = admin_df[(admin_df['Username'] == ad_user) & (admin_df['Password'] == ad_password)]
            if not admin_entry.empty:
                # Store the admin's username and post in session state
                st.session_state.logged_in = True
                st.session_state.admin_username = ad_user  # Store admin's username
                st.session_state.admin_post = admin_entry['Post'].values[0]  # Store admin's post
                st.session_state.state_changed = not st.session_state.state_changed  # Toggle state_changed to force rerender
                st.success("Login Successful! Redirecting...")
            else:
                st.error("Invalid Username or Password")
        else:
            st.error("No Admin Data Found, Please Sign Up First!")
    # Admin Sign-up Section
    st.subheader("Don't have an account? Sign Up as Admin")
    new_admin_user = st.text_input("New Admin Username")
    new_admin_post = st.text_input("Company Name")
    new_admin_password = st.text_input("New Admin Password", type='password')
    # Button for Sign-up
    if st.button('Sign Up'):
        if new_admin_user and new_admin_post and new_admin_password:
            # Create a new record for the new admin
            new_admin_record = {
                'Username': new_admin_user,
                'Post': new_admin_post,
                'Password': new_admin_password
            }
            # Check if admin CSV exists and append, else create
            if os.path.exists(admin_csv_file_path):
                admin_df = pd.read_csv(admin_csv_file_path)
            else:
                admin_df = pd.DataFrame(columns=['Username', 'Post', 'Password'])
            # Append the new admin record
            admin_df = pd.concat([admin_df, pd.DataFrame([new_admin_record])], ignore_index=True)
            # Save to CSV
            admin_df.to_csv(admin_csv_file_path, index=False)
            st.success("Admin Registered Successfully!")
        else:
            st.warning("Please fill in all the fields for Sign Up!")
# Footer with Professional Styling
st.markdown("<hr style='border: 1px solid #666;'>", unsafe_allow_html=True)
st.markdown(
    """
    <p style='text-align: center; color: #cccccc; font-size: 14px;'>
    Make sure the uploaded files are in the correct format. We support PDF, DOCX for resumes and JPG, PNG for images.
    </p>
    """,
    unsafe_allow_html=True,
)