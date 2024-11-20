import streamlit as st
from textblob import TextBlob
import docx2txt
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import language_tool_python
import spacy
from fpdf import FPDF
import re

# Initialize LanguageTool and the transformer model
# tool = language_tool_python.LanguageTool('en-US')


# Streamlit Page Configuration
st.set_page_config(page_title="Cover Letter Sentiment Analyzer", page_icon="✉", layout="centered")

@st.cache_resource
def load_tools():
    # Load and return the LanguageTool and spaCy model
    tool = language_tool_python.LanguageTool('en-US')
    # tool = language_tool_python.LanguageToolPublicAPI('en-US')
    nlp = spacy.load('en_core_web_sm')
    return tool, nlp

# Load the tools using caching
tool, nlp = load_tools()

# Sample cover letters for similarity check
sample_cover_letters = [
    "Dear Hiring Manager,\nI am writing to apply for the Software Engineer position...",
    "To Whom It May Concern,\nI am excited to submit my application for the Data Analyst role...",
    "Hello,\nI am applying for the Marketing Intern position...",
    "Dear [Name],\nI am interested in the Graphic Designer opening at your company..."
]

# # Cache the expensive operations using Streamlit's cache decorator
# @st.cache_resource
# def load_nlp_model():
#     return spacy.load('en_core_web_sm')

# Function to check structural integrity of a cover letter
def has_cover_letter_structure(text):
    salutation = any(salutation in text.lower() for salutation in ["dear", "hello", "to whom it may concern"])
    closing = any(closing in text.lower() for closing in ["sincerely", "regards", "thank you", "best regards"])
    return salutation and closing

# Function to check semantic similarity with sample cover letters
def is_similar_to_cover_letter(text):
    vectorizer = TfidfVectorizer().fit_transform(sample_cover_letters + [text])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors[-1].reshape(1, -1), vectors[:-1])
    return np.max(cosine_sim) > 0.5  # Threshold for similarity

# Function to extract text from DOCX file
def extract_text_from_docx(docx_file):
    return docx2txt.process(docx_file)

# Function to extract text from PDF file
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    return text

# Function to perform sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

# Function to auto-correct and improve the text
def auto_improve_text(text):
    # Step 1: Spell check using TextBlob
    blob = TextBlob(text)
    corrected_sentence = str(blob.correct())

    # Step 2: Grammar check using language_tool_python
    matches = tool.check(corrected_sentence)
    improved_text = language_tool_python.utils.correct(corrected_sentence, matches)

    return improved_text


def is_semantically_similar(text):
    # Sample cover letter phrases to compare against
    sample_cover_letter_texts = [
        "I am writing to express my interest in the position.",
        "Thank you for considering my application.",
        "I believe my skills and experiences align well with the requirements.",
        "I am excited about the opportunity to contribute to your team."
    ]

    # Convert the uploaded text to a spaCy document
    input_doc = nlp(text)

    # Calculate the average similarity score against sample texts
    similarity_scores = []
    for sample in sample_cover_letter_texts:
        sample_doc = nlp(sample)
        similarity_scores.append(input_doc.similarity(sample_doc))

    # Return True if any similarity score is above a threshold, e.g., 0.5
    return any(score > 0.5 for score in similarity_scores)

# Function to validate if text is likely a cover letter
def is_cover_letter(text):
    keywords = ["dear", "sincerely", "cover letter", "application", "position", "resume", "regards"]
    
    # Count keyword occurrences
    keyword_count = sum([1 for word in keywords if word.lower() in text.lower()])
    
    # Check for structural characteristics
    has_salutation = any(greeting in text.lower() for greeting in ["dear", "hi", "hello"])
    has_closing = any(closing in text.lower() for closing in ["sincerely", "regards", "best wishes", "thank you"])

    # Check semantic similarity
    has_semantic_similarity = is_semantically_similar(text)

    # Count valid criteria
    valid_criteria_count = sum([keyword_count > 2, has_salutation, has_closing, has_semantic_similarity])

    # At least two criteria should be valid
    return valid_criteria_count >= 2

class PDF(FPDF):
    def __init__(self, name, orientation='P', unit='mm', format='A4'):
        super().__init__(orientation=orientation, unit=unit, format=format)
        self.name = name  # Store the name for use in the header

    def header(self):
        # Add a header with the name
        self.set_font('DejaVu', 'B', 14)
        self.cell(0, 10, self.name, ln=True, align='C')
        self.ln(5)  # Add space after header
        # Add a horizontal line under the header
        self.set_draw_color(0, 0, 0)  # Black color for the line
        self.line(10, 25, 220, 25)  # Draw a line across the page at y=25

    def footer(self):
        # Add a footer with a page number and top border line (crease)
        self.set_y(-20)
        self.set_draw_color(0, 0, 0)  # Black color for the line
        self.line(10, 240, 220, 240)  # Line just above the footer
        self.set_font('DejaVu', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf_optimized(text, name="Cover Letter", file_name="Improved_Cover_Letter.pdf"):
    # Pass the name to the PDF class with orientation, unit, and format
    pdf = PDF(name, orientation='P', unit='mm', format=(230, 250))

    # Add the fonts for regular, bold, and italic
    pdf.add_font('DejaVu', 'B', 'C:/Users/User/OneDrive/Desktop/Project SmartCVAnalyzer/streamlit-react-project/pages/DejaVuSans-Bold.ttf', uni=True)
    pdf.add_font('DejaVu', '', 'C:/Users/User/OneDrive/Desktop/Project SmartCVAnalyzer/streamlit-react-project/pages/DejaVuSans.ttf', uni=True)
    pdf.add_font('DejaVu', 'I', 'C:/Users/User/OneDrive/Desktop/Project SmartCVAnalyzer/streamlit-react-project/pages/DejaVuSans-Oblique.ttf', uni=True)

    # Add a new page to the PDF
    pdf.add_page()

    # Set font for the body text
    pdf.set_font('DejaVu', '', 10)

    # Adjust line height for more compact text
    line_height = 6  # Reduced line height for a more standard look

    # Add text to the PDF
    lines = text.split('\n')
    for line in lines:
        pdf.cell(0, line_height, line, ln=True)

    # Save the PDF with the specified file name
    pdf.output(file_name)
    return file_name



# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1F4E79;
        text-align: center;
        margin-bottom: 10px;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #5D6D7E;
        text-align: center;
        margin-top: -15px;
        margin-bottom: 30px;
    }
    .content-frame {
        border: 1px solid #DDDDDD;
        border-radius: 5px;
        padding: 20px;
        background-color: #FAFAFA;
        margin-bottom: 20px;
        color:black;
    }
    .footer {
        font-size: 0.9rem;
        text-align: center;
        color: #95A5A6;
        margin-top: 50px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Main Title and Subtitle
st.markdown("<div class='main-title'>Cover Letter Sentiment Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Analyze and enhance your cover letter for a professional presentation.</div>", unsafe_allow_html=True)

# Cover Letter Upload Section
cover_letter_file = st.file_uploader("📄 Upload your Cover Letter (DOCX or PDF)", type=["docx", "pdf"], help="Upload a .docx or .pdf file for sentiment analysis.")

if cover_letter_file is not None:
    # Extract text from cover letter based on file type
    if cover_letter_file.type == "application/pdf":
        cover_letter_text = extract_text_from_pdf(cover_letter_file)
    else:
        cover_letter_text = extract_text_from_docx(cover_letter_file)

    # Check if the extracted text is likely a cover letter
    if is_cover_letter(cover_letter_text):
        st.subheader("📄 Extracted Cover Letter Text")
        st.markdown(f"<div class='content-frame'>{cover_letter_text}</div>", unsafe_allow_html=True)

        # Sentiment analysis on the cover letter
        polarity, subjectivity = analyze_sentiment(cover_letter_text)
        st.subheader("🧠 Sentiment Analysis Results")

        # Display progress bars for polarity and subjectivity
        st.markdown("### Sentiment Polarity")
        st.progress((polarity + 1) / 2)  # Normalize polarity (-1 to 1) to range (0 to 1)
        polarity_label = "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
        st.write(f"Polarity Score: {polarity:.2f} ({polarity_label})")

        st.markdown("### Sentiment Subjectivity")
        st.progress(subjectivity)  # Subjectivity is already in range (0 to 1)
        subjectivity_label = "Highly Subjective" if subjectivity > 0.5 else "Objective"
        st.write(f"Subjectivity Score: {subjectivity:.2f} ({subjectivity_label})")

        # Calculate Cover Letter Score
        # Define new weights and scoring logic
        polarity_weight = 0.75  # Give more weight to positivity in a cover letter
        subjectivity_weight = 0.25  # Less weight to subjectivity

        # Adjust subjectivity: the closer to 0.5, the better
        subjectivity_adjustment = 1 - abs(subjectivity - 0.5)

        # Calculate cover letter score
        cover_letter_score = (polarity_weight * (polarity + 1) / 2) + (subjectivity_weight * subjectivity_adjustment)
        # Display Cover Letter Score
        st.markdown("### Cover Letter Score")
        st.progress(cover_letter_score)
        st.write(f"**Score:** {cover_letter_score:.2f} / 1.00")
        # Provide professional feedback based on polarity and subjectivity
        st.write("---")
        if polarity > 0:
            st.success("The cover letter has a positive tone, making it likely to leave a good impression.")
        elif polarity < 0:
            st.warning("The cover letter has a negative tone. Consider revising it to sound more positive or neutral.")
        else:
            st.info("The cover letter has a neutral tone. Adding some enthusiasm can help it stand out.")

        if subjectivity > 0.6:
            st.warning("The letter is highly subjective. Make sure the content is professional and focused on the job requirements.")

        # Generate auto-improved version of the cover letter
        improved_text = auto_improve_text(cover_letter_text)
        st.subheader("🔧 Auto-Improved Version of Your Cover Letter")
        st.markdown(f"<div class='content-frame'>{improved_text}</div>", unsafe_allow_html=True)

        # Option to download the improved version as a PDF
        if st.button("Download Improved Cover Letter as PDF"):
            pdf_filename = create_pdf_optimized(improved_text)
            with open(pdf_filename, "rb") as pdf_file:
                st.download_button("Download PDF", data=pdf_file, file_name=pdf_filename, mime="application/pdf")
    else:
        st.warning("The uploaded document does not appear to be a cover letter. Please ensure you have uploaded the correct file.")
else:
    st.info("Please upload a cover letter in DOCX or PDF format to start the analysis.")