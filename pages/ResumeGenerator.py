import streamlit as st
from fpdf import FPDF

# Predefined list of skills for search bar
SKILLS = ['Python', 'JavaScript', 'Machine Learning', 'Data Analysis', 'React', 'Django', 'SQL', 'Git', 'CSS', 'HTML', 'AWS', 'DevOps']

# Custom PDF class for handling the resume layout
class PDF(FPDF):
    def header(self):
        # Add name at the top as a header
        self.set_font('DejaVu', 'B', 14)  # Use bold font for name, reduced size to fit
        self.cell(0, 10, profile_data['name'], ln=True, align='C')

    def footer(self):
        # Add a footer with page number
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 8)  # Use italic font for footer
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf(profile_data):
    # Create PDF object
    pdf = PDF(orientation='P', unit='mm', format='A4')
    
    # Add the fonts for regular, bold, and italic
    pdf.add_font('DejaVu', '', 'C:/Users/User/OneDrive/Desktop/Project SmartCVAnalyzer/streamlit-react-project/pages/DejaVuSans.ttf', uni=True)
# Regular font
    pdf.add_font('DejaVu', 'B', 'C:/Users/User/OneDrive/Desktop/Project SmartCVAnalyzer/streamlit-react-project/pages/DejaVuSans-Bold.ttf', uni=True)
  # Bold font
    pdf.add_font('DejaVu', 'I', 'C:/Users/User/OneDrive/Desktop/Project SmartCVAnalyzer/streamlit-react-project/pages/DejaVuSans-Oblique.ttf', uni=True)
 # Italic font
    
    pdf.add_page()

    pdf.set_font('DejaVu', '', 10)  # Smaller regular font for content

    # Contact Info
    pdf.set_font('DejaVu', '', 10)
    pdf.cell(0, 8, profile_data['contact'], ln=True, align='C')
    pdf.ln(3)

    # Education Section
    pdf.set_font('DejaVu', 'B', 12)
    pdf.cell(0, 8, 'Education', ln=True)
    pdf.set_font('DejaVu', '', 10)
    for edu in profile_data['education']:
        pdf.cell(0, 8, f"{edu['degreeName']} - {edu['schoolName']}", ln=True)
    pdf.ln(3)

    # Experience Section
    if profile_data['experience']:
        pdf.set_font('DejaVu', 'B', 12)
        pdf.cell(0, 8, 'Experience', ln=True)
        pdf.set_font('DejaVu', '', 10)
        for exp in profile_data['experience']:
            pdf.cell(0, 8, f"{exp['title']} at {exp['company']}", ln=True)
        pdf.ln(3)

    # Projects Section
    if profile_data['projects']:
        pdf.set_font('DejaVu', 'B', 12)
        pdf.cell(0, 8, 'Projects', ln=True)
        pdf.set_font('DejaVu', '', 10)
        for proj in profile_data['projects']:
            pdf.multi_cell(0, 8, f"{proj['title']}: {proj['description']}")
        pdf.ln(3)

    # Skills Section
    if profile_data['skills']:
        pdf.set_font('DejaVu', 'B', 12)
        pdf.cell(0, 8, 'Skills', ln=True)
        pdf.set_font('DejaVu', '', 10)
        for skill in profile_data['skills']:
            pdf.cell(0, 8, f"• {skill}", ln=True)
        pdf.ln(3)

    # Achievements Section
    if profile_data['achievements']:
        pdf.set_font('DejaVu', 'B', 12)
        pdf.cell(0, 8, 'Achievements', ln=True)
        pdf.set_font('DejaVu', '', 10)
        for achievement in profile_data['achievements']:
            pdf.cell(0, 8, f"• {achievement}", ln=True)
        pdf.ln(3)

    # Save the generated PDF
    pdf.output("resume.pdf")

# Streamlit app
st.title("Resume Generator")

# Input fields for compulsory data
name = st.text_input("Full Name (Required)")
contact = st.text_input("Contact (Required - Email/Phone)")

# Education section (compulsory)
st.subheader("Education (Required)")
education = []
num_edu = st.number_input("Number of Education Entries", min_value=1, max_value=5, value=1)
for i in range(num_edu):
    degree = st.text_input(f"Degree {i+1}")
    school = st.text_input(f"School {i+1}")
    education.append({'degreeName': degree, 'schoolName': school})

# Experience section (optional)
st.subheader("Experience (Optional)")
experience = []
num_exp = st.number_input("Number of Experience Entries", min_value=0, max_value=5, value=0)
for i in range(num_exp):
    company = st.text_input(f"Company {i+1}")
    title = st.text_input(f"Title {i+1}")
    experience.append({'company': company, 'title': title})

# Projects section (optional)
st.subheader("Projects (Optional)")
projects = []
num_proj = st.number_input("Number of Projects", min_value=0, max_value=5, value=0)
for i in range(num_proj):
    project_title = st.text_input(f"Project Title {i+1}")
    project_desc = st.text_area(f"Project Description {i+1}")
    projects.append({'title': project_title, 'description': project_desc})

# Achievements section (optional)
st.subheader("Achievements (Optional)")
achievements = st.text_area("Enter Achievements (Separate by new line)", height=100).split('\n')

# Skills section with search and dynamic addition/removal (optional)
st.subheader("Skills (Optional)")
selected_skills = st.multiselect("Search and Select Your Skills", SKILLS)
st.write("Selected Skills:", selected_skills)

# Button to generate the resume PDF
if st.button("Generate Resume"):
    if not name or not contact or not education:
        st.error("Please fill in all required fields: Name, Contact, and Education.")
    else:
        profile_data = {
            'name': name,
            'contact': contact,
            'education': [{'degreeName': edu['degreeName'], 'schoolName': edu['schoolName']} for edu in education],
            'experience': [{'company': exp['company'], 'title': exp['title']} for exp in experience],
            'projects': [{'title': proj['title'], 'description': proj['description']} for proj in projects],
            'skills': selected_skills,
            'achievements': [ach for ach in achievements if ach.strip() != '']
        }
        generate_pdf(profile_data)
        st.success("Resume Generated! Download it below.")

        # Provide download link for PDF
        with open("resume.pdf", "rb") as file:
            st.download_button("Download Resume", file, file_name="resume.pdf")
