import re

def extract_linkedin(nlp):
    entities = {'LinkedIn': []}
    for ent in nlp.ents:
        if ent.label_ == 'LINKEDIN':
            entities['LinkedIn'].append(ent.text)
    
    # LinkedIn URL regex pattern
    linkedin_pattern = r"(https?:\/\/(www\.)?linkedin\.com\/(?:in|pub|company|school)\/[A-Za-z0-9_-]+)"
    linkedin_match = re.search(linkedin_pattern, ' '.join(entities['LinkedIn']))
    if linkedin_match:
        entities['LinkedIn'] = linkedin_match.group(0)
    else:
        entities['LinkedIn'] = None
    
    return entities

def extract_github(nlp):
    entities = {'GitHub': []}
    for ent in nlp.ents:
        if ent.label_ == 'GITHUB':
            entities['GitHub'].append(ent.text)

    # GitHub URL regex pattern
    github_pattern = r"(https?:\/\/(www\.)?github\.com\/[A-Za-z0-9_-]+)"
    github_match = re.search(github_pattern, ' '.join(entities['GitHub']))
    if github_match:
        entities['GitHub'] = github_match.group(0)
    else:
        entities['GitHub'] = None

    return entities

def extract_experience(nlp):
    experience = []
    # Assume we look for sections labeled as "Experience" or "Work Experience"
    exp_pattern = re.compile(r'(Experience|Work Experience)', re.IGNORECASE)

    # Find sentences or sections matching the pattern
    for sent in nlp.sents:
        if exp_pattern.search(sent.text):
            experience.append(sent.text.strip())  # Collect relevant experience details
    return experience if experience else "No experience found"

def extract_education(nlp):
    education = []
    # Assume we look for sections labeled as "Education"
    edu_pattern = re.compile(r'(Education)', re.IGNORECASE)

    # Find sentences or sections matching the pattern
    for sent in nlp.sents:
        if edu_pattern.search(sent.text):
            education.append(sent.text.strip())  # Collect relevant education details
    return education if education else "No education found"

def extract_entities_with_custom_model(nlp):
    # This should return a dictionary with entity types as keys
    # Ensure that your custom model is set up to extract names and degrees
    entities = {'Name': [], 'Degree': []}
    for ent in nlp.ents:
        if ent.label_ == 'PERSON':
            entities['Name'].append(ent.text)
        elif ent.label_ == 'DEGREE':
            entities['Degree'].append(ent.text)
    return entities

def extract_name(nlp, matcher):
    # Extract name logic here...
    pass

def extract_email(text):
    # Extract email logic here...
    pass

def extract_mobile_number(text, regex):
    # Extract mobile number logic here...
    pass

def extract_skills(nlp, noun_chunks, skills_file):
    # Extract skills logic here...
    pass

def get_number_of_pages(resume):
    # Logic to get number of pages...
    pass

def extract_text(resume, ext):
    # Logic to extract text from the resume...
    pass
