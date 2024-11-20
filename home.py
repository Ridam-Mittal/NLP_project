# import streamlit as st
# from app import resume  # Import the main function from app.py
# from cover import coverletter # Import the main function from cover.py

# st.set_page_config(page_title="Document Generator", page_icon="📝", layout="wide")

# # Function to create the homepage with a horizontal navbar
# def main():
#     # Create a title for the homepage
#     st.title("Welcome to the Professional Document Generator")

#     # Horizontal navigation buttons
#     col1, col2, col3 = st.columns(3)

#     with col1:
#         if st.button("Home"):
#             st.session_state.page = "home"  # Set the session state for home

#     with col2:
#         if st.button("Resume"):
#             st.session_state.page = "resume"  # Set the session state for resume

#     with col3:
#         if st.button("Cover Letter"):
#             st.session_state.page = "cover_letter"  # Set the session state for cover letter

#     # Initialize the session state page if not already done
#     if 'page' not in st.session_state:
#         st.session_state.page = "home"

#     # Render content based on the selected page
#     if st.session_state.page == "home":
#         st.write("This is the main homepage. Use the buttons above to navigate to different sections.")
#     elif st.session_state.page == "resume":
#         resume()  # Call the main function from app.py for the Resume page
#     elif st.session_state.page == "cover_letter":
#         coverletter()  # Call the main function from cover.py for the Cover Letter page

# # Run the main function
# if __name__ == "__main__":
#     main()



# import streamlit as stm

# # Set the page configuration
# stm.set_page_config(page_title="This is a Multipage WebApp")
# stm.sidebar.success("This is the Home Page Geeks.")
# stm.sidebar.success("Select Any Page from here")

# # Define CSS for stylish square frames
# frame_style = """
# <style>
# .frame {
#     border: 2px solid #FFFF;  /* Border color */
#     border-radius: 10px;        /* Rounded corners */
#     padding: 10px;              /* Space inside the frame */
#     margin: 0px 10px 10px 10px;               /* Space between frames */
#     display: inline-block;       /* Allow frames to sit next to each other */
#     width: 350px;                /* Width of each frame */
#     height: 200px;               /* Height of each frame */
#     text-align: center;          /* Center content */
# }
# </style>
# """

# # Create three square frames for images with specific image addresses
# image_addresses = [
#     "./Logo/better.jpg",   # Replace with your first local image path
#     "./Logo/Image1.png",   # Replace with your second local image path
#     "./Logo/images.jpeg"    # Replace with your third local image path
# ]

# for i, img_url in enumerate(image_addresses):
#     stm.markdown(f"""
#     <div class="frame">
#         <img src="{img_url}" alt="Image {i+1}" width="150" height="150" />
#         <p>Image {i+1}</p>
#     </div>
#     """, unsafe_allow_html=True)

# # Close the container
# stm.markdown('</div>', unsafe_allow_html=True)



import streamlit as stm
import base64
import os

# Function to convert image to base64
def get_base64_image(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Set the page configuration
stm.set_page_config(
    page_title="This is a Multipage WebApp",
    page_icon="📊"  # This will display a globe emoji as the favicon
)
stm.sidebar.success("This is the Home Page Geeks.")
stm.sidebar.success("Select Any Page from here")

# Define CSS for centered image and heading box
centered_style = """
<style>
.container {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin-top: 50px;
}

.heading-box {
    background-color: #4CAF50;   /* Background color for the heading box */
    color: white;
    padding: 10px 20px;
    border-radius: 5px;
    font-size: 24px;
    margin-bottom: 20px;
    text-align: center;
}

.image-box {
    border: 2px solid #4CAF50;
    border-radius: 10px;
    padding: 15px;
    max-width: 800px;
}
</style>
"""

# Apply the custom CSS
stm.markdown(centered_style, unsafe_allow_html=True)

# Define path for the third image
img_path = "./Logo/new.png"

# Display the heading and image if it exists
stm.markdown('<div class="container">', unsafe_allow_html=True)
stm.markdown('<div class="heading-box">CV Scan</div>', unsafe_allow_html=True)

if os.path.exists(img_path):
    base64_img = get_base64_image(img_path)
    stm.markdown(f"""
    <div class="image-box">
        <img src="data:image/jpeg;base64,{base64_img}" style="width:100%; height:auto;" />
    </div>
    """, unsafe_allow_html=True)
else:
    stm.markdown("<p>Image not found.</p>", unsafe_allow_html=True)

# Close the container
stm.markdown('</div>', unsafe_allow_html=True)








# import streamlit as st

# # Set the page config for the homepage
# st.set_page_config(page_title="Document Generator", page_icon="📝", layout="wide")

# # Function to create the homepage content
# def main():
#     # Create a title for the homepage
#     st.title("Welcome to the Professional Document Generator")

#     # Information about the project or instructions
#     st.write("""
#     This application helps you generate professional documents like resumes and cover letters.
#     Use the sidebar to navigate to different sections and start creating your documents.
#     """)

# # Run the main function
# if __name__ == "__main__":
#     main()
