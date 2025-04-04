import streamlit as st
from PIL import Image
import pandas as pd
import io
import base64
import hashlib
import sqlite3
import os
from model import predict_disease  # Assuming the model script is in a separate file
from gemini_solutions import get_disease_solution, get_batch_solutions  # Add this import

# Database setup
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        role TEXT NOT NULL
    )
    ''')
    
    # Add default admin if not exists
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if not c.fetchone():
        hashed_password = hashlib.sha256("admin123".encode()).hexdigest()
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                 ("admin", hashed_password, "admin"))
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# User authentication
def authenticate(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    hashed_password = hash_password(password)
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))
    user = c.fetchone()
    
    conn.close()
    return user

# User registration
def register_user(username, password, role):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    try:
        hashed_password = hash_password(password)
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", 
                 (username, hashed_password, role))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

# Set page configuration for better mobile experience
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'role' not in st.session_state:
    st.session_state.role = ""

# Authentication pages
def login_page():
    st.title("🔐 Login to Plant Disease Detection")
    
    # Add role selection tabs for login
    login_tabs = st.tabs(["Farmer Login", "Admin Login"])
    
    with login_tabs[0]:  # Farmer login tab
        st.subheader("🌾 Farmer Login")
        farmer_username = st.text_input("Username", key="farmer_username")
        farmer_password = st.text_input("Password", type="password", key="farmer_password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            farmer_login = st.button("Login as Farmer", key="farmer_login")
        with col2:
            st.info("Access plant disease detection tools")
        
        if farmer_login:
            if farmer_username and farmer_password:
                user = authenticate(farmer_username, farmer_password)
                if user and user[3] == "farmer":  # Check if user exists and is a farmer
                    st.session_state.logged_in = True
                    st.session_state.username = farmer_username
                    st.session_state.role = "farmer"
                    st.success(f"Welcome, Farmer {farmer_username}!")
                    st.rerun()
                else:
                    st.error("Invalid farmer credentials")
            else:
                st.warning("Please enter both username and password")
    
    with login_tabs[1]:  # Admin login tab
        st.subheader("👨‍💼 Admin Login")
        admin_username = st.text_input("Username", key="admin_username")
        admin_password = st.text_input("Password", type="password", key="admin_password")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            admin_login = st.button("Login as Admin", key="admin_login")
        with col2:
            st.warning("Restricted access - Administrators only")
        
        if admin_login:
            if admin_username and admin_password:
                user = authenticate(admin_username, admin_password)
                if user and user[3] == "admin":  # Check if user exists and is an admin
                    st.session_state.logged_in = True
                    st.session_state.username = admin_username
                    st.session_state.role = "admin"
                    st.success(f"Welcome, Administrator {admin_username}!")
                    st.rerun()
                else:
                    st.error("Invalid admin credentials")
            else:
                st.warning("Please enter both username and password")
    
    # Sign up option
    st.markdown("---")
    st.subheader("Don't have an account?")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Create New Account"):
            st.session_state.page = "signup"
            st.rerun()
    with col2:
        st.markdown("New users can register as either farmers or admins")

def signup_page():
    st.title("📝 Sign Up")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        role = st.selectbox("Role", ["farmer", "admin"])
        
        signup_button = st.button("Sign Up")
        
        if signup_button:
            if username and password and confirm_password:
                if password != confirm_password:
                    st.error("Passwords do not match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters long")
                else:
                    if register_user(username, password, role):
                        st.success("Account created successfully! Please log in.")
                        st.session_state.page = "login"
                        st.rerun()
                    else:
                        st.error("Username already exists")
            else:
                st.warning("Please fill in all fields")
    
    with col2:
        st.subheader("Already have an account?")
        st.write("Log in to use the Plant Disease Detection system")
        
        if st.button("Go to Login"):
            st.session_state.page = "login"
            st.rerun()

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.session_state.role = ""
    st.session_state.page = "login"
    st.rerun()

# Main application with role-based functions
def main_app():
    # Sidebar with app information
    with st.sidebar:
        st.title("🌱 Plant Disease Detection")
        
        # User info
        st.markdown(f"### Welcome, {st.session_state.username}")
        st.markdown(f"**Role**: {st.session_state.role.capitalize()}")
        
        st.markdown("### About")
        st.info(
            "This application uses YOLOv8 to detect diseases in plant leaves. "
            "Upload an image to get started."
        )
        
        # Role-specific sidebar content
        if st.session_state.role == "admin":
            display_admin_sidebar()
        else:
            display_farmer_sidebar()
        
        # Add version information
        st.markdown("---")
        st.caption("Version 1.0.0")
        
        # Logout button
        if st.button("Logout"):
            logout()

    # Main content
    st.title("🌱 Plant Disease Detection")

    # Create tabs based on role
    if st.session_state.role == "admin":
        tab1, tab2, tab3, tab4 = st.tabs(["Single Image", "Batch Processing", "User Management", "Help"])
        
        with tab1:
            display_single_image_analysis()
            
        with tab2:
            display_batch_processing()
            
        with tab3:
            display_user_management()
            
        with tab4:
            display_help_documentation()
    else:
        tab1, tab2, tab3 = st.tabs(["Single Image", "Batch Processing", "Help"])
        
        with tab1:
            display_single_image_analysis()
            
        with tab2:
            display_batch_processing()
            
        with tab3:
            display_help_documentation()

    # Add footer
    display_footer()

# Admin-specific sidebar content
def display_admin_sidebar():
    st.markdown("### Admin Controls")
    st.markdown("""
    - View and manage users
    - Process plant images
    - Generate detailed reports
    """)
    
    st.markdown("### Model Information")
    st.markdown("""
    - **Model**: YOLOv8
    - **Input Size**: 640x640
    - **Supported Plants**: Various crop types
    - **Last Updated**: Admin can update model
    """)

# Farmer-specific sidebar content
def display_farmer_sidebar():
    st.markdown("### Instructions")
    st.markdown("""
    1. Upload a plant leaf image (JPG, JPEG, PNG)
    2. Wait for the model to process the image
    3. View the detection results
    4. Download the report if needed
    """)
    
    st.markdown("### Model Information")
    st.markdown("""
    - **Model**: YOLOv8
    - **Input Size**: 640x640
    - **Supported Plants**: Various crop types
    """)

# Single image analysis function (common to both roles)
def display_single_image_analysis():
    uploaded_file = st.file_uploader("Upload an image of the plant", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Add a progress bar for processing
        with st.spinner("Processing image..."):
            processed_image, detected_diseases, inference_time, detection_info = predict_disease(image)

        # Display original and processed images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Processed Image")
            st.image(processed_image, caption="Detected Disease", use_container_width=True)

        # Detection metrics in an expander
        with st.expander("🔍 Detection Metrics", expanded=True):
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Detections", len(detected_diseases))
            col2.metric("Inference Time", f"{inference_time:.4f} sec")
            col3.metric("Confidence Threshold", "0.5")  # Assuming default threshold

        # Detection details
        st.subheader("🌿 Detection Details:")
        
        if detected_diseases:
            # Create a dataframe for better display
            df_data = []
            for info in detection_info:
                # Convert confidence to float if it's not already
                confidence = info['Confidence']
                if not isinstance(confidence, (int, float)):
                    try:
                        confidence = float(confidence)
                    except (ValueError, TypeError):
                        confidence = 0.0  # Default if conversion fails
                
                df_data.append({
                    "Label": info['Label'],
                    "Class Index": info['Class Index'],
                    "Confidence": f"{confidence:.2f}",  # Now using the converted value
                    "Bounding Box": info['Bounding Box']
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # Disease information section
            st.subheader("📋 Disease Information")
            for disease in set([info['Label'] for info in detection_info]):
                with st.expander(f"{disease} Information"):
                    # Get AI-generated solution for this disease
                    # Find the confidence for this disease
                    disease_confidences = [float(info['Confidence']) for info in detection_info if info['Label'] == disease]
                    avg_confidence = sum(disease_confidences) / len(disease_confidences) if disease_confidences else 0
                    
                    solution = get_disease_solution(disease, avg_confidence)
                    
                    if solution["success"]:
                        st.markdown(solution["solution"])
                        st.caption(f"Source: {solution['source']}")
                    else:
                        st.warning(f"Could not connect to Gemini API: {solution.get('error', 'Unknown error')}")
                        st.markdown(solution["fallback_solution"])
            
            # Generate report functionality
            st.subheader("📊 Report Generation")
            
            # Create a CSV report
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_str = csv_buffer.getvalue()
            
            # Create a download button for CSV
            st.download_button(
                label="Download CSV Report",
                data=csv_str,
                file_name=f"plant_disease_report_{uploaded_file.name.split('.')[0]}.csv",
                mime="text/csv",
                key="csv_download"
            )
            
            # Create a simple PDF-like report (as HTML)
            html_report = f"""
            <h1>Plant Disease Detection Report</h1>
            <p>Image: {uploaded_file.name}</p>
            <p>Detection Time: {inference_time:.4f} seconds</p>
            <p>Total Detections: {len(detected_diseases)}</p>
            <h2>Detected Diseases:</h2>
            <ul>
            {"".join([f"<li>{info['Label']} (Confidence: {float(info['Confidence']):.2f})</li>" for info in detection_info])}
            </ul>
            """
            
            # Create a download button for HTML report
            b64_html = base64.b64encode(html_report.encode()).decode()
            href = f'<a href="data:text/html;base64,{b64_html}" download="plant_disease_report_{uploaded_file.name.split(".")[0]}.html">Download HTML Report</a>'
            st.markdown(href, unsafe_allow_html=True)
            
        else:
            st.success("✅ No disease detected. The plant appears healthy.")

# Batch processing function (common to both roles)
def display_batch_processing():
    st.subheader("Batch Processing")
    st.markdown("""
    Upload multiple images to process them in batch. 
    This feature allows you to analyze several plant images at once.
    """)
    
    batch_files = st.file_uploader("Upload multiple plant images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if batch_files:
        st.info(f"Uploaded {len(batch_files)} images for batch processing.")
        
        # Process button
        if st.button("Process All Images"):
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Container for batch results
            batch_results_container = st.container()
            
            # Process each image
            all_results = []
            
            for i, file in enumerate(batch_files):
                # Update progress
                progress = (i + 1) / len(batch_files)
                progress_bar.progress(progress)
                status_text.text(f"Processing image {i+1} of {len(batch_files)}: {file.name}")
                
                try:
                    # Open and process the image
                    image = Image.open(file)
                    processed_image, detected_diseases, inference_time, detection_info = predict_disease(image)
                    
                    # Store results
                    result = {
                        "filename": file.name,
                        "image": image,
                        "processed_image": processed_image,
                        "detected_diseases": detected_diseases,
                        "inference_time": inference_time,
                        "detection_info": detection_info
                    }
                    all_results.append(result)
                    
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
            
            # Clear progress indicators when done
            progress_bar.empty()
            status_text.empty()
            
            # Display batch results
            with batch_results_container:
                st.subheader("Batch Processing Results")
                
                # Summary statistics
                total_images = len(all_results)
                images_with_diseases = sum(1 for r in all_results if r["detected_diseases"])
                total_diseases_detected = sum(len(r["detected_diseases"]) for r in all_results)
                avg_inference_time = sum(r["inference_time"] for r in all_results) / total_images if total_images > 0 else 0
                
                # Display summary in columns
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Images", total_images)
                col2.metric("Images with Diseases", images_with_diseases)
                col3.metric("Total Detections", total_diseases_detected)
                col4.metric("Avg. Processing Time", f"{avg_inference_time:.4f} sec")
                
                # Create a summary dataframe
                summary_data = []
                for result in all_results:
                    diseases = ", ".join(set(info["Label"] for info in result["detection_info"])) if result["detection_info"] else "Healthy"
                    
                    # Convert all confidence values to float before summing
                    confidence_values = []
                    if result["detection_info"]:
                        for info in result["detection_info"]:
                            try:
                                # Convert each confidence value to float
                                confidence = float(info["Confidence"])
                                confidence_values.append(confidence)
                            except (ValueError, TypeError):
                                # Skip invalid values
                                pass
                    
                    avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
                    
                    summary_data.append({
                        "Filename": result["filename"],
                        "Status": "Diseased" if result["detected_diseases"] else "Healthy",
                        "Diseases": diseases,
                        "Detections": len(result["detected_diseases"]),
                        "Avg. Confidence": f"{avg_confidence:.2f}" if confidence_values else "N/A",
                        "Processing Time": f"{result['inference_time']:.4f} sec"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Generate batch report
                st.subheader("📊 Batch Report Generation")
                
                # Create CSV report
                csv_buffer = io.StringIO()
                summary_df.to_csv(csv_buffer, index=False)
                csv_str = csv_buffer.getvalue()
                
                # Download button for CSV
                st.download_button(
                    label="Download Batch CSV Report",
                    data=csv_str,
                    file_name=f"batch_plant_disease_report_{len(batch_files)}_images.csv",
                    mime="text/csv",
                    key="batch_csv_download"
                )
                
                # Create a detailed HTML report
                html_report = f"""
                <html>
                <head>
                    <title>Batch Plant Disease Detection Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2 {{ color: #2e7d32; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .summary {{ background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    </style>
                </head>
                <body>
                    <h1>Batch Plant Disease Detection Report</h1>
                    <div class="summary">
                        <h2>Summary</h2>
                        <p>Total Images Processed: {total_images}</p>
                        <p>Images with Diseases: {images_with_diseases}</p>
                        <p>Total Disease Detections: {total_diseases_detected}</p>
                        <p>Average Processing Time: {avg_inference_time:.4f} seconds</p>
                    </div>
                    
                    <h2>Detailed Results</h2>
                    <table>
                        <tr>
                            <th>Image</th>
                            <th>Status</th>
                            <th>Diseases</th>
                            <th>Detections</th>
                            <th>Avg. Confidence</th>
                            <th>Processing Time</th>
                        </tr>
                """
                
                for row in summary_data:
                    html_report += f"""
                        <tr>
                            <td>{row['Filename']}</td>
                            <td>{row['Status']}</td>
                            <td>{row['Diseases']}</td>
                            <td>{row['Detections']}</td>
                            <td>{row['Avg. Confidence']}</td>
                            <td>{row['Processing Time']}</td>
                        </tr>
                    """
                
                html_report += """
                    </table>
                </body>
                </html>
                """
                
                # Create a download button for HTML report
                b64_html = base64.b64encode(html_report.encode()).decode()
                href = f'<a href="data:text/html;base64,{b64_html}" download="batch_plant_disease_report_{len(batch_files)}_images.html">Download Detailed HTML Report</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                # Display individual results in expandable sections
                st.subheader("Individual Image Results")
                
                # Instead of nesting expanders, use columns or other layout elements
                for i, result in enumerate(all_results):
                    with st.expander(f"Image {i+1}: {result['filename']}"):
                        # Display images side by side
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(result["image"], caption="Original Image", use_container_width=True)
                        with col2:
                            st.image(result["processed_image"], caption="Processed Image", use_container_width=True)
                        
                        # Display detection info
                        if result["detected_diseases"]:
                            st.write(f"**Detected Diseases:** {', '.join(set(info['Label'] for info in result['detection_info']))}")
                            st.write(f"**Number of Detections:** {len(result['detected_diseases'])}")
                            st.write(f"**Processing Time:** {result['inference_time']:.4f} seconds")
                            
                            # Add disease solutions if diseases detected
                            st.subheader("🌿 Disease Solutions")
                            diseases = set(info['Label'] for info in result['detection_info'])
                            solutions = get_batch_solutions(diseases)
                            
                            # Use tabs instead of nested expanders
                            disease_tabs = st.tabs([f"Solution for {d}" for d in diseases])
                            
                            for tab, disease in zip(disease_tabs, diseases):
                                with tab:
                                    solution = solutions.get(disease, {})
                                    if solution.get("success", False):
                                        st.markdown(solution["solution"])
                                        st.caption(f"Source: {solution['source']}")
                                    else:
                                        st.warning(f"Could not connect to Gemini API: {solution.get('error', 'Unknown error')}")
                                        st.markdown(solution.get("fallback_solution", "No solution available."))
                                
                                # Create a dataframe for the detections
                                detection_df_data = []
                                for info in result["detection_info"]:
                                    confidence = info['Confidence']
                                    if not isinstance(confidence, (int, float)):
                                        try:
                                            confidence = float(confidence)
                                        except (ValueError, TypeError):
                                            confidence = 0.0
                                        
                                    detection_df_data.append({
                                        "Label": info['Label'],
                                        "Class Index": info['Class Index'],
                                        "Confidence": f"{confidence:.2f}",
                                        "Bounding Box": info['Bounding Box']
                                    })
                                
                                detection_df = pd.DataFrame(detection_df_data)
                                st.dataframe(detection_df)
                        else:
                            st.success("✅ No disease detected. The plant appears healthy.")
        else:
            # Preview of uploaded images
            st.markdown("### Preview of uploaded images")
            image_cols = st.columns(3)
            for i, file in enumerate(batch_files[:6]):  # Show first 6 images
                with image_cols[i % 3]:
                    st.image(Image.open(file), caption=file.name, width=200)
            
            if len(batch_files) > 6:
                st.caption(f"... and {len(batch_files) - 6} more")

# Admin-only user management function
def display_user_management():
    st.subheader("👥 User Management")
    
    # Display all users
    conn = sqlite3.connect('users.db')
    users_df = pd.read_sql_query("SELECT id, username, role FROM users", conn)
    conn.close()
    
    st.dataframe(users_df, use_container_width=True)
    
    # Add new user section
    with st.expander("Add New User"):
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        new_role = st.selectbox("New User Role", ["farmer", "admin"])
        
        if st.button("Add User"):
            if new_username and new_password:
                if register_user(new_username, new_password, new_role):
                    st.success(f"User '{new_username}' added successfully!")
                    st.rerun()
                else:
                    st.error("Username already exists")
            else:
                st.warning("Please enter both username and password")
    
    # Delete user section
    with st.expander("Delete User"):
        user_to_delete = st.selectbox("Select User to Delete", users_df['username'])
        
        if st.button("Delete User"):
            if user_to_delete != "admin":  # Prevent deleting the main admin
                conn = sqlite3.connect('users.db')
                c = conn.cursor()
                c.execute("DELETE FROM users WHERE username = ?", (user_to_delete,))
                conn.commit()
                conn.close()
                st.success(f"User '{user_to_delete}' deleted successfully!")
                st.rerun()
            else:
                st.error("Cannot delete the main admin account")

# Help documentation function (common to both roles)
def display_help_documentation():
    st.subheader("Help & Documentation")
    
    st.markdown("""
    ### How to Use This Application
    
    1. **Upload an Image**: Click on the 'Browse files' button in the Single Image tab and select a plant leaf image.
    
    2. **View Results**: After uploading, the system will automatically process the image and display:
       - The original and processed images
       - Detection metrics (total detections, inference time)
       - Detailed information about detected diseases
       
    3. **Generate Reports**: If diseases are detected, you can download reports in CSV or HTML format.
    
    4. **Batch Processing**: For analyzing multiple images, use the Batch Processing tab.
    
    ### Supported File Types
    - JPG/JPEG
    - PNG
    
    ### Troubleshooting
    
    **Image not processing?**
    - Ensure the image is clear and well-lit
    - Check that the file format is supported
    - Try a different image if problems persist
    
    **Unexpected results?**
    - The model works best with clear, close-up images of plant leaves
    - Ensure the leaf is the main subject of the image
    - Multiple leaves in one image may affect detection accuracy
    
    ### Contact Support
    
    For technical issues or questions, please contact support at support@plantdisease.example.com
    """)

# Footer function
def display_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center">
            <p>Developed as part of the Plant Disease Detection Project</p>
            <p>© 2023 All Rights Reserved</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Initialize page state if not exists
if 'page' not in st.session_state:
    st.session_state.page = "login"

# Main app flow
if not st.session_state.logged_in:
    if st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "signup":
        signup_page()
else:
    main_app()