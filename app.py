import streamlit as st
import pandas as pd
from utilis import extract_text_from_pdf, clean_text
from model import rank_resumes

# Page config
st.set_page_config(page_title="Resume Screening AI", layout="wide")

# Load custom CSS
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Screen Resumes", "About"])

# ---------------- HOME ----------------
if page == "Home":
    st.title("🤖 Resume Screening Dashboard")
    
    st.markdown("""
    Welcome to the AI-powered Resume Screening System.
    
    ✅ Upload resumes  
    ✅ Match with job description  
    ✅ Get ranked candidates instantly  
    """)
    
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Model", "TF-IDF")
    col2.metric("Accuracy Type", "Cosine Similarity")
    col3.metric("Status", "Ready ✅")

# ---------------- SCREENING ----------------
elif page == "Screen Resumes":
    st.title("📄 Resume Screening")

    # Input section
    with st.container():
        st.subheader("📌 Job Description")
        jd = st.text_area("Paste job description here...", height=150)

        st.subheader("📂 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

    if st.button("🚀 Analyze Resumes"):
        if jd and uploaded_files:
            with st.spinner("Processing resumes..."):
                
                cleaned_jd = clean_text(jd)

                resumes_text = []
                filenames = []

                for file in uploaded_files:
                    text = extract_text_from_pdf(file)
                    cleaned = clean_text(text)

                    resumes_text.append(cleaned)
                    filenames.append(file.name)

                ranked = rank_resumes(cleaned_jd, resumes_text)

                # Convert to DataFrame
                results = []
                for index, score in ranked:
                    results.append({
                        "Resume": filenames[index],
                        "Score": round(score, 3)
                    })

                df = pd.DataFrame(results)

            st.success("✅ Analysis Complete!")

            # ---------------- Dashboard ----------------
            st.subheader("📊 Results Overview")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Resumes", len(df))
            col2.metric("Top Score", df["Score"].max())
            col3.metric("Average Score", round(df["Score"].mean(), 2))

            # Table
            st.subheader("🏆 Ranked Candidates")
            st.dataframe(df, use_container_width=True)

            # Bar chart
            st.subheader("📈 Score Distribution")
            st.bar_chart(df.set_index("Resume"))

            # Highlight top candidate
            top_candidate = df.iloc[0]

            st.subheader("🥇 Best Match")
            st.info(f"""
            **{top_candidate['Resume']}**  
            Score: **{top_candidate['Score']}**
            """)

        else:
            st.warning("⚠️ Please enter job description and upload resumes.")

# ---------------- ABOUT ----------------
elif page == "About":
    st.title("ℹ️ About This Project")

    st.markdown("""
    This project uses NLP techniques to rank resumes.

    ### 🧠 Tech Used:
    - TF-IDF Vectorization
    - Cosine Similarity
    - Streamlit Dashboard

    ### 🚀 Future Improvements:
    - BERT embeddings
    - Skill extraction
    - Recruiter analytics dashboard
    """)