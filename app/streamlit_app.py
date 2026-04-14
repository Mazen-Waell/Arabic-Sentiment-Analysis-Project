import os
import streamlit as st
import requests

st.set_page_config(
    page_title="Arabic Sentiment Analysis",
    page_icon="🎭",
    layout="centered"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600&family=IBM+Plex+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans Arabic', sans-serif;
    }

    .main { background-color: #0f0f0f; }

    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        max-width: 700px;
    }

    .header-tag {
        display: inline-block;
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #888;
        border: 1px solid #333;
        padding: 4px 10px;
        border-radius: 4px;
        margin-bottom: 1rem;
    }

    h1 {
        font-size: 2.2rem !important;
        font-weight: 600 !important;
        color: #f0f0f0 !important;
        line-height: 1.2 !important;
        margin-bottom: 0.4rem !important;
    }

    .subtitle {
        font-size: 15px;
        color: #666;
        margin-bottom: 2.5rem;
    }

    .stTextArea textarea {
        background-color: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 8px !important;
        color: #e0e0e0 !important;
        font-family: 'IBM Plex Sans Arabic', sans-serif !important;
        font-size: 16px !important;
        padding: 14px !important;
        direction: rtl;
    }

    .stTextArea textarea:focus {
        border-color: #444 !important;
        box-shadow: none !important;
    }

    .stButton button {
        background-color: #f0f0f0 !important;
        color: #0f0f0f !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 10px 28px !important;
        width: 100%;
    }

    .stButton button:hover {
        background-color: #d4d4d4 !important;
    }

    .footer {
        margin-top: 3rem;
        font-size: 12px;
        color: #333;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-tag">NLP · Arabic · Sentiment</div>', unsafe_allow_html=True)
st.markdown("# Arabic Sentiment Analysis")
st.markdown('<p class="subtitle">Enter Arabic text to classify its sentiment using fine-tuned AraBERT.</p>', unsafe_allow_html=True)

text = st.text_area(
    label="",
    height=140,
    placeholder="اكتب النص العربي هنا...",
    label_visibility="collapsed"
)

if st.button("Analyze →"):
    if not text.strip():
        st.error("Please enter some text before analyzing.")
    else:
        try:
            with st.spinner(""):
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": text},
                    timeout=30
                )
            result     = response.json()
            sentiment  = result["sentiment"]
            confidence = result["confidence"]

            color = (
                "#4ade80" if sentiment == "Positive"
                else "#f87171" if sentiment == "Negative"
                else "#facc15"
            )

            st.markdown(f"""
            <div style="background:#1a1a1a; border:1px solid #2a2a2a; border-radius:10px; padding:24px 28px; margin-top:1.5rem;">
                <div style="font-size:11px; font-weight:500; letter-spacing:0.1em; text-transform:uppercase; color:#555; margin-bottom:8px;">Sentiment</div>
                <div style="font-size:2rem; font-weight:600; margin-bottom:4px; color:{color};">{sentiment}</div>
                <div style="font-size:13px; color:#555; font-family:monospace;">confidence: {confidence}</div>
            </div>
            """, unsafe_allow_html=True)

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Make sure the FastAPI server is running.")
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown('<div class="footer">Arabic Sentiment Analysis · AraBERT fine-tuned · 99K reviews</div>', unsafe_allow_html=True)