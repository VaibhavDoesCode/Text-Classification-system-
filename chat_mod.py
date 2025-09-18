import streamlit as st
from PIL import Image
import easyocr
from transformers import pipeline
import pandas as pd
import numpy as np
import io
import re
import plotly.express as px

st.title("💬 Chat Insight AI: Understand Your Conversations")
# --- Custom CSS for a cleaner look ---
st.markdown("""
<style>
    /* General body styling */
    body {
        font-family: 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        color: #333;
        background-color: #f0f2f6; /* Light gray background */
    }
    /* Main content area padding */
    .main .block-container {
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    /* Header styling */
    h1 {
        color: #2c3e50; /* Dark blue-gray */
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    h2, h3, h4 {
        color: #34495e; /* Slightly lighter dark blue-gray */
        margin-top: 1.5em;
        margin-bottom: 0.8em;
    }
    /* Info/Success/Warning boxes */
    .stAlert {
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stAlert.info {
        background-color: #e0f7fa; /* Light cyan */
        color: #00796b; /* Dark cyan */
        border-left: 5px solid #00bcd4; /* Cyan border */
    }
    .stAlert.success {
        background-color: #e8f5e9; /* Light green */
        color: #2e7d32; /* Dark green */
        border-left: 5px solid #4caf50; /* Green border */
    }
    .stAlert.warning {
        background-color: #fff3e0; /* Light orange */
        color: #ef6c00; /* Dark orange */
        border-left: 5px solid #ff9800; /* Orange border */
    }
    /* Buttons */
    .stButton > button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 0.5rem;
        border: none;
        font-size: 1rem;
        transition: background-color 0.3s ease;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #45a049; /* Darker green on hover */
    }
    .stButton > button:active {
        background-color: #3e8e41;
    }
    /* Primary button styling */
    .stButton > button.primary {
        background-color: #007bff; /* Blue */
    }
    .stButton > button.primary:hover {
        background-color: #0056b3; /* Darker blue on hover */
    }
    /* Sidebar styling */
    .css-1d391kg { /* This class might change, inspect your app for the correct one */
        background-color: #2c3e50; /* Dark blue-gray sidebar */
        color: white;
    }
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4, .css-1d391kg p, .css-1d391kg .stCheckbox > label {
        color: white !important;
    }
    .css-1d391kg .stFileUploader label {
        color: white !important;
    }
    /* Data editor styling */
    .stDataFrame {
        border-radius: 0.5rem;
        overflow: hidden; /* Ensures border-radius applies */
    }
    /* Markdown links */
    a {
        color: #007bff;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions (from previous versions, slightly refined) ---
@st.cache_resource
def get_ocr_reader(use_gpu: bool = False):
    """Initializes and caches the EasyOCR reader."""
    return easyocr.Reader(["en"], gpu=use_gpu)

@st.cache_resource
def get_emotion_classifier():
    """Initializes and caches the Hugging Face emotion classifier pipeline."""
    return pipeline(
        "text-classification",
        model="bhadresh-savani/distilbert-base-uncased-emotion",
        return_all_scores=False,
    )

def detect_flirtish(text: str) -> bool:
    """Detects if a message contains flirtatious keywords or emojis."""
    flirt_words = ["😘", "😍", "❤️", "baby", "babe", "cutie", "hot", "sexy",
                   "sweetheart", "miss you", "love you", "xoxo", "honey", "darling", "hugs", "kisses"]
    text_lower = text.lower()
    return any(word in text_lower for word in flirt_words)

def preprocess_ocr_lines(lines):
    """Cleans and preprocesses OCR extracted lines."""
    out = []
    for l in lines:
        if not l:
            continue
        l = l.strip()
        # Remove common chat timestamps (e.g., "10:30 AM", "1:05pm")
        l = re.sub(r"\d{1,2}:\d{2}\s?(AM|PM|am|pm)?", "", l)
        # Remove leading non-alphanumeric characters (e.g., bullet points, special symbols)
        l = re.sub(r"^\W+", "", l)
        # Remove common chat metadata like "You" or contact names if they appear at the start
        l = re.sub(r"^(You|Me|[\w\s]+?):\s*", "", l, flags=re.IGNORECASE)
        if l:
            out.append(l)
    return out

def create_summary(predictions):
    """
    Generates a summary of the chat's emotional tone and provides advice.
    Returns the summary string and a dictionary of label counts.
    """
    if not predictions:
        return "No messages to summarize.", {}

    label_counts = {}
    for p in predictions:
        label = p["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    top_label, top_count = sorted_labels[0] if sorted_labels else ("neutral", 0)

    label_explanations = {
        "joy": "happy/excited — positive, enthusiastic tone",
        "love": "affectionate/romantic — warm or loving messages",
        "optimism": "optimistic/hopeful — forward-looking or encouraging",
        "anger": "angry/annoyed — upset or frustrated tone",
        "disgust": "disgusted — strong negative reaction",
        "sadness": "sad/upset — disappointed or hurt tone",
        "fear": "anxious/worried — expressing concern or fear",
        "surprise": "surprised — unexpected reaction",
        "neutral": "neutral/informative — straightforward, non-emotional",
        "embarrassment": "embarrassed — awkward or apologetic tone",
        "confusion": "confused — unclear or seeking clarification",
        "flirtish": "romantic/flirty — playful or intimate tone 😉",
    }

    top_expl = label_explanations.get(top_label.lower(), f"{top_label} (tone)")

    summary_lines = []
    summary_lines.append(f"### Overall Chat Tone Analysis")
    summary_lines.append(f"The dominant tone detected is: **{top_label.upper()}** (found in {top_count} message(s)).")
    summary_lines.append(f"This suggests an interpretation of: *{top_expl}*.")

    if len(sorted_labels) > 1:
        other_tones = ", ".join([f"{lbl} ({cnt})" for lbl, cnt in sorted_labels[1:4]])
        summary_lines.append(f"Other significant tones observed include: {other_tones}.")

    advice_map = {
        "joy": "Reply in kind — positive engagement (emojis, plans, appreciation).",
        "love": "Be warm and reciprocate if appropriate.",
        "anger": "Acknowledge feelings, don't escalate. A short apology or clarification may help.",
        "sadness": "Show empathy, ask if they're okay, offer support.",
        "confusion": "Clarify and ask what they mean.",
        "neutral": "Respond as usual with info or logistics.",
        "flirtish": "Play along if you like them 😉, or keep it light if unsure.",
        "disgust": "Address the source of disgust carefully, or change the topic.",
        "fear": "Reassure them and offer support.",
        "surprise": "Acknowledge their surprise and ask for more context if needed.",
        "embarrassment": "Offer reassurance or change the subject to ease their discomfort.",
        "optimism": "Encourage their positive outlook and share in their hope."
    }
    advice = advice_map.get(top_label.lower(), "Consider a calm, clarifying reply based on context.")
    summary_lines.append(f"\n**Quick Suggestion for your reply:** {advice}")

    return "\n\n".join(summary_lines), label_counts

# --- Main Application UI ---
st.title("💬 Chat Insight AI: Understand Your Conversations")
st.markdown(
    """
    Upload a chat screenshot (WhatsApp / Telegram / SMS) and let AI analyze the emotional tone of each message.
    Get a quick summary and actionable advice on how to respond!
    """
)

# --- Sidebar for Upload and Settings ---
with st.sidebar:
    st.header("🚀 Get Started")
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Chat Screenshot", type=["png", "jpg", "jpeg"], help="Supported formats: PNG, JPG, JPEG.")
    use_gpu = st.checkbox("Enable GPU for OCR (if available)", value=False, help="Check this for faster text extraction if you have a compatible GPU (e.g., NVIDIA CUDA).")

    # Use a form to group OCR and Analyze buttons for better state management
    with st.form("ocr_form"):
        st.markdown("---")
        st.subheader("Process Screenshot")
        submitted_ocr = st.form_submit_button("Extract Text & Analyze", type="primary")

    st.markdown("---")
    st.header("🛠️ About This Tool")
    st.info(
        """
        This application leverages cutting-edge AI models to provide insights into your chat conversations:
        - **Text Extraction (OCR):** Powered by [EasyOCR](https://github.com/JaidedAI/EasyOCR) (English only).
        - **Emotion Classification:** Uses a fine-tuned [DistilBERT model](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion) from Hugging Face.
        - **Flirt Detection:** Custom keyword and emoji matching.
        - **Built with:** [Streamlit](https://streamlit.io/) for interactive web app development.
        """
    )
    st.markdown("**Important Notes:**\n"
                "- OCR accuracy depends on image quality (clear, well-lit screenshots work best).\n"
                "- AI models are not perfect; results are interpretations and should be used as guidance.\n"
                "- All processing happens directly in your browser session (or on the Streamlit Cloud server); your data is **private** and not stored.")
    st.markdown("---")
    st.markdown("Developed with ❤️ by [Vaibhav Singh](https://github.com/vaibhavsingh97)") # Link to your GitHub

# --- Main Content Area ---
if uploaded_file is None:
    st.image("https://via.placeholder.com/800x400?text=Upload+a+Screenshot+to+Begin", use_column_width=True) # Placeholder image
    st.info("⬆️ **Upload a chat screenshot from the sidebar to get started!**")
else:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("📸 Your Uploaded Screenshot")
    st.image(image, caption="Uploaded screenshot", use_column_width=True)

    # --- OCR Step Triggered by Form Submission ---
    if submitted_ocr:
        with st.spinner("Extracting text from image... This might take a moment."):
            try:
                reader = get_ocr_reader(use_gpu=use_gpu)
                img_np = np.array(image)
                ocr_results = reader.readtext(img_np, detail=0, paragraph=False)
                extracted_lines = preprocess_ocr_lines(ocr_results)
                st.session_state["extracted_lines"] = extracted_lines
                st.session_state["analysis_performed"] = False # Reset analysis flag
                st.success("Text extraction complete! Review and proceed to analysis.")
            except Exception as e:
                st.error(f"Error during OCR: {e}. Please try another image or check GPU settings.")
                st.session_state["extracted_lines"] = [] # Clear previous state on error

    # --- Editable DataFrame for Review ---
    extracted_lines = st.session_state.get("extracted_lines", [])
    if extracted_lines:
        st.subheader("📝 Review & Edit Extracted Messages")
        st.info("The AI extracted these messages. Please review and correct any errors. You can add/remove rows too!")
        editable_df = pd.DataFrame({"text": extracted_lines})
        edited_df = st.data_editor(
            editable_df,
            num_rows="dynamic",
            use_container_width=True,
            key="message_editor" # Unique key for the data editor
        )

        # --- Classification Triggered by a separate button ---
        if st.button("✨ Analyze Emotions & Generate Report", type="primary", key="analyze_button"):
            with st.spinner("Analyzing emotions and generating insights..."):
                classifier = get_emotion_classifier()
                preds = []
                for txt in edited_df["text"].tolist():
                    if not str(txt).strip():
                        continue

                    try:
                        res = classifier(str(txt))
                        label = res[0].get("label") if isinstance(res, list) else res.get("label")
                        score = float(res[0].get("score", 0.0)) if isinstance(res, list) else float(res.get("score", 0.0))

                        if detect_flirtish(txt):
                            label = "flirtish"
                            score = 0.99
                    except Exception:
                        label = "error"
                        score = 0.0

                    preds.append({
                        "text": txt,
                        "label": label,
                        "score": round(score, 3)
                    })

            preds_df = pd.DataFrame(preds)
            st.session_state["preds_df"] = preds_df
            st.session_state["analysis_performed"] = True
            st.success("Analysis complete! Scroll down for your insights.")

    # --- Display Results (only if analysis has been performed) ---
    if st.session_state.get("analysis_performed", False):
        preds_df = st.session_state.get("preds_df")
        if preds_df is not None:
            st.markdown("---")
            st.subheader("📊 Detailed Message Analysis")
            st.dataframe(preds_df, use_container_width=True)

            summary, label_counts = create_summary(preds_df.to_dict('records')) # Pass as list of dicts
            st.markdown("---")
            st.subheader("💡 Chat Summary & Interpretation")
            st.markdown(summary)

            # --- Engaging Pie Chart ---
            if label_counts:
                st.markdown("---")
                st.subheader("📈 Emotional Breakdown of the Conversation")
                chart_data_plotly = pd.DataFrame(label_counts.items(), columns=['Emotion', 'Count'])

                # Sort for consistent pie chart slices (largest first)
                chart_data_plotly = chart_data_plotly.sort_values(by='Count', ascending=False)

                fig_pie = px.pie(
                    chart_data_plotly,
                    values='Count',
                    names='Emotion',
                    title='Distribution of Emotions in Messages',
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel,
                    labels={'Emotion':'Emotion Type', 'Count':'Number of Messages'},
                    hover_data=['Count']
                )
                fig_pie.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    marker=dict(line=dict(color='#000000', width=1)),
                    pull=[0.05 if emotion == chart_data_plotly['Emotion'].iloc[0] else 0 for emotion in chart_data_plotly['Emotion']] # Pull out dominant slice
                )
                fig_pie.update_layout(
                    showlegend=True,
                    title_x=0.5,
                    font=dict(size=14),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5) # Horizontal legend at bottom
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("---")
            st.subheader("⬇️ Download Your Analysis")
            csv_buf = io.StringIO()
            preds_df.to_csv(csv_buf, index=False)
            csv_contents = csv_buf.getvalue().encode("utf-8")
            st.download_button(
                "Download Classified Messages as CSV",
                csv_contents,
                file_name="chat_predictions.csv",
                mime="text/csv",
                key="download_csv_btn"
            )

            st.markdown("---")
            st.subheader("📖 Full Extracted Text (for reference)")
            st.text_area("Raw Text", "\n".join(edited_df["text"].tolist()), height=200, disabled=True)
        else:
            st.warning("No analysis data found. Please run the analysis first.")
    elif "extracted_lines" in st.session_state and not st.session_state.get("analysis_performed", False):
        st.info("Click 'Analyze Emotions & Generate Report' to see the results!")
    else:
        st.info("Upload a screenshot and click 'Extract Text & Analyze' to begin.")


