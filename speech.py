import streamlit as st
import speech_recognition as sr
from textblob import TextBlob
from gtts import gTTS
import os
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import nltk
import google.generativeai as genai
import pandas as pd
from datetime import datetime
import base64
from io import BytesIO
import time
import uuid

# Configure Gemini API key
api_key=st.secrets["bilal_api"]

# Configure Gemini API key
genai.configure(api_key=api_key)  # 🔁 Replace with your actual API key

# Download necessary NLTK resources
nltk.download("punkt")

def get_base64_image():
    for ext in ["webp", "jpg", "jpeg", "png"]:
        image_path = f"bg2.{ext}"
        if os.path.exists(image_path):
            with open(image_path, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode()
    return None

bg_img = get_base64_image()

# --- Page Setup ---
st.set_page_config(page_title="Wellness Planner", layout="centered")

if bg_img:
    st.markdown(f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255, 255, 255, 0.25), rgba(255, 255, 255, 0.85)),
                        url("data:image/png;base64,{bg_img}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .block-container {{
            background-color: rgba(255, 248, 243, 0.45);
            padding: 2rem 3rem;
            border-radius: 18px;
            margin-top: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #4B4B4B;
            font-family: 'Segoe UI', sans-serif;
        }}
        .export-buttons {{
            margin-top: 20px;
        }}
        /* Hide audio player */
        .stAudio {{
            display: none;
        }}
        </style>
    """, unsafe_allow_html=True)

# Global variable to track audio state
if 'audio_playing' not in st.session_state:
    st.session_state.audio_playing = False

# Text-to-speech function with proper audio element handling
def speak_text(text):
    # Clear any previous audio
    st.session_state.audio_playing = False
    
    # Generate audio
    tts = gTTS(text=text, lang='en', slow=False)
    audio_file = BytesIO()
    tts.write_to_fp(audio_file)
    audio_file.seek(0)
    
    # Create a unique key for this audio element
    audio_key = f"audio_{uuid.uuid4()}"
    
    # Store the audio data in session state
    st.session_state[audio_key] = audio_file
    st.session_state.current_audio = audio_key
    st.session_state.audio_playing = True
    
    # Wait for audio to play
    time.sleep(len(text) * 0.1)

# Render audio elements separately
def render_audio():
    if st.session_state.audio_playing and 'current_audio' in st.session_state:
        audio_key = st.session_state.current_audio
        audio_file = st.session_state[audio_key]
        st.audio(audio_file, format='audio/mp3', autoplay=True)
        st.session_state.audio_playing = False

# Speech-to-text function
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        speak_text("Please speak when you're ready...")
        
        # Show listening indicator
        with st.spinner("Listening..."):
            audio = recognizer.listen(source)

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        speak_text("I didn't catch that. Could you say it again?")
        return ""
    except sr.RequestError:
        speak_text("There's an issue with the speech recognition service.")
        return ""

# Ask Gemini for a response
def ask_genai(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()

# Nickname introduction
def get_nickname():
    speak_text("Welcome to our session. What is your name?")
    render_audio()
    name = speech_to_text()

    speak_text("Do you prefer me to call you by your name or a nickname?")
    render_audio()
    response = speech_to_text()

    if response.lower() in ['nickname', 'yes', 'y']:
        speak_text("What nickname would you like me to use?")
        render_audio()
        nickname = speech_to_text()
    else:
        nickname = name

    speak_text(f"Thank you. I'll call you {nickname}. Let's get started.")
    render_audio()
    return nickname

# Ask questions and get GenAI feedback
def ask_questions(nickname):
    # Create a main container that will hold everything
    main_container = st.container()
    
    with main_container:
        # Create sub-containers with exact spacing control
        question_section = st.empty()
        response_section = st.empty()
        feedback_section = st.empty()
        
        questions = [
            "Can you tell me how your day has been so far?",
            "What is something you enjoy doing?",
            "How do you feel when you spend time with your family or friends?",
            "What is one thing that makes you happy?",
            "How do you feel when trying something new or unexpected?"
        ]

        polarity_sum = 0
        responses = {}

        for i, question in enumerate(questions, 1):
            # Display question
            with question_section:
                st.markdown(f"### Question {i}")
                st.write(question)
                speak_text(question)
                render_audio()

            # Get response
            response = speech_to_text()
            if response:
                with response_section:
                    st.write(f"**Your response:** {response}")
                    responses[f"Question {i}"] = response

                    # Process response
                    sentiment = TextBlob(response)
                    polarity_sum += sentiment.polarity

                    # Get feedback
                    genai_response = ask_genai(f"The child said: '{response}'. Give a short, supportive reply in 1 sentence.")
                    with feedback_section:
                        st.write(f"**Feedback:** {genai_response}")
                        speak_text(genai_response)
                        render_audio()

                    # Follow-up question
                    followup_question = ask_genai(f"""Based on: "{response}", ask a simple follow-up question.""")
                    with question_section:
                        st.write(f"**Follow-up:** {followup_question}")
                        speak_text(followup_question)
                        render_audio()
                        followup_response = speech_to_text()

                    if followup_response:
                        responses[f"Follow-up to Question {i}"] = followup_response
                        with response_section:
                            st.write(f"**Follow-up response:** {followup_response}")
                        sentiment = TextBlob(followup_response)
                        polarity_sum += sentiment.polarity

            # Minimal separator between questions
            if i < len(questions):
                st.markdown("---")

        # Clear all temporary sections before showing summary
        question_section.empty()
        response_section.empty()
        feedback_section.empty()

        # Return responses immediately after last question
        return responses, polarity_sum



# Final mood interpretation
def interpret_score(score):
    if score > 2.5:
        return "You seem to be in a positive and cheerful mood overall. That's wonderful!"
    elif score >= 0.5:
        return "You appear to be feeling okay, with some positive moments. It's important to acknowledge those feelings."
    elif score > -0.5:
        return "It seems you might be feeling neutral or a mix of emotions. That's perfectly okay. Talking more about it could help."
    else:
        return "You might be experiencing some challenges or negative feelings. I'm here to listen and support you."

# Type-Token Ratio (TTR)
def calculate_ttr(text):
    tokens = word_tokenize(text)
    unique_tokens = set(tokens)
    return len(unique_tokens) / len(tokens) if tokens else 0

# Repeated words
def detect_repeated_words(text):
    tokens = word_tokenize(text.lower())
    freq_dist = FreqDist(tokens)
    return {word: count for word, count in freq_dist.items() if count > 1}

# Overall sentiment
def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

# === Streamlit UI ===
st.title("🧠 Child Emotion & Language Interaction System")
st.markdown("This app analyzes a child's spoken emotional expression using speech recognition, sentiment analysis, and Gemini AI feedback.")

# Render any pending audio
render_audio()
def show_summary(nickname, final_feedback):
    """Display summary and play TTS without delay"""
    # Create a dedicated container for instant summary display
    summary_container = st.container()
    
    with summary_container:
        st.subheader(f"🗣️ Feedback for {nickname}")
        st.write(final_feedback)
        
        # Immediate TTS without waiting for previous audio
        if 'current_audio' in st.session_state:
            st.session_state.audio_playing = False
            del st.session_state.current_audio
        
        # Generate and play summary audio
        tts = gTTS(text=f"Thanks for sharing, {nickname}. Here's what I learned: {final_feedback}", 
                  lang='en', slow=False)
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        
        # Play immediately without container management
        st.audio(audio_file, format='audio/mp3', autoplay=True)

# In your main app flow:

    
    # Rest of your download buttons etc...
if st.button("🎤 Start Interaction"):
    nickname = get_nickname()
    responses, polarity_score = ask_questions(nickname)
    combined_responses = " ".join(responses.values())

    ttr = calculate_ttr(combined_responses)
    repeated_words = detect_repeated_words(combined_responses)
    polarity, subjectivity = analyze_sentiment(combined_responses)
    mood = interpret_score(polarity_score)

    # Summary to Gemini
    summary_prompt = f"""
You are a warm-hearted language coach and child communication specialist analyzing a speech session with an autistic child. Please gently assess their strengths and needs, based on the transcript.

Transcript:
{combined_responses}

Session data:
- Vocabulary Diversity (Type-Token Ratio): {ttr:.2f}
- Repeated Words (and their counts): {repeated_words}
- Overall Mood: {mood}
- Combined Sentiment Score: {polarity_score:.2f}
- Sentiment Polarity: {polarity:.2f}
- Subjectivity: {subjectivity:.2f}

✨ Your role:
Kindly rate the following aspects *from 1 to 5*, and for each, offer a gentle, child-friendly note — as if encouraging a caregiver after seeing a child try their best. Keep the tone uplifting and hopeful. Avoid sounding clinical or judgmental.

🎯 Rating scale:
- 5 = Flourishing 🌟
- 4 = Growing Strong 🌿
- 3 = Budding 🌱
- 2 = Sprouting 🌧
- 1 = Just Beginning 🌑

🎨 Aspects to Rate:
1. Emotional Expression  
2. Grammar & Sentence Formation  
3. Vocabulary Usage  
4. Emotional Intelligence  
5. Language Clarity & Confidence  
6. Imaginative Thinking  
7. Social Understanding  
8. Sensory or Cognitive Clues  

📋 Format like this:
- Aspect Name: Rating – Gentle, loving feedback in one to two lines.

🌱 Suggestions for Gentle Growth:
After the ratings, include a *short, encouraging bullet list of 3–4 ways to support their next steps* — focused on joy, safety, and exploration. Keep it soft and age-appropriate.

🎁 Important:
Keep the entire tone nurturing and emotionally safe. You're not diagnosing — you're appreciating effort and offering light guidance.
"""

    final_feedback = ask_genai(summary_prompt)

    # Create data record
    data_record = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Nickname": nickname,
        "Combined_Sentiment_Score": polarity_score,
        "Vocabulary_Diversity_TTR": ttr,
        "Repeated_Words": str(repeated_words),
        "Overall_Mood": mood,
        "Sentiment_Polarity": polarity,
        "Subjectivity": subjectivity,
        "Full_Responses": combined_responses,
        "Therapist_Feedback": final_feedback
    }

    # Convert to DataFrame and save to CSV
    df = pd.DataFrame([data_record])
    csv_filename = "child_interaction_data.csv"
    
    try:
        existing_df = pd.read_csv(csv_filename)
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        updated_df = df
    
    updated_df.to_csv(csv_filename, index=False)

    st.subheader(f"🗣️ Feedback for {nickname}")
    st.write(final_feedback)
    show_summary(nickname, final_feedback)
    render_audio()
    # speak_text(final_feedback)
    render_audio()

    # Download buttons
    st.subheader("📥 Download Session Data")
    
    with open(csv_filename, "rb") as f:
        st.download_button(
            label="Download Full Session History (CSV)",
            data=f,
            file_name=csv_filename,
            mime="text/csv"
        )
    
    st.download_button(
        label="Download This Session's Feedback (TXT)",
        data=final_feedback,
        file_name=f"{nickname}_feedback_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain"
    )
