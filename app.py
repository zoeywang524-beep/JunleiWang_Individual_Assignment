
# Program title: Magic Storybook App
# Description: A storytelling application using Hugging Face models, designed for 3-10-year-old kids.

import streamlit as st
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from gtts import gTTS
import io

# ==========================================
# Phase 0: Model Loading & Caching
# ==========================================
@st.cache_resource
def load_caption_model():
    """
    Load the Hugging Face image captioning model.
    Using explicit processor and model to avoid pipeline compatibility errors.
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def load_story_model():
    """Load the Hugging Face text generation model."""
    return pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")

# ==========================================
# Phase 1: Image Processing & Captioning
# ==========================================
def generate_image_caption(image):
    """
    Process the uploaded image and generate a text caption.
    (Requirement 1: Image Processing & Captioning)
    """
    processor, model = load_caption_model()
    
    # Convert image to RGB to prevent tensor errors with PNG alpha channels
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
        
    # Generate caption
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption

# ==========================================
# Phase 2: Story Generation
# ==========================================
def generate_kid_story(caption):
    """
    Expand the image caption into a 50-100 words story.
    (Requirement 2: Story Generation)
    """
    story_generator = load_story_model()
    # Add a magical prompt to guide the model for kids
    prompt = f"Once upon a time, in a magical land, there was a {caption}. "
    
    # Generate story with length constraints (50-100 words)
    story_result = story_generator(
        prompt, 
        max_new_tokens=80,   # Limit length to match 50-100 words requirement
        min_new_tokens=40, 
        do_sample=True, 
        temperature=0.7      # Make it creative
    )
    story_text = story_result[0]['generated_text']
    return story_text

# ==========================================
# Phase 3: Text-to-Speech Conversion
# ==========================================
def text_to_speech(story_text):
    """
    Convert the generated story text into an audio file using gTTS.
    (Requirement 3: Text-to-Speech Conversion)
    """
    # Using gTTS as allowed by guidelines for better cloud stability
    tts = gTTS(text=story_text, lang='en', slow=False)
    
    # Save audio to a bytes buffer instead of a local file
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    
    return audio_buffer

# ==========================================
# Main UI & App Execution
# ==========================================
def main():
    # 1. Page Configuration
    st.set_page_config(
        page_title="Magic Storybook", 
        page_icon="🧚‍♀️", 
        layout="wide"
    )

    # 2. Kid-friendly UI Header
    st.title("🧚‍♀️ The Magic Storybook 🦄")
    st.markdown("""
        **Welcome, little adventurer!** 🌟 
        Upload a picture, and our magic AI will write a bedtime story just for you and read it aloud!
    """)
    st.divider()

    # 3. Sidebar for Instructions
    with st.sidebar:
        st.header("🛠️ How to Play?")
        st.write("1️⃣ Upload a fun picture.")
        st.write("2️⃣ Wait for the magic to happen.")
        st.write("3️⃣ Read and listen to your story!")
        st.image("https://cdn-icons-png.flaticon.com/512/3069/3069172.png", width=150) # Cute decorative image

    # 4. Image Upload Widget
    uploaded_file = st.file_uploader("🖼️ Upload your picture here (JPG or PNG):", type=["jpg", "jpeg", "png"])

    # 5. Process Flow
    if uploaded_file is not None:
        # Layout: Left column for image, Right column for story & audio
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Your Magic Picture 📸")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("Your Story 📖")
            
            # Step 1: Captioning (with spinner for UI)
            with st.spinner("🔍 The Magic Eye is looking at your picture..."):
                caption = generate_image_caption(image)
            st.info(f"**Magic sees:** {caption.capitalize()}")

            # Step 2: Story Generation
            with st.spinner("✍️ The Magic Pen is writing your story..."):
                story = generate_kid_story(caption)
            st.success(f"**{story}**")

            # Step 3: Audio Generation
            with st.spinner("🗣️ The Storyteller is preparing to read..."):
                audio_bytes = text_to_speech(story)
            
            # Audio Player UI
            st.markdown("### 🎧 Listen to the Story!")
            st.audio(audio_bytes, format='audio/mp3')

            # Celebration UI element
            st.balloons()

if __name__ == '__main__':
    main()
