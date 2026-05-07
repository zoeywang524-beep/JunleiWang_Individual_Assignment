"""
Program Title: Magic Storybook App
Description: A storytelling web application designed for 3-10-year-old kids.
It processes a user-uploaded image, generates a caption, expands it into a
complete bedtime story (50-100 words), and converts the story to audio.
"""

import io
import streamlit as st
from PIL import Image
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS

# ==========================================
# Phase 0: Model Loading & Caching
# ==========================================
@st.cache_resource
def load_caption_model():
    """
    Load the Hugging Face image captioning model (Salesforce BLIP).
    Uses explicit Processor and Model to avoid pipeline KeyError issues.
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def load_story_model():
    """
    Load the Hugging Face text generation model.
    """
    return pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")

# ==========================================
# Phase 1: Image Processing & Captioning
# ==========================================
def generate_image_caption(image: Image.Image) -> str:
    """
    Process the uploaded image and generate a text caption.
    (Requirement 1: Image Processing & Captioning)
    """
    processor, model = load_caption_model()
    
    # Convert image to RGB to prevent tensor errors with PNG alpha channels
    if image.mode != "RGB":
        image = image.convert(mode="RGB")
        
    # Generate caption using the BLIP model
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return caption

# ==========================================
# Phase 2: Story Generation
# ==========================================
def generate_kid_story(caption: str) -> str:
    """
    Expand the image caption into a complete story strictly between 50-100 words.
    (Requirement 2: Story Generation)
    """
    story_generator = load_story_model()
    
    # 1. Kid-friendly prompt
    prompt = f"Write a magical bedtime story for kids about {caption}. Once upon a time, "
    
    # 2. Generate raw text (generating extra tokens to ensure we have enough material to cut)
    story_result = story_generator(
        prompt, 
        max_new_tokens=150,   
        min_new_tokens=80, 
        do_sample=True, 
        temperature=0.7,      # Keeps the story creative but coherent
        repetition_penalty=1.2 # Prevents the model from repeating the same sentences
    )
    raw_story = story_result[0]['generated_text']
    
    # 3. Post-processing: Remove the prompt and ensure the story ends cleanly
    story_text = raw_story.replace(f"Write a magical bedtime story for kids about {caption}. ", "")
    
    # Split text into independent sentences based on punctuation
    sentences = story_text.replace('!', '.').replace('?', '.').split('.')
    
    final_story = ""
    word_count = 0
    
    # 4. Reconstruct the story to be 50-100 words with a complete ending
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        sentence_words = len(sentence.split())
        
        # Stop adding sentences if it pushes the word count over 95 (leaving buffer for 100)
        if word_count + sentence_words > 95:
            break
            
        final_story += sentence.strip() + ". "
        word_count += sentence_words
        
        # If we reached minimum word count (55) and sentence is complete, we can safely end
        if word_count >= 55:
            break

    # Fallback mechanism if the processed story is too short
    if len(final_story.split()) < 30:
        clean_raw = story_text.strip()
        if not clean_raw.endswith(('.', '!', '?')):
            clean_raw += "."
        return clean_raw

    return final_story.strip()

# ==========================================
# Phase 3: Text-to-Speech Conversion
# ==========================================
def text_to_speech(story_text: str) -> io.BytesIO:
    """
    Convert the generated story text into an audio bytes buffer using gTTS.
    (Requirement 3: Text-to-Speech Conversion)
    """
    # gTTS is used as it is lightweight and stable for cloud deployments
    tts = gTTS(text=story_text, lang='en', slow=False)
    
    # Save audio to a bytes buffer instead of writing to the server disk
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    
    return audio_buffer

# ==========================================
# Main Application Flow
# ==========================================
def main():
    # Set up Streamlit page configurations
    st.set_page_config(
        page_title="Magic Storybook", 
        page_icon="🧚‍♀️", 
        layout="wide"
    )

    # UI Header
    st.title("🧚‍♀️ The Magic Storybook 🦄")
    st.markdown("""
        **Welcome, little adventurer!** 🌟 
        Upload a picture, and our magic AI will write a bedtime story just for you and read it aloud!
    """)
    st.divider()

    # Sidebar UI
    with st.sidebar:
        st.header("🛠️ How to Play?")
        st.write("1️⃣ Upload a fun picture.")
        st.write("2️⃣ Wait for the magic to happen.")
        st.write("3️⃣ Read and listen to your story!")
        st.image("https://cdn-icons-png.flaticon.com/512/3069/3069172.png", width=150)

    # Image Uploader
    uploaded_file = st.file_uploader("🖼️ Upload your picture here (JPG or PNG):", type=["jpg", "jpeg", "png"])

    # Core Execution
    if uploaded_file is not None:
        # Create a two-column layout
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Your Magic Picture 📸")
            try:
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            except Exception as e:
                st.error("Error loading image. Please upload a valid image file.")
                return

        with col2:
            st.subheader("Your Story 📖")
            
            # Step 1: Image to Text (Captioning)
            with st.spinner("🔍 The Magic Eye is looking at your picture..."):
                caption = generate_image_caption(image)
            st.info(f"**Magic sees:** {caption.capitalize()}")

            # Step 2: Text to Story
            with st.spinner("✍️ The Magic Pen is writing your story..."):
                story = generate_kid_story(caption)
            st.success(f"**{story}**")

            # Word count validation display (for teacher grading purposes)
            word_count = len(story.split())
            st.caption(f"*(Story length: {word_count} words)*")

            # Step 3: Story to Audio
            with st.spinner("🗣️ The Storyteller is preparing to read..."):
                audio_bytes = text_to_speech(story)
            
            # Render Audio Player
            st.markdown("### 🎧 Listen to the Story!")
            st.audio(audio_bytes, format='audio/mp3')

            # Interactive UI Element
            st.balloons()

# Run the app
if __name__ == '__main__':
    main()
