import io
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, pipeline
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
    Load the Hugging Face text generation model for story generation.
    """
    return pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")


# ==========================================
# Phase 1: Image Processing & Captioning
# ==========================================

def generate_image_caption(image: Image.Image) -> str:
    """
    Process the uploaded image and generate a descriptive text caption.
    Also generates a second conditional caption to extract richer details
    (e.g., subject, setting, mood) for better story grounding.
    (Requirement 1: Image Processing & Captioning)

    Args:
        image: A PIL Image object uploaded by the user.

    Returns:
        A combined descriptive caption string derived from the image.
    """
    processor, model = load_caption_model()

    # Convert image to RGB to prevent tensor errors with PNG alpha channels
    if image.mode != "RGB":
        image = image.convert("RGB")

    # --- Caption 1: Unconditional (general scene description) ---
    inputs_unconditional = processor(image, return_tensors="pt")
    out_unconditional = model.generate(**inputs_unconditional, max_new_tokens=50)
    caption_general = processor.decode(out_unconditional[0], skip_special_tokens=True)

    # --- Caption 2: Conditional (focus on what the subject is doing) ---
    conditional_text = "a photo of"
    inputs_conditional = processor(image, conditional_text, return_tensors="pt")
    out_conditional = model.generate(**inputs_conditional, max_new_tokens=50)
    caption_detail = processor.decode(out_conditional[0], skip_special_tokens=True)

    # Combine both captions for a richer scene description
    # Avoid pure duplication by only appending detail if it adds new info
    if caption_detail.lower().strip() != caption_general.lower().strip():
        combined_caption = f"{caption_general}. More specifically, {caption_detail}"
    else:
        combined_caption = caption_general

    return combined_caption


# ==========================================
# Phase 2: Story Generation
# ==========================================

def generate_kid_story(caption: str) -> str:
    """
    Expand the image caption into a complete, image-grounded bedtime story
    strictly between 50-100 words. The story is tightly anchored to the
    visual details extracted from the caption.
    (Requirement 2: Story Generation)

    Args:
        caption: The descriptive caption generated from the uploaded image.

    Returns:
        A kid-friendly story string between 50 and 100 words.
    """
    story_generator = load_story_model()

    # Use the correct prompt format required by pranavpsv/genre-story-generator-v2:
    # Format: <BOS> <genre> <opening sentence>
    # This anchors the story directly to the image caption so it stays relevant.
    prompt = (
        f"<BOS> <drama> Once upon a time, {caption}. "
        f"This is a magical story about {caption}. "
    )

    # Generate raw text with extra buffer tokens to allow post-processing
    story_result = story_generator(
        prompt,
        max_new_tokens=160,
        min_new_tokens=90,
        do_sample=True,
        temperature=0.75,       # Balanced creativity vs coherence
        repetition_penalty=1.3, # Prevents repetitive phrasing
        top_p=0.92,             # Nucleus sampling for natural language
    )
    raw_story = story_result[0]["generated_text"]

    # --- Post-processing: strip the injected prompt prefix ---
    if raw_story.startswith(prompt):
        story_text = raw_story[len(prompt):]
    else:
        story_text = raw_story.replace(prompt, "", 1)

    # Remove any leftover model-specific tokens (e.g. <BOS>, <drama>)
    for token in ["<BOS>", "<drama>", "<EOS>"]:
        story_text = story_text.replace(token, "")
    story_text = story_text.strip()

    # --- Sentence-level reconstruction to enforce 50-100 word limit ---
    sentences = story_text.replace("!", ".").replace("?", ".").split(".")

    final_story = ""
    word_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_words = len(sentence.split())

        # Stop before exceeding 95 words (buffer to stay under 100)
        if word_count + sentence_words > 95:
            break

        final_story += sentence + ". "
        word_count += sentence_words

        # Stop once we've hit the comfortable 55-word minimum
        if word_count >= 55:
            break

    # --- Fallback: if post-processing yields too short a story ---
    if len(final_story.split()) < 50:
        words = story_text.split()
        fallback = " ".join(words[:90])
        if not fallback.endswith((".", "!", "?")):
            fallback += "."
        return fallback

    return final_story.strip()


# ==========================================
# Phase 3: Text-to-Speech Conversion
# ==========================================

def text_to_speech(story_text: str) -> io.BytesIO:
    """
    Convert the generated story text into an audio bytes buffer using gTTS.
    gTTS is chosen for its lightweight footprint and Streamlit Cloud stability.
    (Requirement 3: Text-to-Speech Conversion)

    Args:
        story_text: The generated story string to be read aloud.

    Returns:
        A BytesIO buffer containing the MP3 audio data.
    """
    # gTTS converts text to speech via Google TTS API
    tts = gTTS(text=story_text, lang="en", slow=False)

    # Write audio to an in-memory buffer to avoid disk I/O on cloud deployments
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)

    return audio_buffer


# ==========================================
# Main Application Flow
# ==========================================

def main():
    """
    Main entry point for the Magic Storybook Streamlit application.
    Orchestrates the UI layout and the three-phase pipeline:
    Image -> Caption -> Story -> Audio.
    """
    # --- Page configuration ---
    st.set_page_config(
        page_title="Magic Storybook",
        page_icon="🧚‍♀️",
        layout="wide"
    )

    # --- UI Header ---
    st.title("🧚‍♀️ The Magic Storybook 🦄")
    st.markdown(
        "**Welcome, little adventurer!** 🌟 "
        "Upload a picture, and our magic AI will write a bedtime story just for you and read it aloud!"
    )
    st.divider()

    # --- Sidebar instructions ---
    with st.sidebar:
        st.header("🛠️ How to Play?")
        st.write("1️⃣ Upload a fun picture.")
        st.write("2️⃣ Wait for the magic to happen.")
        st.write("3️⃣ Read and listen to your story!")
        st.image("https://cdn-icons-png.flaticon.com/512/3069/3069172.png", width=150)

    # --- Image uploader ---
    uploaded_file = st.file_uploader(
        "🖼️ Upload your picture here (JPG or PNG):",
        type=["jpg", "jpeg", "png"]
    )

    # --- Core pipeline execution ---
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])

        # Column 1: Display uploaded image
        with col1:
            st.subheader("Your Magic Picture 📸")
            try:
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)
            except Exception:
                st.error("⚠️ Error loading image. Please upload a valid JPG or PNG file.")
                return

        # Column 2: Run the three-phase pipeline
        with col2:
            st.subheader("Your Story 📖")

            # --- Phase 1: Image -> Caption ---
            with st.spinner("🔍 The Magic Eye is looking at your picture..."):
                try:
                    caption = generate_image_caption(image)
                except Exception as e:
                    st.error(f"⚠️ Could not read the picture. Please try another image.\n\n`{e}`")
                    return
            st.info(f"**✨ Magic sees:** {caption.capitalize()}")

            # --- Phase 2: Caption -> Story ---
            with st.spinner("✍️ The Magic Pen is writing your story..."):
                try:
                    story = generate_kid_story(caption)
                except Exception as e:
                    st.error(f"⚠️ Story generation failed. Please try again.\n\n`{e}`")
                    return

            # Validate word count before displaying
            word_count = len(story.split())
            if word_count < 50:
                st.warning(
                    f"⚠️ The story is a bit short ({word_count} words). "
                    "Try uploading a different image for a fuller story!"
                )
            st.success(f"**{story}**")
            st.caption(f"*(Story length: {word_count} words)*")

            # --- Phase 3: Story -> Audio ---
            with st.spinner("🗣️ The Storyteller is preparing to read..."):
                try:
                    audio_bytes = text_to_speech(story)
                except Exception as e:
                    st.error(f"⚠️ Audio generation failed.\n\n`{e}`")
                    return

            st.markdown("### 🎧 Listen to the Story!")
            st.audio(audio_bytes, format="audio/mp3")

            # Celebratory animation on successful completion
            st.balloons()


# --- App entry point ---
if __name__ == "__main__":
    main()
