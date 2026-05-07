# Program title: Storytelling App

# Import part
import streamlit as st
from transformers import pipeline
import numpy as np # 用于处理音频数据维度

# Function part
@st.cache_resource
def get_img2text_pipeline():
    return pipeline("image-text-to-text", model="Salesforce/blip-image-captioning-base")

@st.cache_resource
def get_story_pipeline():
    return pipeline("text-generation", model="pranavpsv/genre-story-generator-v2")

@st.cache_resource
def get_audio_pipeline():
    return pipeline("text-to-audio", model="Matthijs/mms-tts-eng")

def img2text(url):
    image_to_text_model = get_img2text_pipeline()
    text = image_to_text_model(url)[0]["generated_text"]
    return text

def text2story(scenario):
    story_pipe = get_story_pipeline()
    
    # 修改 prompt 格式解决故事不相关的问题，并引导模型写儿童故事
    prompt = f"<BOS> <drama> Once upon a time, there was {scenario}. "
    
    # 限制字数，并生成故事
    story_results = story_pipe(
        prompt, 
        max_new_tokens=100,  # 限制最大长度
        min_new_tokens=50,   # 限制最小长度
        do_sample=True
    )
    raw_story = story_results[0]['generated_text']
    
    # 去除特殊的起始标签
    story = raw_story.replace("<BOS> <drama> ", "")
    
    # 截断未说完的最后一句话，保证故事完整
    last_punctuation = max(story.rfind('.'), story.rfind('!'), story.rfind('?'))
    if last_punctuation != -1:
        story = story[:last_punctuation+1]
        
    return story

def text2audio(story):
    audio_pipe = get_audio_pipeline()
    audio_data = audio_pipe(story)
    return audio_data

# Main part
st.set_page_config(page_title="Your Image to Audio Story", page_icon="🦜")
st.header("Turn Your Image to Audio Story")
uploaded_file = st.file_uploader("Select an Image...")

if uploaded_file is not None:
    # Save file locally (保留你的原始逻辑)
    bytes_data = uploaded_file.getvalue()
    with open(uploaded_file.name, "wb") as file:
        file.write(bytes_data)

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Stage 1: Image to Text
    st.text('Processing img2text...')
    scenario = img2text(uploaded_file.name)
    st.write(f"**Scenario:** {scenario}")

    # Stage 2: Text to Story (已封装为函数)
    st.text('Generating a story...')
    story = text2story(scenario)
    st.write(f"**Story:** {story}")

    # Stage 3: Story to Audio (已封装为函数)
    st.text('Generating audio data...')
    audio_data = text2audio(story)

    # Play button
    if st.button("Play Audio"):
        # 修正音频数据维度，避免 Streamlit 播放报错
        audio_array = np.squeeze(audio_data["audio"])
        sample_rate = audio_data["sampling_rate"]
        st.audio(audio_array, sample_rate=sample_rate)
