from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
import tempfile

import streamlit as st
import json
import logfire
from supabase import Client
from openai import OpenAI

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)

from deepseek_assistant import (
    deepseek_assistant,
    DeepSeekAssistantDeps,
    translate_to_english,
    translate_from_english
)
from initialize import initialize_assistant

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI client with DeepSeek endpoint
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# Initialize Supabase client
supabase = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configure logfire
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""
    role: Literal['user', 'model']
    timestamp: str
    content: str
    language: str

def display_message_part(part):
    """Display a single part of a message in the Streamlit UI."""
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)
    # tool-call
    elif part.part_kind == 'tool-call':
        with st.chat_message("assistant"):
            st.info(f"Processing: {part.name}")
            if part.name == "translate_to_english":
                st.info("Translating to English...")
            elif part.name == "process_with_deepseek":
                st.info("Processing with DeepSeek-R1...")
            elif part.name == "translate_from_english":
                st.info("Translating back to user language...")
    # tool-return
    elif part.part_kind == 'tool-return':
        with st.chat_message("assistant"):
            st.success(f"Result: {part.content}")

async def process_audio(audio_file, language_code: str, deps: DeepSeekAssistantDeps):
    """Process uploaded audio file and return transcription."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Get transcription
        transcription = await deepseek_assistant.transcribe_audio.arun(
            audio_path=tmp_path,
            language_code=language_code,
            deps=deps
        )

        # Store transcription
        await deepseek_assistant.store_transcription.arun(
            transcription=transcription,
            language_code=language_code,
            deps=deps
        )

        return transcription
    finally:
        os.unlink(tmp_path)

async def run_agent_with_streaming(user_input: str, language_code: str, deps: DeepSeekAssistantDeps):
    """Run the agent with streaming text and translation flow."""
    async with deepseek_assistant.run_stream(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[:-1],
    ) as result:
        partial_text = ""
        message_placeholder = st.empty()

        # Show translation status
        if language_code != 'en':
            with st.status("Processing...", expanded=True) as status:
                st.write("Translating to English...")
                english_input = await translate_to_english(deps.openai_client, user_input, language_code)
                
                st.write("Processing with DeepSeek-R1...")
                async for chunk in result.stream_text(delta=True):
                    partial_text += chunk
                    message_placeholder.markdown(partial_text)
                
                st.write("Translating response back...")
                translated_response = await translate_from_english(
                    deps.openai_client,
                    partial_text,
                    language_code
                )
                status.update(label="Complete!", state="complete")
                
                # Update the display with translated response
                message_placeholder.markdown(translated_response)
                partial_text = translated_response
        else:
            # For English, stream directly
            async for chunk in result.stream_text(delta=True):
                partial_text += chunk
                message_placeholder.markdown(partial_text)

        # Update message history
        filtered_messages = [msg for msg in result.new_messages() 
                           if not (hasattr(msg, 'parts') and 
                                 any(part.part_kind == 'user-prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)

        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )

async def main():
    st.title("DeepSeek Multilingual Assistant")
    st.write("""I can help you with:
    - Transcribing audio in multiple languages
    - Answering questions about transcriptions
    - Analyzing content across languages
    
All processing is done in English for best results, with automatic translation to and from your chosen language!""")
    
    # Initialize the assistant
    if 'assistant_deps' not in st.session_state:
        st.session_state.assistant_deps = await initialize_assistant()

    # Language selector
    languages = {
        'English': 'en',
        'Hindi': 'hi',
        'Arabic': 'ar',
        'German': 'de',
        'Spanish': 'es',
        'French': 'fr',
        'Japanese': 'ja',
        'Russian': 'ru',
        'Tamil': 'ta',
        'Turkish': 'tr'
    }
    selected_language = st.selectbox(
        "Select Language",
        options=list(languages.keys()),
        index=0
    )
    language_code = languages[selected_language]

    # Audio file upload
    audio_file = st.file_uploader("Upload Audio File (WAV format)", type=['wav'])
    if audio_file:
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing audio..."):
                transcription = await process_audio(
                    audio_file, 
                    language_code,
                    st.session_state.assistant_deps
                )
                st.success("Transcription complete!")
                
                # Show both original and translated versions if not English
                if language_code != 'en':
                    with st.expander("Show translation details"):
                        st.write("Original transcription:")
                        st.write(transcription)
                        st.write("English translation:")
                        english_trans = await translate_to_english(
                            st.session_state.assistant_deps.openai_client,
                            transcription,
                            language_code
                        )
                        st.write(english_trans)
                else:
                    st.write(transcription)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input
    user_input = st.chat_input(f"Ask a question in {selected_language}")

    if user_input:
        # Append new request to conversation
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        # Display user prompt
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display assistant's response while streaming
        with st.chat_message("assistant"):
            await run_agent_with_streaming(
                user_input,
                language_code,
                st.session_state.assistant_deps
            )

if __name__ == "__main__":
    asyncio.run(main()) 