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

from openai_assistant import (
    openai_assistant,
    OpenAIAssistantDeps,
    process_with_gpt4,
    get_embedding
)
from initialize import initialize_assistant

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
            if part.name == "process_with_gpt4":
                st.info("Analyzing with GPT-4...")
    # tool-return
    elif part.part_kind == 'tool-return':
        with st.chat_message("assistant"):
            st.success(f"Result: {part.content}")

async def process_audio(audio_file, language_code: str, deps: OpenAIAssistantDeps):
    """Process uploaded audio file and return transcription."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Get transcription
        transcription = await openai_assistant.transcribe_audio.arun(
            audio_path=tmp_path,
            language_code=language_code,
            deps=deps
        )

        # Store transcription
        await openai_assistant.store_transcription.arun(
            transcription=transcription,
            language_code=language_code,
            deps=deps
        )

        return transcription
    finally:
        os.unlink(tmp_path)

async def run_agent_with_streaming(user_input: str, language_code: str, deps: OpenAIAssistantDeps):
    """Run the agent with streaming text."""
    async with openai_assistant.run_stream(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[:-1],
    ) as result:
        partial_text = ""
        message_placeholder = st.empty()

        # Show processing status
        with st.status("Processing...", expanded=True) as status:
            st.write("Analyzing with GPT-4...")
            async for chunk in result.stream_text(delta=True):
                partial_text += chunk
                message_placeholder.markdown(partial_text)
            status.update(label="Complete!", state="complete")

        # Update message history
        filtered_messages = [msg for msg in result.new_messages() 
                           if not (hasattr(msg, 'parts') and 
                                 any(part.part_kind == 'user-prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)

        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )

async def main():
    st.title("OpenAI Speech Analysis Assistant")
    st.write("""I can help you with:
    - Transcribing audio in multiple languages
    - Answering questions about transcriptions
    - Analyzing content with GPT-4
    """)
    
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
    user_input = st.chat_input(f"Ask a question")

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