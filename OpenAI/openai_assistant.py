from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from openai import OpenAI
from supabase import Client
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel

@dataclass
class OpenAIAssistantDeps:
    supabase: Client
    openai_client: OpenAI
    whisper_models: Dict[str, torch.nn.Module]
    whisper_processors: Dict[str, object]
    openai_models: Dict[str, str]

async def process_with_gpt4(
    client: OpenAI,
    text: str,
    instruction: str,
    models: Dict[str, str]
) -> str:
    """Process text with GPT-4."""
    response = await client.chat.completions.create(
        model=models['chat'],
        messages=[{
            "role": "system",
            "content": instruction
        }, {
            "role": "user",
            "content": text
        }],
        temperature=0.7
    )
    return response.choices[0].message.content

async def get_embedding(text: str, client: OpenAI, models: Dict[str, str]) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await client.embeddings.create(
            model=models['embedding'],
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536

# Initialize the model with GPT-4
model = OpenAIModel("gpt-4")

system_prompt = """
You are an expert AI assistant powered by GPT-4 that can:
1. Process and understand transcriptions in multiple languages
2. Answer questions about the transcribed content
3. Provide detailed analysis and insights

When processing requests:
1. Use the fine-tuned Whisper models for transcription
2. Process and analyze with GPT-4
3. Maintain context and provide comprehensive responses
"""

openai_assistant = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=OpenAIAssistantDeps,
    retries=2
)

@openai_assistant.tool
async def transcribe_audio(
    ctx: RunContext[OpenAIAssistantDeps],
    audio_path: str,
    language_code: str
) -> str:
    """Transcribe audio using fine-tuned Whisper models."""
    try:
        if language_code not in ctx.deps.whisper_models:
            return f"Unsupported language code: {language_code}"
            
        # Get transcription using fine-tuned model
        model = ctx.deps.whisper_models[language_code]
        processor = ctx.deps.whisper_processors[language_code]
        audio_input = processor(audio_path, return_tensors="pt", sampling_rate=16000)
        
        with torch.no_grad():
            output = model.generate(**audio_input)
        transcription = processor.batch_decode(output, skip_special_tokens=True)[0]
            
        return transcription
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

@openai_assistant.tool
async def analyze_content(
    ctx: RunContext[OpenAIAssistantDeps],
    content: str,
    analysis_type: str = "general"
) -> str:
    """Analyze content with GPT-4."""
    try:
        instruction = {
            "general": "Analyze this content and provide key insights, main points, and important details.",
            "sentiment": "Analyze the sentiment, tone, and emotional content of this text.",
            "summary": "Provide a comprehensive summary of this content."
        }.get(analysis_type, "Analyze this content thoroughly.")
        
        analysis = await process_with_gpt4(
            ctx.deps.openai_client,
            content,
            instruction,
            ctx.deps.openai_models
        )
        
        return analysis
        
    except Exception as e:
        return f"Error analyzing content: {str(e)}"

@openai_assistant.tool
async def answer_question(
    ctx: RunContext[OpenAIAssistantDeps],
    question: str,
    context: str
) -> str:
    """Answer questions using GPT-4."""
    try:
        instruction = "Answer the question based on the provided context. Be specific and accurate."
        prompt = f"Context: {context}\n\nQuestion: {question}"
        
        answer = await process_with_gpt4(
            ctx.deps.openai_client,
            prompt,
            instruction,
            ctx.deps.openai_models
        )
        
        return answer
        
    except Exception as e:
        return f"Error answering question: {str(e)}"

@openai_assistant.tool
async def store_transcription(
    ctx: RunContext[OpenAIAssistantDeps],
    transcription: str,
    language_code: str,
    metadata: Optional[Dict] = None
) -> bool:
    """Store transcription in the database."""
    try:
        embedding = await get_embedding(
            transcription,
            ctx.deps.openai_client,
            ctx.deps.openai_models
        )
        
        data = {
            'content': transcription,
            'language': language_code,
            'embedding': embedding,
            'metadata': metadata or {}
        }
        
        result = ctx.deps.supabase.from_('transcriptions').insert(data).execute()
        return bool(result.data)
        
    except Exception as e:
        print(f"Error storing transcription: {e}")
        return False

@openai_assistant.tool
async def search_transcriptions(
    ctx: RunContext[OpenAIAssistantDeps],
    query: str,
    language_code: Optional[str] = None
) -> List[Dict]:
    """Search through stored transcriptions."""
    try:
        query_embedding = await get_embedding(
            query,
            ctx.deps.openai_client,
            ctx.deps.openai_models
        )
        
        match_params = {
            'query_embedding': query_embedding,
            'match_count': 5
        }
        
        if language_code:
            match_params['filter'] = {'language': language_code}
        
        result = ctx.deps.supabase.rpc(
            'match_transcriptions',
            match_params
        ).execute()
        
        return result.data or []
        
    except Exception as e:
        print(f"Error searching transcriptions: {e}")
        return []

@openai_assistant.tool
async def analyze_transcription(
    ctx: RunContext[OpenAIAssistantDeps],
    transcription: str,
    language_code: str
) -> Dict:
    """Analyze a transcription for key information."""
    try:
        analysis = await process_with_gpt4(
            ctx.deps.openai_client,
            transcription,
            "Analyze this transcription for key points, sentiment, and main topics.",
            ctx.deps.openai_models
        )
        
        return {
            'language': language_code,
            'analysis': analysis,
            'length': len(transcription.split())
        }
        
    except Exception as e:
        print(f"Error analyzing transcription: {e}")
        return {
            'error': str(e),
            'language': language_code
        } 