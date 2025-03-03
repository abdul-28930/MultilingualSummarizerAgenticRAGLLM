from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
from openai import OpenAI
from supabase import Client
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel

@dataclass
class DeepSeekAssistantDeps:
    supabase: Client
    deepseek_client: OpenAI  # DeepSeek client for both translation and reasoning
    whisper_models: Dict[str, torch.nn.Module]
    whisper_processors: Dict[str, object]
    deepseek_models: Dict[str, str]  # Configuration for different DeepSeek models

async def translate_to_english(
    client: OpenAI,
    text: str,
    source_lang: str,
    models: Dict[str, str]
) -> str:
    """Translate any text to English using DeepSeek chat."""
    response = await client.chat.completions.create(
        model=models['chat'],  # Use DeepSeek chat model
        messages=[{
            "role": "system",
            "content": f"Translate the following {source_lang} text to English. Maintain the original meaning and tone."
        }, {
            "role": "user",
            "content": text
        }],
        temperature=0.3
    )
    return response.choices[0].message.content

async def translate_from_english(
    client: OpenAI,
    text: str,
    target_lang: str,
    models: Dict[str, str]
) -> str:
    """Translate English text back to target language using DeepSeek chat."""
    response = await client.chat.completions.create(
        model=models['chat'],  # Use DeepSeek chat model
        messages=[{
            "role": "system",
            "content": f"Translate the following English text to {target_lang}. Maintain the original meaning and tone."
        }, {
            "role": "user",
            "content": text
        }],
        temperature=0.3
    )
    return response.choices[0].message.content

async def process_with_deepseek(
    client: OpenAI,
    text: str,
    instruction: str,
    models: Dict[str, str]
) -> str:
    """Process English text with DeepSeek reasoner."""
    response = await client.chat.completions.create(
        model=models['reason'],  # Use DeepSeek reasoner model
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

# Configure OpenAI client for DeepSeek
client = OpenAI(
    api_key="your_deepseek_api_key",
    base_url="https://api.deepseek.com"
)

# Initialize the model with DeepSeek
model = OpenAIModel("deepseek-reasoner")

system_prompt = """
You are an expert multilingual AI assistant powered by DeepSeek-R1 that can:
1. Process and understand transcriptions in multiple languages
2. Answer questions about the transcribed content
3. Provide context and explanations across languages

When processing requests:
1. Always translate non-English input to English first
2. Process and analyze in English
3. Translate responses back to the user's language
4. Maintain context and nuance across translations
"""

deepseek_assistant = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=DeepSeekAssistantDeps,
    retries=2
)

@deepseek_assistant.tool
async def transcribe_audio(
    ctx: RunContext[DeepSeekAssistantDeps],
    audio_path: str,
    language_code: str
) -> str:
    """Transcribe audio and translate to English if needed."""
    try:
        if language_code not in ctx.deps.whisper_models:
            return f"Unsupported language code: {language_code}"
            
        # Get transcription in original language
        model = ctx.deps.whisper_models[language_code]
        processor = ctx.deps.whisper_processors[language_code]
        audio_input = processor(audio_path, return_tensors="pt", sampling_rate=16000)
        
        with torch.no_grad():
            output = model.generate(**audio_input)
        transcription = processor.batch_decode(output, skip_special_tokens=True)[0]
        
        # Translate to English if not already in English
        if language_code != 'en':
            transcription = await translate_to_english(ctx.deps.deepseek_client, transcription, language_code, ctx.deps.deepseek_models)
            
        return transcription
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"

@deepseek_assistant.tool
async def analyze_content(
    ctx: RunContext[DeepSeekAssistantDeps],
    content: str,
    language_code: str,
    analysis_type: str = "general"
) -> str:
    """Analyze content with translation flow."""
    try:
        # 1. Translate to English if needed
        english_content = content if language_code == 'en' else await translate_to_english(
            ctx.deps.deepseek_client, content, language_code, ctx.deps.deepseek_models
        )
        
        # 2. Process with DeepSeek-R1
        instruction = {
            "general": "Analyze this content and provide key insights, main points, and important details.",
            "sentiment": "Analyze the sentiment, tone, and emotional content of this text.",
            "summary": "Provide a comprehensive summary of this content."
        }.get(analysis_type, "Analyze this content thoroughly.")
        
        english_analysis = await process_with_deepseek(
            ctx.deps.deepseek_client,
            english_content,
            instruction,
            ctx.deps.deepseek_models
        )
        
        # 3. Translate back to user's language if needed
        if language_code != 'en':
            return await translate_from_english(
                ctx.deps.deepseek_client,
                english_analysis,
                language_code,
                ctx.deps.deepseek_models
            )
        return english_analysis
        
    except Exception as e:
        return f"Error analyzing content: {str(e)}"

@deepseek_assistant.tool
async def answer_question(
    ctx: RunContext[DeepSeekAssistantDeps],
    question: str,
    context: str,
    language_code: str
) -> str:
    """Answer questions with translation flow."""
    try:
        # 1. Translate question and context to English
        english_question = question if language_code == 'en' else await translate_to_english(
            ctx.deps.deepseek_client, question, language_code, ctx.deps.deepseek_models
        )
        english_context = context if language_code == 'en' else await translate_to_english(
            ctx.deps.deepseek_client, context, language_code, ctx.deps.deepseek_models
        )
        
        # 2. Process with DeepSeek-R1
        instruction = "Answer the question based on the provided context. Be specific and accurate."
        prompt = f"Context: {english_context}\n\nQuestion: {english_question}"
        
        english_answer = await process_with_deepseek(
            ctx.deps.deepseek_client,
            prompt,
            instruction,
            ctx.deps.deepseek_models
        )
        
        # 3. Translate answer back to user's language
        if language_code != 'en':
            return await translate_from_english(
                ctx.deps.deepseek_client,
                english_answer,
                language_code,
                ctx.deps.deepseek_models
            )
        return english_answer
        
    except Exception as e:
        return f"Error answering question: {str(e)}"

async def get_embedding(text: str, openai_client: OpenAI) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536

@deepseek_assistant.tool
async def store_transcription(
    ctx: RunContext[DeepSeekAssistantDeps],
    transcription: str,
    language_code: str,
    metadata: Optional[Dict] = None
) -> bool:
    """Store transcription in the database."""
    try:
        embedding = await get_embedding(transcription, ctx.deps.deepseek_client)
        
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

@deepseek_assistant.tool
async def search_transcriptions(
    ctx: RunContext[DeepSeekAssistantDeps],
    query: str,
    language_code: Optional[str] = None
) -> List[Dict]:
    """Search through stored transcriptions."""
    try:
        # Translate query to English if not in English
        if language_code and language_code != 'en':
            translation = await ctx.deps.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{
                    "role": "system",
                    "content": f"Translate the following {language_code} text to English:"
                }, {
                    "role": "user",
                    "content": query
                }]
            )
            query = translation.choices[0].message.content
            
        query_embedding = await get_embedding(query, ctx.deps.deepseek_client)
        
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

@deepseek_assistant.tool
async def analyze_transcription(
    ctx: RunContext[DeepSeekAssistantDeps],
    transcription: str,
    language_code: str
) -> Dict:
    """Analyze a transcription for key information."""
    try:
        # Translate to English if needed
        if language_code != 'en':
            translation = await ctx.deps.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{
                    "role": "system",
                    "content": f"Translate the following {language_code} text to English:"
                }, {
                    "role": "user",
                    "content": transcription
                }]
            )
            transcription = translation.choices[0].message.content
            
        # Use DeepSeek for analysis
        response = await ctx.deps.deepseek_client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "Analyze the following transcription for key points, sentiment, and main topics."},
                {"role": "user", "content": transcription}
            ]
        )
        
        analysis = response.choices[0].message.content
        
        # Translate analysis back if needed
        if language_code != 'en':
            translation = await ctx.deps.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[{
                    "role": "system",
                    "content": f"Translate the following analysis to {language_code}:"
                }, {
                    "role": "user",
                    "content": analysis
                }]
            )
            analysis = translation.choices[0].message.content
        
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