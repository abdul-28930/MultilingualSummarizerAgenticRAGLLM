// API service for communicating with the backend

const API_BASE_URL = 'http://localhost:5000/api';

// Transcribe audio file
export const transcribeAudio = async (file, language) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('language', language);

    const response = await fetch(`${API_BASE_URL}/transcribe`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to transcribe audio');
    }

    return await response.json();
  } catch (error) {
    console.error('Transcription error:', error);
    throw error;
  }
};

// Generate summary
export const generateSummary = async (text, type) => {
  try {
    const response = await fetch(`${API_BASE_URL}/summarize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        type, // 'extractive' or 'abstractive'
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to generate summary');
    }

    return await response.json();
  } catch (error) {
    console.error('Summarization error:', error);
    throw error;
  }
};

// Translate text
export const translateText = async (text, targetLanguage) => {
  try {
    const response = await fetch(`${API_BASE_URL}/translate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text,
        target_language: targetLanguage,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to translate text');
    }

    return await response.json();
  } catch (error) {
    console.error('Translation error:', error);
    throw error;
  }
};

// Query the summary
export const querySummary = async (prompt, context) => {
  try {
    const response = await fetch(`${API_BASE_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt,
        context,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to query summary');
    }

    return await response.json();
  } catch (error) {
    console.error('Query error:', error);
    throw error;
  }
};

// Generate flowchart
export async function generateFlowchart(text, numKeywords = 10) {
  try {
    const response = await fetch(`${API_BASE_URL}/flowchart`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text, num_keywords: numKeywords }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to generate flowchart');
    }

    return await response.json();
  } catch (error) {
    console.error('Flowchart generation error:', error);
    throw error;
  }
}

// Health check
export const checkApiHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch (error) {
    console.error('API health check failed:', error);
    return false;
  }
};
