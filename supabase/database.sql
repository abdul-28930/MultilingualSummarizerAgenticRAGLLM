-- Enable the pgvector extension for embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Create transcriptions table
CREATE TABLE transcriptions (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    language VARCHAR(10) NOT NULL,
    embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW()),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT TIMEZONE('utc', NOW())
);

-- Create function to match transcriptions by embedding similarity
CREATE OR REPLACE FUNCTION match_transcriptions(
    query_embedding vector(1536),
    match_count int DEFAULT 5,
    filter jsonb DEFAULT '{}'
)
RETURNS TABLE (
    id bigint,
    content text,
    language varchar(10),
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        t.id,
        t.content,
        t.language,
        1 - (t.embedding <=> query_embedding) as similarity
    FROM transcriptions t
    WHERE
        CASE
            WHEN filter->>'language' IS NOT NULL
            THEN t.language = filter->>'language'
            ELSE true
        END
    ORDER BY t.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Create index for faster similarity searches
CREATE INDEX transcriptions_embedding_idx ON transcriptions
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100); 