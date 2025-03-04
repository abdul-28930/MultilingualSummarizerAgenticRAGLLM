-- Enable vector extension
create extension if not exists vector;

-- Create documents table
create table documents (
    id bigint primary key generated always as identity,
    content text,
    embedding vector(1536),
    metadata jsonb,
    keywords text[],
    chunk_index integer,
    total_chunks integer,
    created_at timestamp with time zone
);

-- Create function for similarity search
create or replace function match_documents (
    query_embedding vector(1536),
    match_count int
) returns table (
    id bigint,
    content text,
    keywords text[],
    similarity float
)
language plpgsql
as $$
begin
    return query
    select
        id,
        content,
        keywords,
        1 - (documents.embedding <=> query_embedding) as similarity
    from documents
    order by documents.embedding <=> query_embedding
    limit match_count;
end;
$$;