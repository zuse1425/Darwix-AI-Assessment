# suggestions/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import BlogContentSerializer

from .utils import (
    clean_blog_text,
    get_semantic_chunks,
    get_top_k_chunks,
    summarize_chunk,
    combine_summaries,
    generate_three_titles
)

class TitleSuggestionAPIView(APIView):
    """
    POST payload: { "content": "<raw blog post HTML or text>" }
    Response: { "titles": ["Title 1", "Title 2", "Title 3"] }
    """

    def post(self, request, *args, **kwargs):
        serializer = BlogContentSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        raw_blog = serializer.validated_data["content"]

        # 1. Clean the raw input once
        try:
            cleaned = clean_blog_text(raw_blog)
        except Exception as e:
            return Response(
                {"error": f"Cleaning failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # 2. Semantic chunking (on cleaned text)
        try:
            semantic_chunks = get_semantic_chunks(cleaned)
        except Exception as e:
            return Response(
                {"error": f"Semantic chunking failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # 3. Retrieve top 20 chunks (FAISS on cleaned text + semantic_chunks)
        try:
            top_chunks = get_top_k_chunks(cleaned, semantic_chunks, k=20)
        except Exception as e:
            return Response(
                {"error": f"FAISS retrieval failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # 4. Summarize each of those top 20 chunks
        chunk_summaries = []
        for idx, chunk in enumerate(top_chunks):
            try:
                summary = summarize_chunk(chunk)
            except Exception as e:
                return Response(
                    {"error": f"Summarization failed on chunk {idx+1}: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            chunk_summaries.append(summary)

        # 5. Combine those chunk summaries into a global summary
        try:
            global_summary = combine_summaries(chunk_summaries)
        except Exception as e:
            return Response(
                {"error": f"Global summarization failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # 6. Generate three titles from that global summary
        try:
            titles = generate_three_titles(global_summary)
        except Exception as e:
            return Response(
                {"error": f"Title generation failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response({"titles": titles}, status=status.HTTP_200_OK)
