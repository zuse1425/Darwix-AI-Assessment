# suggestions/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import BlogContentSerializer
from .utils import (
    get_recursive_chunks,
    summarize_chunk,
    combine_summaries,
    generate_three_titles
)

class TitleSuggestionAPIView(APIView):
    """
    POST payload: { "content": "<full blog post text>" }
    Response: { "titles": ["Title 1", "Title 2", "Title 3"] }
    """

    def post(self, request, *args, **kwargs):
        serializer = BlogContentSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        blog_text = serializer.validated_data["content"]

        # 1. Split into recursive chunks
        try:
            chunks = get_recursive_chunks(blog_text)
        except Exception as e:
            return Response(
                {"error": f"Chunking failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # 2. Summarize each chunk
        chunk_summaries = []
        for idx, chunk in enumerate(chunks):
            try:
                summary = summarize_chunk(chunk)
            except Exception as e:
                return Response(
                    {"error": f"Summarization failed on chunk {idx+1}: {str(e)}"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            chunk_summaries.append(summary)

        # 3. Combine into a global summary
        try:
            global_summary = combine_summaries(chunk_summaries)
        except Exception as e:
            return Response(
                {"error": f"Global summarization failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # 4. Generate three titles
        try:
            titles = generate_three_titles(global_summary)
        except Exception as e:
            return Response(
                {"error": f"Title generation failed: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        return Response({"titles": titles}, status=status.HTTP_200_OK)
