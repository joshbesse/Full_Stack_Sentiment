from rest_framework.decorators import api_view
from rest_framework.response import Response 
from .models import Analysis
from sentiment_code import SentimentAnalysisFacade

facade = SentimentAnalysisFacade()

@api_view(["POST"])
def analyze_text(request):
    text = request.data.get("text")
    analyzer_type = request.data.get("analyzer_type", "basic")

    facade.select_analyzer(analyzer_type)
    result = facade.analyze_text(text)

    analysis_result = Analysis.objects.create(
        text=text, 
        sentiment=result.get_sentiment(), 
        score=result.get_score()
    )

    return Response({
        'sentiment': result.get_sentiment(),
        'score': result.get_score()
    })


