RSS_FEEDS = {
    "tech_reviews": [
        "https://www.theverge.com/rss/index.xml",
        "https://www.wired.com/feed/category/gear/latest/rss"
    ]
}

# API 모델 설정
OPENAI_MODELS = {
    "content_creation": "gpt-4o-mini",  # 콘텐츠 생성용 (고품질)
    "code_examples": "gpt-4o-mini",  # 코드 예제 생성용
    "fact_check": "gpt-4o-mini",  # 기술적 정확성 검증용 (정확도 중요)
    "readability": "gpt-4o-mini",  # 가독성 개선용
    "seo": "gpt-4o-mini",  # SEO 최적화용
    "translation": "gpt-4o-mini"  # 번역용
}

# DeepInfra 모델 설정
# DeepInfra에서 지원하는 모델 이름으로 설정
DEEPINFRA_MODELS = {
    "content_creation": "google/gemma-3-4b-it", 
    "code_examples": "google/gemma-3-4b-it",
    "fact_check": "google/gemma-3-4b-it",
    "readability": "google/gemma-3-4b-it",
    "seo": "google/gemma-3-4b-it",
    "translation": "google/gemma-3-4b-it"
}

# 기타 설정 값들
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7
CODE_TEMPERATURE = 0.6 