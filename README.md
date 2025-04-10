# 블로그 자동화 파이프라인

완전 자동화된 블로그 콘텐츠 생성, 최적화 및 배포 시스템입니다.

## 주요 기능

- 콘텐츠 아이디어 자동 발굴 및 트렌드 분석
- AI 기반 콘텐츠 생성 및 개선
- SEO 최적화 및 품질 검증
- 워드프레스 자동 발행
- 성과 분석 및 피드백 수집

## AI 모델 사용

이 프로젝트는 OpenAI 또는 DeepInfra API를 사용할 수 있습니다:

- `.env` 파일의 `USE_DEEPINFRA=true/false` 설정으로 AI 제공자 선택
- 각 환경에 맞는 API 키 설정 필요 (`OPENAI_API_KEY` 또는 `DEEPINFRA_API_KEY`)

### 모델 구성

모델 설정은 `config.py`에서 관리됩니다:

```python
# OpenAI 모델
OPENAI_MODELS = {
    "content_creation": "gpt-4o-mini",
    "code_examples": "gpt-4o-mini",
    # ... 기타 모델 설정
}

# DeepInfra 모델
DEEPINFRA_MODELS = {
    "content_creation": "meta-llama/Meta-Llama-3-8B-Instruct",
    "code_examples": "meta-llama/Meta-Llama-3-8B-Instruct", 
    # ... 기타 모델 설정
}
```

## 설치 및 실행

1. 환경 변수 설정 (.env 파일)
2. 필요 패키지 설치: `pip install -r requirements.txt`
3. 실행: `python blog-automation-pipeline.py`
