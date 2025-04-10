#!/bin/bash

# 가상환경 디렉토리 이름 설정
VENV_NAME="blog_venv"

# 가상환경이 없으면 생성
if [ ! -d "$VENV_NAME" ]; then
    echo "가상환경 생성 중..."
    python3 -m venv $VENV_NAME
fi

# 가상환경 활성화
source $VENV_NAME/bin/activate

# 필요한 패키지 설치
echo "필요한 패키지 설치 중..."
pip install requests pandas numpy openai schedule PyGithub google-api-python-client python-dotenv textblob

# 환경변수 파일 확인
if [ ! -f ".env" ]; then
    echo "환경변수 파일(.env)이 없습니다. 샘플 파일을 생성합니다."
    cat > .env << EOL
OPENAI_API_KEY=your_openai_api_key
WORDPRESS_URL=your_wordpress_url
WORDPRESS_USERNAME=your_username
WORDPRESS_PASSWORD=your_password
WORDPRESS_APP_PASSWORD=false
GITHUB_TOKEN=your_github_token
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_cse_id
EOL
    echo ".env 파일을 수정하여 올바른 API 키와 인증 정보를 입력하세요."
    exit 1
fi

# prompts 디렉토리 확인 및 생성
if [ ! -d "prompts" ]; then
    echo "prompts 디렉토리 생성 중..."
    mkdir prompts
    
    # 기본 프롬프트 파일들 생성
    echo "기본 프롬프트 템플릿 생성 중..."
    echo "주제: {topic}\n설명: {description}\n\n다음 주제에 대한 기술 블로그 글을 작성해주세요." > prompts/content_draft_base.prompt
    echo "실제 개발 경험을 바탕으로 한 튜토리얼 형식으로 작성해주세요." > prompts/content_draft_tutorial.prompt
    echo "프레임워크/라이브러리의 장단점과 실제 사용 경험을 포함해주세요." > prompts/content_draft_review.prompt
    echo "개념을 쉽게 설명하고 실제 적용 사례를 포함해주세요." > prompts/content_draft_concept.prompt
    echo "{topic}에 대한 실용적인 코드 예제를 제공해주세요." > prompts/code_examples.prompt
    echo "다음 내용의 기술적 정확성을 검증해주세요:\n\n{content}" > prompts/fact_check.prompt
    echo "다음 내용의 가독성을 개선해주세요:\n\n{content}" > prompts/readability.prompt
    echo "다음 내용의 SEO를 최적화해주세요:\n주제: {topic}\n내용:\n{content}" > prompts/seo_optimization.prompt
fi

# 파이썬 스크립트 실행
echo "블로그 자동화 파이프라인 시작..."
python blog-automation-pipeline.py

# 가상환경 비활성화
deactivate