import requests
import json
import time
import os
import pandas as pd
import numpy as np
import openai
import schedule
from requests.auth import HTTPBasicAuth
from googleapiclient.discovery import build
from dotenv import load_dotenv
from datetime import datetime
from textblob import TextBlob
<<<<<<< Updated upstream
from prompts.tech_reviews.tech_reviews_prompts import tech_reviews_prompts
import feedparser
from config import RSS_FEEDS, OPENAI_MODELS, DEEPINFRA_MODELS, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, CODE_TEMPERATURE
import re
from PIL import Image
from io import BytesIO
from ai_clients import OpenAIClient, DeepInfraClient
=======
import warnings
>>>>>>> Stashed changes

# 환경 변수 로드
load_dotenv()

# API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WORDPRESS_URL = os.getenv("WORDPRESS_URL")
WORDPRESS_USERNAME = os.getenv("WORDPRESS_USERNAME")
WORDPRESS_PASSWORD = os.getenv("WORDPRESS_PASSWORD")
WORDPRESS_APP_PASSWORD = os.getenv("WORDPRESS_APP_PASSWORD", "false").lower() == "true"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# 사용 API 클라이언트 선택 - 기본값 : OpenAI, 필요시 .env에서 true 설정으로 DeepInfra 선택 가능
USE_DEEPINFRA = os.getenv("USE_DEEPINFRA", "false").lower() == "true"
if USE_DEEPINFRA:
    llm_client = DeepInfraClient()
    GPT_MODELS = DEEPINFRA_MODELS
    print("DeepInfra API 사용 중...")
else:
    llm_client = OpenAIClient()
    GPT_MODELS = OPENAI_MODELS
    print("OpenAI API 사용 중...")

# 이전 OpenAI 클라이언트 설정 (호환성을 위해 유지)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# GitHub API 헤더 설정
github_headers = {
    'Authorization': f'token {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

<<<<<<< Updated upstream
# GitHub API 직접 호출 함수
def search_github_repositories(query, sort='stars', order='desc'):
    url = f'https://api.github.com/search/repositories?q={query}&sort={sort}&order={order}'
    response = requests.get(url, headers=github_headers)
    if response.status_code == 200:
        return response.json()['items']
    else:
        print(f"GitHub API 오류: {response.status_code}")
        return []

def fetch_news_articles():
    print("뉴스 기사 가져오기 중...")
    articles = []
    
    for feed_url in RSS_FEEDS.get("tech_reviews", []):
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            articles.append({
                "source": "RSS Feed",
                "topic": entry.title,
                "description": entry.summary,
                "url": entry.link
            })
    
    return articles

def generate_content_ideas(category="programming"):
    print("콘텐츠 아이디어 발굴 중...")
    ideas = []
    
    # 이미 포스팅된 주제 로드
    posted_topics = set()
    if os.path.exists('posted_topics.txt'):
        with open('posted_topics.txt', 'r', encoding='utf-8') as f:
            posted_topics = set(line.strip() for line in f)
    
    if category == "programming":
        try:
            # GitHub 트렌드 분석 - 인기 있는 저장소 가져오기
            popular_languages = ["python", "javascript", "java", "go", "rust"]
            
            for lang in popular_languages:
                try:
                    repos = search_github_repositories(f"language:{lang}")
                    for i, repo in enumerate(repos[:2]):  # 각 언어별로 2개씩
                        if repo.get('name'):  # repo.name이 None이 아닌 경우에만
                            ideas.append({
                                "source": f"GitHub {lang.capitalize()} Trend",
                                "topic": repo['name'],
                                "description": repo.get('description', f"{repo['name']} 저장소에 대한 분석"),
                                "popularity": repo.get('stargazers_count', 0)
                            })
                        if i >= 1:  # 각 언어당 최대 2개 저장소만
                            break
                except Exception as e:
                    print(f"GitHub {lang} 트렌드 검색 중 오류: {str(e)}")
                    continue
            
            # Google 검색 트렌드 분석
            try:
                search_service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
                search_terms = ["python tutorial", "javascript framework", "web development", "machine learning", "devops"]
                
                for term in search_terms:
                    try:
                        results = search_service.cse().list(q=term, cx=GOOGLE_CSE_ID, num=3).execute()
                        for item in results.get('items', []):
                            if item['title']:  # item['title']이 None이 아닌 경우에만
                                ideas.append({
                                    "source": "Google Search",
                                    "topic": item['title'],
                                    "description": item['snippet'],
                                    "url": item['link']
                                })
                    except Exception as e:
                        print(f"Google 검색 '{term}' 중 오류: {str(e)}")
                        continue
            except Exception as e:
                print(f"Google 검색 API 초기화 중 오류: {str(e)}")
        
        except Exception as e:
            print(f"콘텐츠 아이디어 발굴 중 오류: {str(e)}")
            # 오류 발생 시 기본 아이디어 추가
            ideas.append({
                "source": "Default",
                "topic": "Python 비동기 프로그래밍 가이드",
                "description": "파이썬에서 비동기 프로그래밍을 구현하는 방법과 asyncio 라이브러리 활용법",
                "popularity": 100
            })
    else:
        articles = fetch_news_articles()
        ideas.extend(articles)
    
    # 이미 포스팅된 주제 제외 및 None 주제 필터링
    ideas = [idea for idea in ideas if idea['topic'] not in posted_topics and idea['topic'] is not None]
    
    # 아이디어를 CSV로 저장
    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame(ideas)
    df.to_csv('data/content_ideas.csv', index=False)
    print(f"{len(ideas)}개의 콘텐츠 아이디어 발굴 완료")
    return ideas
=======
# urllib3 경고 메시지 무시
warnings.filterwarnings('ignore', message='.*OpenSSL.*')
>>>>>>> Stashed changes

# 템플릿 타입 결정 함수
def get_template_type(topic):
    """주제에 따라 적절한 템플릿 타입을 반환합니다."""
    topic_lower = topic.lower()
    if "tutorial" in topic_lower or "guide" in topic_lower or "방법" in topic_lower or "하는 법" in topic_lower or "되고싶다면" in topic_lower:
        return "tutorial"
    elif "framework" in topic_lower or "library" in topic_lower or "도구" in topic_lower or "플랫폼" in topic_lower:
        return "review"
    else:
        return "concept"

# 2. 콘텐츠 구조화 및 초안 작성 함수
def fetch_latest_articles(feed_urls):
    """여러 RSS 피드에서 최신 기사를 가져옵니다."""
    articles = []
    for feed_url in feed_urls:
        feed = feedparser.parse(feed_url)
        if feed.entries:
            latest_entry = feed.entries[0]
            articles.append((latest_entry.title, latest_entry.summary))
    return articles

def sanitize_filename(filename):
    """파일 이름에서 특수문자를 제거하고 안전한 이름으로 변환합니다."""
    # HTML 엔티티를 일반 문자로 변환
    filename = filename.replace('&#8217;', "'")
    filename = filename.replace('&#8216;', "'")
    filename = filename.replace('&#8220;', '"')
    filename = filename.replace('&#8221;', '"')
    
    # 특수문자를 제거하고 공백을 언더스코어로 변환
    filename = re.sub(r'[^\w\s-]', '', filename)
    filename = re.sub(r'[-\s]+', '_', filename)
    
    # 파일 이름이 너무 길면 자르기
    if len(filename) > 100:
        filename = filename[:100]
    
    return filename.lower()

def download_and_upload_image(image_url, post_id):
    """이미지를 다운로드하고 WordPress에 업로드합니다."""
    try:
        # 이미지 다운로드
        response = requests.get(image_url)
        if response.status_code != 200:
            print(f"이미지 다운로드 실패: {image_url}")
            return None
        
        # 이미지 데이터
        image_data = response.content
        
        # 이미지 파일 이름 추출
        filename = image_url.split('/')[-1]
        if '?' in filename:
            filename = filename.split('?')[0]
        
        # WordPress 미디어 업로드 URL
        upload_url = f"{WORDPRESS_URL}/wp-json/wp/v2/media"
        
        # 파일 확장자 확인
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # 이미지 형식 확인 및 변환
            img = Image.open(BytesIO(image_data))
            filename = f"{filename.split('.')[0]}.jpg"
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='JPEG')
            image_data = img_byte_arr.getvalue()
        
        # 파일 업로드
        files = {
            'file': (filename, image_data)
        }
        
        # 인증 헤더 설정
        if WORDPRESS_APP_PASSWORD:
            auth = (WORDPRESS_USERNAME, WORDPRESS_PASSWORD)
        else:
            auth = (WORDPRESS_USERNAME, WORDPRESS_PASSWORD)
        
        # 업로드 요청
        response = requests.post(
            upload_url,
            auth=auth,
            files=files,
            data={
                'post': post_id,
                'title': filename,
                'caption': f'Featured image for post {post_id}'
            }
        )
        
        if response.status_code in [200, 201]:
            media_data = response.json()
            return media_data['id']
        else:
            print(f"이미지 업로드 실패: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"이미지 처리 중 오류 발생: {str(e)}")
        return None

def create_content_draft(topic, description, category):
    if topic is None:
        raise ValueError("Topic cannot be None")
    
    print(f"'{topic}' 주제로 콘텐츠 초안 작성 중...")
    
    if category == "programming":
        content = create_programming_draft(topic, description)
    elif category == "tech_reviews":
        content = create_tech_review_draft(topic, description)
    else:
        raise ValueError("지원하지 않는 카테고리입니다.")
    
    if content:
        # 이미지 URL 추출 (예: ![alt text](url) 형식)
        image_urls = re.findall(r'!\[.*?\]\((.*?)\)', content)
        
        # 콘텐츠 저장
        os.makedirs('drafts', exist_ok=True)
        safe_filename = sanitize_filename(topic)
        with open(f"drafts/{safe_filename}.md", "w", encoding="utf-8") as f:
            f.write(content)
        print(f"'{topic}' 콘텐츠 초안 작성 완료")
        
        # 이미지 URL 저장
        if image_urls:
            with open(f"drafts/{safe_filename}_images.txt", "w", encoding="utf-8") as f:
                for url in image_urls:
                    f.write(f"{url}\n")
    
    return content

def create_programming_draft(topic, description):
    prompt_path = 'prompts/programming/'
    base_prompt_file = 'content_draft_base.prompt'
    template_prompt_file = f'content_draft_{get_template_type(topic)}.prompt'
    
    # 프롬프트 파일 로드
    with open(f'{prompt_path}{base_prompt_file}', 'r', encoding='utf-8') as f:
        base_prompt = f.read().format(topic=topic, description=description)
    
    with open(f'{prompt_path}{template_prompt_file}', 'r', encoding='utf-8') as f:
        template_prompt = f.read()
    
    prompt = base_prompt + template_prompt
    
    # OpenAI API를 사용하여 콘텐츠 초안 생성
    try:
        content = llm_client.chat_completion(
            model=GPT_MODELS["content_creation"],
            messages=[
                {"role": "system", "content": "당신은 경험 많은 개발자이자 기술 블로그 작가입니다. 실제 개발 경험을 바탕으로 한 글을 작성하며, 형식적이지 않고 개발자들이 실제로 작성한 것 같은 자연스러운 글을 씁니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        # 코드 예제 추가 요청
        with open(f'{prompt_path}code_examples.prompt', 'r', encoding='utf-8') as f:
            code_template = f.read()
        
        code_prompt = code_template.format(topic=topic)
        
        code_examples = llm_client.chat_completion(
            model=GPT_MODELS["code_examples"],
            messages=[
                {"role": "system", "content": "당신은 8년차 시니어 개발자로, 실무에서 많은 코드를 작성해왔습니다. 실용적이고 효율적인 코드를 작성하며, 과도한 주석이나 불필요한 설명 없이 깔끔한 코드를 제공합니다."},
                {"role": "user", "content": code_prompt}
            ],
            temperature=0.6
        )
        
        content += "\n\n## 실무에서 바로 쓸 수 있는 코드 예제\n\n" + code_examples
        
        # AI 티가 나는 패턴 제거
        content = clean_content(content)
        
    except Exception as e:
        print(f"콘텐츠 생성 중 오류 발생: {str(e)}")
        content = f"""
        # {topic}
        
        ## 소개
        {description}
        
        ## 이 글은 OpenAI API 오류로 인해 자동 생성되지 못했습니다.
        오류 메시지: {str(e)}
        """
    
    return content

def create_tech_review_draft(topic, description):
    # 영어 토픽 제목을 한글로 번역
    korean_title = translate_to_korean(topic)
    print(f"원본 제목: '{topic}'")
    print(f"번역된 제목: '{korean_title}'")
    
    # 번역된 한글 제목과 원본 영어 제목을 함께 사용
    base_prompt = tech_reviews_prompts["1_content_draft_base.prompt"].replace(
        "{{PRODUCT_NAME}}", f"{korean_title} ({topic})"
    )
    
    # OpenAI API를 사용하여 콘텐츠 초안 생성
    try:
        content = llm_client.chat_completion(
            model=GPT_MODELS["content_creation"],
            messages=[
                {"role": "system", "content": "당신은 최신 테크 기기에 대한 정보를 온라인에서 수집하고 정리하는 AI 리서처입니다."},
                {"role": "user", "content": base_prompt}
            ],
            temperature=0.7
        )
        
        # 제목을 한글 제목으로 대체
        content = content.replace(topic, korean_title)
        
        # AI 티가 나는 패턴 제거
        content = clean_content(content)
        
    except Exception as e:
        print(f"콘텐츠 생성 중 오류 발생: {str(e)}")
        content = f"""
        # {korean_title}
        
        ## 소개
        {description}
        
        ## 이 글은 OpenAI API 오류로 인해 자동 생성되지 못했습니다.
        오류 메시지: {str(e)}
        """
    
    return content

def translate_to_korean(text):
    """영어 텍스트를 한글로 번역합니다."""
    # 이미 한글이면 그대로 반환
    if any('\uAC00' <= char <= '\uD7A3' for char in text):
        return text
    
    try:
        translated = llm_client.chat_completion(
            model=GPT_MODELS["translation"],
            messages=[
                {"role": "system", "content": "당신은 영어를 한국어로 자연스럽게 번역하는 전문가입니다. 제품명이나 기술 용어는 적절히 한글과 영문 병기하되, 의미 전달이 자연스러운 한글 제목으로 번역하세요."},
                {"role": "user", "content": f"다음 텍스트를 한국어로 번역해주세요. 번역 결과만 출력하세요:\n\n{text}"}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        return translated.strip()
        
    except Exception as e:
        print(f"번역 중 오류 발생: {str(e)}")
        return text  # 오류 발생 시 원본 반환

def clean_content(content):
    """AI 티가 나는 패턴 제거"""
    content = content.replace("**", "")  # 불필요한 볼드체 제거
    content = content.replace("*", "")   # 불필요한 이탤릭체 제거
    content = content.replace("제 경험을 바탕으로", "제가 개발하면서 겪었던 경험으로는")
    content = content.replace("다음은", "여기 보면")
    content = content.replace("다음과 같습니다", "이렇게 작성할 수 있어요")
    content = content.replace("마치겠습니다", "마무리하겠습니다")
    content = content.replace("소개해 드리겠습니다", "살펴보겠습니다")
    return content

# 3. 품질 검증 및 개선 함수
def validate_and_improve_content(topic, content):
    print(f"'{topic}' 콘텐츠 품질 검증 및 개선 중...")
    
    try:
        # 기술적 정확성 검증 (프롬프트 파일에서 로드)
        with open('prompts/fact_check.prompt', 'r', encoding='utf-8') as f:
            fact_check_template = f.read()
        
        fact_check_prompt = fact_check_template.format(content=content[:4000])
        
        fact_check_result = llm_client.chat_completion(
            model=GPT_MODELS["fact_check"],
            messages=[
                {"role": "system", "content": "당신은 기술 검증 전문가입니다. 기술 콘텐츠의 정확성을 검증하고 객관적인 피드백을 제공합니다."},
                {"role": "user", "content": fact_check_prompt}
            ]
        )
        
        # 가독성 개선 (프롬프트 파일에서 로드)
        with open('prompts/readability.prompt', 'r', encoding='utf-8') as f:
            readability_template = f.read()
        
        readability_prompt = readability_template.format(content=content[:4000])
        
        improved_content = llm_client.chat_completion(
            model=GPT_MODELS["readability"],
            messages=[
                {"role": "system", "content": "당신은 경험 많은 개발 블로그 편집자입니다. 기술적 정확성을 유지하면서도 글을 더 자연스럽고 읽기 쉽게 만드는 전문가입니다."},
                {"role": "user", "content": readability_prompt}
            ],
            temperature=0.7
        )
        
        # 마크다운 포맷팅 정리 (AI가 생성한 티를 줄이기 위한 후처리)
        improved_content = improved_content.replace("**", "")  # 볼드체 제거
        improved_content = improved_content.replace("*", "")   # 이탤릭체 제거
        improved_content = improved_content.replace("\n\n\n", "\n\n")  # 과도한 줄바꿈 제거
        
        # SEO 최적화 (프롬프트 파일에서 로드)
        with open('prompts/seo_optimization.prompt', 'r', encoding='utf-8') as f:
            seo_template = f.read()
        
        seo_prompt = seo_template.format(content=improved_content[:4000], topic=topic)
        
        seo_suggestions = llm_client.chat_completion(
            model=GPT_MODELS["seo"],
            messages=[
                {"role": "system", "content": "당신은 개발자 블로그 SEO 전문가입니다. 기술 콘텐츠의 검색 엔진 최적화를 위한 실용적인 제안을 제공합니다."},
                {"role": "user", "content": seo_prompt}
            ]
        )
    
    except Exception as e:
        print(f"품질 검증 중 오류 발생: {str(e)}")
        fact_check_result = "API 오류로 검증 실패"
        improved_content = content
        seo_suggestions = "API 오류로 SEO 제안 실패"
    
    # 최종 개선된 콘텐츠 저장
    os.makedirs('improved', exist_ok=True)
    os.makedirs('meta', exist_ok=True)
    safe_filename = sanitize_filename(topic)
    with open(f"improved/{safe_filename}.md", "w", encoding="utf-8") as f:
        f.write(improved_content)
    
    # SEO 메타데이터 추출
    meta = {
        "original_title": topic,
        "fact_check": fact_check_result,
        "seo_suggestions": seo_suggestions
    }
    
    with open(f"meta/{safe_filename}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)
    
    print(f"'{topic}' 콘텐츠 품질 검증 및 개선 완료")
    return improved_content, meta

# 4. 발행 및 배포 자동화 함수
def publish_to_wordpress(topic, content, meta):
    print(f"'{topic}' 콘텐츠 워드프레스 발행 중...")
    
    # 원문 콘텐츠 유효성 검사
    if not content or content.strip() == "":
        print(f"[경고] '{topic}' 콘텐츠 발행 실패: 원문 콘텐츠가 비어 있습니다.")
        print(f"[경고] 원문 콘텐츠 없이 발행할 수 없습니다. 발행 프로세스를 중단합니다.")
        return None
        
    # 콘텐츠에 오류 메시지가 포함되어 있는지 확인
    if "이 글은 OpenAI API 오류로 인해 자동 생성되지 못했습니다" in content:
        print(f"[경고] '{topic}' 콘텐츠 발행 실패: 콘텐츠 생성 중 오류가 발생했습니다.")
        print(f"[경고] 오류가 포함된 콘텐츠는 발행하지 않습니다. 발행 프로세스를 중단합니다.")
        return None
    
    # REST API로 발행
    api_url = f"{WORDPRESS_URL}/wp-json/wp/v2/posts"
    
    # Application Password 사용 여부 확인
    use_app_password = WORDPRESS_APP_PASSWORD
    
    # 인증 방식 설정
    if use_app_password:
        # Application Password 사용 시 (권장)
        auth = (WORDPRESS_USERNAME, WORDPRESS_PASSWORD)
        print("Application Password 인증 방식 사용")
    else:
        # 기본 인증 (덜 안전함)
        auth = (WORDPRESS_USERNAME, WORDPRESS_PASSWORD)
        print("기본 인증 방식 사용 (Application Password 사용 권장)")
    
    # API 요청 헤더 설정
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    
    # 제목 설정 로직 개선
    seo_title = ""
    seo_suggestions = meta.get("seo_suggestions", "")
    
    # SEO 제안에서 제목 추출 시도 (더 정확한 패턴 매칭)
    if seo_suggestions:
        # 먼저 정확한 패턴 매칭 시도
        title_patterns = [
            r"제목:\s*(.+?)(?:\n|$)",
            r"title:\s*(.+?)(?:\n|$)",
            r"제목 제안:\s*(.+?)(?:\n|$)",
            r"최적화된 제목:\s*(.+?)(?:\n|$)",
            r"SEO 제목:\s*(.+?)(?:\n|$)"
        ]
        
        for pattern in title_patterns:
            matches = re.search(pattern, seo_suggestions, re.IGNORECASE)
            if matches:
                seo_title = matches.group(1).strip()
                # 마크다운 서식 제거
                seo_title = re.sub(r'[*#`]', '', seo_title).strip()
                break
        
        # 패턴 매칭 실패 시 줄별 검색
        if not seo_title:
            for line in seo_suggestions.split("\n"):
                if "제목:" in line or "title:" in line.lower() or "제목 제안:" in line:
                    seo_title = line.split(":", 1)[1].strip() if ":" in line else ""
                    # 마크다운 서식 제거
                    seo_title = re.sub(r'[*#`]', '', seo_title).strip()
                    break
    
    # 제목과 콘텐츠의 일관성 검증
    if seo_title and len(seo_title) >= 5 and "SEO" not in seo_title and "최적화" not in seo_title:
        # 기본 일관성 체크: 제목의 주요 키워드가 콘텐츠에 포함되는지
        # 제목에서 중요 키워드 추출 (2~4 단어)
        keywords = re.findall(r'\w+', seo_title.lower())
        keywords = [k for k in keywords if len(k) > 2]  # 짧은 단어 제외
        
        # 콘텐츠에 키워드 존재 여부 확인
        content_lower = content.lower()
        matched_keywords = [k for k in keywords if k in content_lower]
        
        # 키워드 일치율 계산
        match_rate = len(matched_keywords) / len(keywords) if keywords else 0
        
        # 일치율이 낮으면 원본 주제 사용
        if match_rate < 0.5:  # 50% 미만일 경우
            print(f"[경고] SEO 제안 제목({seo_title})이 콘텐츠와 불일치합니다(일치율: {match_rate:.2f})")
            print(f"[정보] 원본 주제를 사용합니다: '{topic}'")
            seo_title = topic
    else:
        # 제목이 유효하지 않으면 원본 주제 사용
        print(f"[정보] SEO 제안 제목이 유효하지 않아 원본 주제를 사용합니다: '{topic}'")
        seo_title = topic
    
    print(f"사용할 제목: '{seo_title}'")
    
    # 포스트 데이터 구성 - Tech 카테고리 ID로 변경
    tech_category_id = 15  # "Tech" 카테고리의 실제 ID로 변경 필요
    post_data = {
        'title': seo_title,
        'content': content,
        'status': 'publish',  # 바로 게시 상태로 발행
        'categories': [tech_category_id]  # Tech 카테고리 ID
    }
    
    # 포스트 발행
    post_id = None
    try:
        response = requests.post(api_url, auth=auth, headers=headers, json=post_data)
        
        if response.status_code in [200, 201]:
            post_id = response.json().get('id')
            print(f"'{seo_title}' 콘텐츠 워드프레스 발행 완료 (ID: {post_id})")
            
            # 이미지 업로드 및 첨부
            safe_filename = sanitize_filename(topic)
            image_file = f"drafts/{safe_filename}_images.txt"
            if os.path.exists(image_file):
                with open(image_file, 'r', encoding='utf-8') as f:
                    image_urls = [line.strip() for line in f if line.strip()]
                
                for image_url in image_urls:
                    media_id = download_and_upload_image(image_url, post_id)
                    if media_id:
                        print(f"이미지 업로드 성공 (ID: {media_id})")
            
            # 발행 성과 시 주제를 기록
            with open('posted_topics.txt', 'a', encoding='utf-8') as f:
                f.write(f"{topic}\n")
            
            return post_id
        else:
            print(f"발행 실패: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"발행 중 오류 발생: {str(e)}")
        return None

# 5. 성과 분석 및 피드백 수집 함수
def analyze_performance(post_id, days=7):
    print(f"포스트 ID {post_id}의 성과 분석 중...")
    
    # 여기서는 Google Analytics API 연동이 필요하지만, 예시로만 구현
    # 실제 구현시 Google Analytics API를 사용하여 트래픽 데이터 수집
    
    # 가상의 성과 데이터 생성 (예시)
    performance = {
        "post_id": post_id,
        "views": np.random.randint(100, 1000),
        "avg_time_on_page": np.random.randint(30, 300),
        "bounce_rate": np.random.uniform(0.3, 0.8),
        "conversion_rate": np.random.uniform(0.01, 0.1),
        "date": datetime.now().strftime("%Y-%m-%d")
    }
    
    # 댓글 감성 분석 (예시)
    # 실제로는 WordPress API를 통해 댓글을 수집하고 분석해야 함
    comments = ["정말 유용한 정보네요!", "코드 예제가 잘 작동하지 않습니다.", "더 자세한 설명이 필요합니다"]
    sentiment_scores = []
    
    for comment in comments:
        analysis = TextBlob(comment)
        sentiment_scores.append(analysis.sentiment.polarity)
    
    performance["comment_sentiment"] = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    
    # 성과 데이터 저장
    df = pd.DataFrame([performance])
    
    if os.path.exists('performance.csv'):
        existing_df = pd.read_csv('performance.csv')
        df = pd.concat([existing_df, df])
    
    df.to_csv('performance.csv', index=False)
    
    print(f"포스트 ID {post_id}의 성과 분석 완료")
    return performance

# 파이프라인 통합 함수
def run_content_pipeline(topic=None, description=None, category="programming"):
    print(f"블로그 콘텐츠 자동화 파이프라인 실행 중... 카테고리: {category}")
    
    # 디렉토리 생성
    os.makedirs('drafts', exist_ok=True)
    os.makedirs('improved', exist_ok=True)
    os.makedirs('meta', exist_ok=True)
    
    # 1. 아이디어 발굴 (topic이 제공되지 않은 경우)
    if not topic:
        ideas = generate_content_ideas(category)
        if ideas:
            # 인기도 기준으로 정렬하고 최상위 아이디어 선택
            sorted_ideas = sorted(ideas, key=lambda x: x.get('popularity', 0), reverse=True)
            idea = sorted_ideas[0]
            topic = idea['topic']
            description = idea['description']
    
    # 2. 콘텐츠 초안 작성
    content = create_content_draft(topic, description, category)
    
    # 콘텐츠 유효성 검사
    if not content or content.strip() == "":
        print(f"[경고] '{topic}' 콘텐츠 초안 생성 실패: 콘텐츠가 비어 있습니다.")
        print("[경고] 파이프라인을 중단합니다.")
        return None
    
    # 3. 품질 검증 및 개선
    improved_content, meta = validate_and_improve_content(topic, content)
    
    # 개선된 콘텐츠 유효성 검사
    if not improved_content or improved_content.strip() == "":
        print(f"[경고] '{topic}' 콘텐츠 개선 실패: 개선된 콘텐츠가 비어 있습니다.")
        print("[경고] 원본 콘텐츠를 사용하여 계속 진행합니다.")
        improved_content = content
    
    # 4. 워드프레스 발행
    post_id = publish_to_wordpress(topic, improved_content, meta)
    
    # 발행 결과 확인
    if post_id:
        # 5. 성과 분석 스케줄링 (7일 후)
        # 실제 구현에서는 schedule 라이브러리를 사용하여 향후 분석 일정을 잡을 수 있음
        print(f"포스트 ID {post_id}의 성과 분석이 7일 후로 예약되었습니다.")
        print("블로그 콘텐츠 자동화 파이프라인 실행 완료")
    else:
        print("[경고] 콘텐츠 발행에 실패했습니다. 파이프라인이 완전히 완료되지 않았습니다.")
    
    return post_id

# 정기적 실행 스케줄링 (매주 월요일과 목요일 오전 10시)
def schedule_pipeline():
<<<<<<< Updated upstream
    # 카테고리별로 스케줄링
    schedule.every().monday.at("10:00").do(run_content_pipeline, category="programming")
    schedule.every().tuesday.at("10:00").do(run_content_pipeline, category="tech_reviews")
    schedule.every().wednesday.at("10:00").do(run_content_pipeline, category="programming")
    schedule.every().thursday.at("23:56").do(run_content_pipeline, category="tech_reviews")
    schedule.every().friday.at("10:00").do(run_content_pipeline, category="programming")
    schedule.every().saturday.at("10:00").do(run_content_pipeline, category="tech_reviews")
    schedule.every().sunday.at("10:00").do(run_content_pipeline, category="programming")
=======
    print("스케줄러가 시작되었습니다.")
    print("다음 실행 일정:")
    print("- 매일 오전 10:00")
    print("- 수요일 18:35")
    
    schedule.every().monday.at("10:00").do(run_content_pipeline)
    schedule.every().tuesday.at("10:00").do(run_content_pipeline)
    schedule.every().wednesday.at("18:35").do(run_content_pipeline)
    schedule.every().thursday.at("10:00").do(run_content_pipeline)
    schedule.every().friday.at("10:00").do(run_content_pipeline)
    schedule.every().saturday.at("10:00").do(run_content_pipeline)
    schedule.every().sunday.at("10:00").do(run_content_pipeline)
>>>>>>> Stashed changes
    
    print("스케줄러가 실행 중입니다...")
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)
            # 매 분마다 . 을 출력하여 프로그램이 실행 중임을 표시
            print(".", end="", flush=True)
        except Exception as e:
            print(f"\n스케줄러 실행 중 오류 발생: {str(e)}")

# 프로그램 실행
if __name__ == "__main__":
<<<<<<< Updated upstream
    # 즉시 실행 (테스트용)
    #run_content_pipeline(
    #    topic=None,  # 테크 리뷰의 경우 topic과 description은 RSS에서 가져옴
    #    description=None,
    #    category="tech_reviews"
    #)
    
    # 정기적 실행 스케줄링
    schedule_pipeline()
=======
    try:
        print("블로그 자동화 파이프라인 시작...")
        # 정기적 실행 스케줄링
        schedule_pipeline()
    except KeyboardInterrupt:
        print("\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n예상치 못한 오류가 발생했습니다: {str(e)}")
    finally:
        print("프로그램이 종료되었습니다.")
>>>>>>> Stashed changes
