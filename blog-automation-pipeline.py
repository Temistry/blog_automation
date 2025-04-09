import requests
import json
import time
import os
import pandas as pd
import numpy as np
import openai
import schedule
from requests.auth import HTTPBasicAuth
from github import Github
from googleapiclient.discovery import build
from dotenv import load_dotenv
from datetime import datetime
from textblob import TextBlob

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

# OpenAI 클라이언트 설정
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# GitHub 클라이언트 설정
github_client = Github(GITHUB_TOKEN)

# API 모델 설정 (파일 상단에 추가)
GPT_MODELS = {
    "content_creation": "gpt-4o-mini",  # 콘텐츠 생성용 (고품질)
    "code_examples": "gpt-3.5-turbo",  # 코드 예제 생성용
    "fact_check": "gpt-3.5-turbo",  # 기술적 정확성 검증용 (정확도 중요)
    "readability": "gpt-3.5-turbo",  # 가독성 개선용
    "seo": "gpt-3.5-turbo"  # SEO 최적화용
}

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

# 1. 콘텐츠 아이디어 발굴 함수
def generate_content_ideas():
    print("콘텐츠 아이디어 발굴 중...")
    ideas = []
    
    # 이미 포스팅된 주제 로드
    posted_topics = set()
    if os.path.exists('posted_topics.txt'):
        with open('posted_topics.txt', 'r', encoding='utf-8') as f:
            posted_topics = set(line.strip() for line in f)
    
    try:
        # GitHub 트렌드 분석 - 인기 있는 저장소 가져오기
        # get_trending_repositories 메서드가 없으므로 직접 구현
        # 언어별 인기 저장소 몇 개 가져오기
        popular_languages = ["python", "javascript", "java", "go", "rust"]
        
        for lang in popular_languages:
            try:
                repos = github_client.search_repositories(f"language:{lang}", "stars", "desc")
                for i, repo in enumerate(repos[:2]):  # 각 언어별로 2개씩
                    ideas.append({
                        "source": f"GitHub {lang.capitalize()} Trend",
                        "topic": repo.name,
                        "description": repo.description or f"{repo.name} 저장소에 대한 분석",
                        "popularity": repo.stargazers_count
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
    
    # 아이디어가 없으면 기본 아이디어 추가
    if not ideas:
        ideas.append({
            "source": "Default",
            "topic": "Python 비동기 프로그래밍 가이드",
            "description": "파이썬에서 비동기 프로그래밍을 구현하는 방법과 asyncio 라이브러리 활용법",
            "popularity": 100
        })
    
    # 이미 포스팅된 주제 제외
    ideas = [idea for idea in ideas if idea['topic'] not in posted_topics]
    
    # 아이디어를 CSV로 저장
    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame(ideas)
    df.to_csv('data/content_ideas.csv', index=False)
    print(f"{len(ideas)}개의 콘텐츠 아이디어 발굴 완료")
    return ideas

# 2. 콘텐츠 구조화 및 초안 작성 함수
def create_content_draft(topic, description):
    print(f"'{topic}' 주제로 콘텐츠 초안 작성 중...")
    
    # 프롬프트 파일 로드
    with open('prompts/content_draft_base.prompt', 'r', encoding='utf-8') as f:
        base_prompt = f.read().format(topic=topic, description=description)
    
    template = get_template_type(topic)
    
    with open(f'prompts/content_draft_{template}.prompt', 'r', encoding='utf-8') as f:
        template_prompt = f.read()
    
    prompt = base_prompt + template_prompt
    
    # OpenAI API를 사용하여 콘텐츠 초안 생성
    try:
        response = client.chat.completions.create(
            model=GPT_MODELS["content_creation"],
            messages=[
                {"role": "system", "content": "당신은 경험 많은 개발자이자 기술 블로그 작가입니다. 실제 개발 경험을 바탕으로 한 글을 작성하며, 형식적이지 않고 개발자들이 실제로 작성한 것 같은 자연스러운 글을 씁니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7  # 약간의 창의성 추가
        )
        
        content = response.choices[0].message.content
        
        # 코드 예제 추가 요청 (프롬프트 파일에서 로드)
        with open('prompts/code_examples.prompt', 'r', encoding='utf-8') as f:
            code_template = f.read()
        
        code_prompt = code_template.format(topic=topic)
        
        code_response = client.chat.completions.create(
            model=GPT_MODELS["code_examples"],
            messages=[
                {"role": "system", "content": "당신은 8년차 시니어 개발자로, 실무에서 많은 코드를 작성해왔습니다. 실용적이고 효율적인 코드를 작성하며, 과도한 주석이나 불필요한 설명 없이 깔끔한 코드를 제공합니다."},
                {"role": "user", "content": code_prompt}
            ],
            temperature=0.6
        )
        
        code_examples = code_response.choices[0].message.content
        
        # 최종 콘텐츠 조합 및 후처리
        final_content = content + "\n\n## 실무에서 바로 쓸 수 있는 코드 예제\n\n" + code_examples
        
        # AI 티가 나는 패턴 제거
        final_content = final_content.replace("**", "")  # 불필요한 볼드체 제거
        final_content = final_content.replace("*", "")   # 불필요한 이탤릭체 제거
        
        # 불필요한 정형화된 문구 제거 
        final_content = final_content.replace("제 경험을 바탕으로", "제가 개발하면서 겪었던 경험으로는")
        final_content = final_content.replace("다음은", "여기 보면")
        final_content = final_content.replace("다음과 같습니다", "이렇게 작성할 수 있어요")
        final_content = final_content.replace("마치겠습니다", "마무리하겠습니다")
        final_content = final_content.replace("소개해 드리겠습니다", "살펴보겠습니다")
        
    except Exception as e:
        print(f"콘텐츠 생성 중 오류 발생: {str(e)}")
        # 간단한 예시 콘텐츠 생성
        final_content = f"""
        # {topic}
        
        ## 소개
        {description}
        
        ## 이 글은 OpenAI API 오류로 인해 자동 생성되지 못했습니다.
        오류 메시지: {str(e)}
        """
    
    # 콘텐츠 저장
    os.makedirs('drafts', exist_ok=True)
    with open(f"drafts/{topic.replace(' ', '_').lower()}.md", "w", encoding="utf-8") as f:
        f.write(final_content)
    
    print(f"'{topic}' 콘텐츠 초안 작성 완료")
    return final_content

# 3. 품질 검증 및 개선 함수
def validate_and_improve_content(topic, content):
    print(f"'{topic}' 콘텐츠 품질 검증 및 개선 중...")
    
    try:
        # 기술적 정확성 검증 (프롬프트 파일에서 로드)
        with open('prompts/fact_check.prompt', 'r', encoding='utf-8') as f:
            fact_check_template = f.read()
        
        fact_check_prompt = fact_check_template.format(content=content[:4000])
        
        fact_check_response = client.chat.completions.create(
            model=GPT_MODELS["fact_check"],
            messages=[
                {"role": "system", "content": "당신은 기술 검증 전문가입니다. 기술 콘텐츠의 정확성을 검증하고 객관적인 피드백을 제공합니다."},
                {"role": "user", "content": fact_check_prompt}
            ]
        )
        
        fact_check_result = fact_check_response.choices[0].message.content
        
        # 가독성 개선 (프롬프트 파일에서 로드)
        with open('prompts/readability.prompt', 'r', encoding='utf-8') as f:
            readability_template = f.read()
        
        readability_prompt = readability_template.format(content=content[:4000])
        
        readability_response = client.chat.completions.create(
            model=GPT_MODELS["readability"],
            messages=[
                {"role": "system", "content": "당신은 경험 많은 개발 블로그 편집자입니다. 기술적 정확성을 유지하면서도 글을 더 자연스럽고 읽기 쉽게 만드는 전문가입니다."},
                {"role": "user", "content": readability_prompt}
            ],
            temperature=0.7
        )
        
        improved_content = readability_response.choices[0].message.content
        
        # 마크다운 포맷팅 정리 (AI가 생성한 티를 줄이기 위한 후처리)
        improved_content = improved_content.replace("**", "")  # 볼드체 제거
        improved_content = improved_content.replace("*", "")   # 이탤릭체 제거
        improved_content = improved_content.replace("\n\n\n", "\n\n")  # 과도한 줄바꿈 제거
        
        # SEO 최적화 (프롬프트 파일에서 로드)
        with open('prompts/seo_optimization.prompt', 'r', encoding='utf-8') as f:
            seo_template = f.read()
        
        seo_prompt = seo_template.format(content=improved_content[:4000], topic=topic)
        
        seo_response = client.chat.completions.create(
            model=GPT_MODELS["seo"],
            messages=[
                {"role": "system", "content": "당신은 개발자 블로그 SEO 전문가입니다. 기술 콘텐츠의 검색 엔진 최적화를 위한 실용적인 제안을 제공합니다."},
                {"role": "user", "content": seo_prompt}
            ]
        )
        
        seo_suggestions = seo_response.choices[0].message.content
    
    except Exception as e:
        print(f"품질 검증 중 오류 발생: {str(e)}")
        fact_check_result = "API 오류로 검증 실패"
        improved_content = content
        seo_suggestions = "API 오류로 SEO 제안 실패"
    
    # 최종 개선된 콘텐츠 저장
    os.makedirs('improved', exist_ok=True)
    os.makedirs('meta', exist_ok=True)
    with open(f"improved/{topic.replace(' ', '_').lower()}.md", "w", encoding="utf-8") as f:
        f.write(improved_content)
    
    # SEO 메타데이터 추출
    meta = {
        "original_title": topic,
        "fact_check": fact_check_result,
        "seo_suggestions": seo_suggestions
    }
    
    with open(f"meta/{topic.replace(' ', '_').lower()}.json", "w", encoding="utf-8") as f:
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
    
    # 제목 설정 - SEO 메타데이터에서 제안된 제목 추출 로직 개선
    seo_title = ""
    seo_suggestions = meta.get("seo_suggestions", "")
    
    # SEO 제안에서 제목 추출 시도
    if seo_suggestions:
        for line in seo_suggestions.split("\n"):
            if "제목:" in line or "title:" in line.lower() or "제목 제안:" in line:
                seo_title = line.replace("제목:", "").replace("Title:", "").replace("title:", "").replace("제목 제안:", "").strip()
                # 마크다운 서식 제거 (**, ##, * 등)
                seo_title = seo_title.replace("*", "").replace("#", "").replace("`", "").strip()
                break
    
    # 제목이 유효하지 않으면 원본 주제 사용
    if not seo_title or len(seo_title) < 5 or "SEO" in seo_title or "최적화" in seo_title:
        print(f"[정보] SEO 제안 제목이 유효하지 않아 원본 주제를 사용합니다: '{topic}'")
        seo_title = topic
    
    print(f"사용할 제목: '{seo_title}'")
    
    # 포스트 데이터 구성 - Exp 카테고리 ID 직접 지정
    post_data = {
        'title': seo_title,
        'content': content,
        'status': 'publish',  # 바로 게시 상태로 발행
        'categories': [13]    # Exp 카테고리 ID
    }
    
    # API 호출
    try:
        headers = {'Content-Type': 'application/json'}
        response = requests.post(api_url, auth=auth, headers=headers, json=post_data)
        
        # 응답 처리
        if response.status_code in [200, 201]:  # 200 OK 또는 201 Created
            post_id = response.json().get('id')
            print(f"'{seo_title}' 콘텐츠 워드프레스 발행 완료 (ID: {post_id})")
            
            # 발행 성공 시 주제를 기록
            with open('posted_topics.txt', 'a', encoding='utf-8') as f:
                f.write(f"{topic}\n")
            
            return post_id
        else:
            print(f"발행 실패: {response.status_code} - {response.text}")
            print("WordPress 설정을 확인하세요. 원인:")
            print("1. REST API가 비활성화되어 있을 수 있습니다.")
            print("2. 사용자에게 글 작성 권한이 없을 수 있습니다.")
            
            if not use_app_password:
                print("3. Application Passwords 사용을 권장합니다:")
                print("   - WordPress 관리자 > 사용자 > 내 프로필 > Application Passwords에서 새 비밀번호 생성")
                print("   - .env 파일에 WORDPRESS_APP_PASSWORD=true 추가")
                print("   - WORDPRESS_PASSWORD에 생성된 애플리케이션 비밀번호 설정 (공백 포함)")
            
            # 오류 상세 정보 출력 시도
            try:
                error_detail = response.json()
                print(f"오류 세부 정보: {json.dumps(error_detail, ensure_ascii=False, indent=2)}")
            except:
                pass
            
            # 임시글로 시도
            print("임시글로 저장 시도...")
            post_data['status'] = 'draft'
            draft_response = requests.post(api_url, auth=auth, headers=headers, json=post_data)
            
            if draft_response.status_code in [200, 201]:
                draft_id = draft_response.json().get('id')
                print(f"'{seo_title}' 콘텐츠를 임시글로 저장했습니다 (ID: {draft_id})")
                return draft_id
            else:
                print(f"임시글 저장도 실패: {draft_response.status_code} - {draft_response.text}")
            
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
def run_content_pipeline(topic=None, description=None):
    print("블로그 콘텐츠 자동화 파이프라인 실행 중...")
    
    # 디렉토리 생성
    os.makedirs('drafts', exist_ok=True)
    os.makedirs('improved', exist_ok=True)
    os.makedirs('meta', exist_ok=True)
    
    # 1. 아이디어 발굴 (topic이 제공되지 않은 경우)
    if not topic:
        ideas = generate_content_ideas()
        if ideas:
            # 인기도 기준으로 정렬하고 최상위 아이디어 선택
            sorted_ideas = sorted(ideas, key=lambda x: x.get('popularity', 0), reverse=True)
            idea = sorted_ideas[0]
            topic = idea['topic']
            description = idea['description']
    
    # 2. 콘텐츠 초안 작성
    content = create_content_draft(topic, description)
    
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
    schedule.every().monday.at("10:00").do(run_content_pipeline)
    schedule.every().tuesday.at("10:00").do(run_content_pipeline)
    schedule.every().wednesday.at("18:35").do(run_content_pipeline)
    schedule.every().thursday.at("10:00").do(run_content_pipeline)
    schedule.every().friday.at("10:00").do(run_content_pipeline)
    schedule.every().saturday.at("10:00").do(run_content_pipeline)
    schedule.every().sunday.at("10:00").do(run_content_pipeline)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

# 프로그램 실행
if __name__ == "__main__":
    # 즉시 실행 (테스트용)
    #run_content_pipeline(
    #    topic="게임 개발자가 되고싶다면", 
    #    description="게임 개발자가 되고싶다면 어떻게 해야할까?"
    #)
    
    # 정기적 실행 스케줄링
    schedule_pipeline()
