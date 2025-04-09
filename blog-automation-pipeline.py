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

# 프로젝트 API 키 직접 할당 - 환경 변수에서 불러오는 대신 직접 할당
api_key = "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"

# OpenAI 클라이언트 설정 - 직접 API 키 지정
client = openai.OpenAI(api_key=api_key)

# GitHub 클라이언트 설정
github_client = Github(GITHUB_TOKEN)

# 1. 콘텐츠 아이디어 발굴 함수
def generate_content_ideas():
    print("콘텐츠 아이디어 발굴 중...")
    ideas = []
    
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
    
    # 아이디어를 CSV로 저장
    os.makedirs('data', exist_ok=True)
    df = pd.DataFrame(ideas)
    df.to_csv('data/content_ideas.csv', index=False)
    print(f"{len(ideas)}개의 콘텐츠 아이디어 발굴 완료")
    return ideas

# 2. 콘텐츠 구조화 및 초안 작성 함수
def create_content_draft(topic, description):
    print(f"'{topic}' 주제로 콘텐츠 초안 작성 중...")
    
    # 콘텐츠 구조 템플릿 선택 (주제에 따라 다른 템플릿 적용)
    if "tutorial" in topic.lower() or "guide" in topic.lower():
        template = "tutorial"
    elif "framework" in topic.lower() or "library" in topic.lower():
        template = "review"
    else:
        template = "concept"
    
    # 템플릿별 프롬프트 생성
    base_prompt = f"""
    당신은 5년 경력의 개발자로서 주니어 개발자들에게 도움이 되는 블로그를 운영하고 있습니다.
    '{topic}'에 대한 기술 블로그 글을 작성해주세요.
    
    다음 지침을 반드시 따라주세요:
    1. 자연스러운 대화체를 사용하고 실제 개발자가 작성한 것 같은 어투를 유지하세요.
    2. 개인적인 경험이나 실수담, 학습 과정에 대한 이야기를 간략히 포함하세요.
    3. 마크다운 형식을 사용하되, #, ##와 같은 마크다운 기호는 스페이스를 포함해 자연스럽게 작성하세요.
    4. 불필요한 반복이나 형식적인 문구는 피하세요.
    5. 긴 문단보다는 짧고 집중된 단락으로 작성하세요.
    6. 실무에서 직접 마주친 것 같은 구체적인 예시를 포함하세요.
    7. 읽는 개발자에게 직접 말하는 듯한 어투를 사용하세요.
    8. 너무 형식적이거나 교과서적인 설명보다는 실용적인 팁과 함께 설명하세요.
    9. 기술적 정확성은 유지하되, 복잡한 개념은 쉬운 비유로 설명하세요.
    10. 한국어로 작성하되, 자연스러운 개발 용어는 영어로 혼용하세요.
    
    주제 설명: {description}
    """
    
    if template == "tutorial":
        prompt = base_prompt + """
        다음과 같은 구조로 작성해주세요:
        
        1. 도입부 - 왜 이 기술을 배워야 하는지, 어떤 문제를 해결할 수 있는지 공감대 형성
        2. 배경 지식 - 필요한 사전 지식과 준비물 설명
        3. 단계별 구현 - 실습 코드와 함께 설명 (내가 직접 해봤던 경험 추가)
        4. 핵심 개념 - 튜토리얼 과정에서 알아야 할 주요 개념 설명
        5. 문제 해결 - 내가 겪었던 오류나 흔한 실수와 해결책
        6. 실제 활용 - 이 기술을 실무에서 어떻게 활용했는지 사례
        7. 다음 단계 - 더 배울 수 있는 자료나 심화 주제 제안
        
        코드 예제는 작동하는 실제 코드로 제공하고, 주석도 달아주세요.
        """
    elif template == "review":
        prompt = base_prompt + """
        다음과 같은 구조로 작성해주세요:
        
        1. 솔직한 첫인상 - 이 기술을 처음 접했을 때 느낌과 기대
        2. 주요 특징 - 직접 사용해보며 발견한 핵심 기능들
        3. 장단점 분석 - 실제 프로젝트에 적용해보며 느낀 좋은 점과 아쉬운 점
        4. 대안과 비교 - 유사한 다른 기술과 비교 (내가 모두 사용해본 것처럼)
        5. 실제 사례 - 이 기술을 활용한 프로젝트나 업무 사례
        6. 성능과 한계 - 실제 환경에서 테스트한 성능과 한계점
        7. 결론 - 어떤 상황에서 추천하는지, 나의 최종 평가
        
        코드 예제나 설정 방법도 포함해주세요.
        """
    else:  # concept
        prompt = base_prompt + """
        다음과 같은 구조로 작성해주세요:
        
        1. 개념 소개 - 이 개념을 처음 접했을 때의 경험과 중요성
        2. 핵심 원리 - 복잡한 개념을 일상적인 비유로 설명
        3. 발전 과정 - 이 개념이 어떻게 발전해왔는지 간략한 역사
        4. 실제 적용 - 이 개념이 실무에서 어떻게 활용되는지 사례
        5. 관련 도구 - 이 개념을 활용하는 주요 도구나 라이브러리
        6. 미래 전망 - 앞으로의 발전 가능성과 트렌드
        7. 시작하기 - 이 개념을 배우기 위한 실용적인 조언
        
        가능하면 간단한 코드 예제나 다이어그램으로 설명해주세요.
        """
    
    # OpenAI API를 사용하여 콘텐츠 초안 생성
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 경험 많은 개발자이자 기술 블로그 작가입니다. 실제 개발 경험을 바탕으로 한 글을 작성하며, 형식적이지 않고 개발자들이 실제로 작성한 것 같은 자연스러운 글을 씁니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7  # 약간의 창의성 추가
        )
        
        content = response.choices[0].message.content
        
        # 코드 예제 추가 요청 - 자연스러운 예제 코드 생성 
        code_prompt = f"""
        '{topic}'에 대한 실제 개발 현장에서 사용할 수 있는 코드 예제 3개를 작성해주세요.
        
        코드 예제는 다음과 같은 특징을 가져야 합니다:
        1. 초급, 중급, 고급 수준으로 구분하여 제공
        2. 각 예제는 실제 작동하는 코드여야 함
        3. 주석은 꼭 필요한 부분에만 달고, 실제 개발자가 작성한 것처럼 자연스럽게
        4. 불필요하게 완벽한 코드보다는 실무에서 자주 볼 수 있는 스타일로 작성
        5. 라이브러리나 프레임워크를 사용한다면 현업에서 많이 사용하는 최신 버전 기준으로 작성
        6. 실제 문제 해결 시나리오를 담은 코드 (가능하면 내가 실무에서 작성했던 코드처럼)
        
        마치 당신이 동료 개발자에게 실제 업무에서 사용할 수 있는 코드를 공유하는 것처럼 작성해주세요.
        """
        
        code_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 5년차 시니어 개발자로, 실무에서 많은 코드를 작성해왔습니다. 실용적이고 효율적인 코드를 작성하며, 과도한 주석이나 불필요한 설명 없이 깔끔한 코드를 제공합니다."},
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
        # 기술적 정확성 검증 (GPT를 사용한 팩트 체크)
        fact_check_prompt = f"""
        다음 개발자 블로그 콘텐츠의 기술적 정확성을 검증해주세요:
        
        {content[:4000]}...
        
        다음 사항을 확인해주세요:
        1. 코드나 기술 설명이 최신 버전과 관행에 맞는지
        2. 잘못된 정보나 오해의 소지가 있는 내용이 있는지
        3. 부정확하거나 불완전한 설명이 있는지
        4. 누락된 중요 정보가 있는지
        
        개선이 필요한 부분만 간략하게 지적해주세요.
        """
        
        fact_check_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 기술 검증 전문가입니다. 기술 콘텐츠의 정확성을 검증하고 객관적인 피드백을 제공합니다."},
                {"role": "user", "content": fact_check_prompt}
            ]
        )
        
        fact_check_result = fact_check_response.choices[0].message.content
        
        # 가독성 개선
        readability_prompt = f"""
        당신은 개발자 블로그 편집자입니다. 다음 글을 더 읽기 쉽고 자연스럽게 개선해주세요:

        {content[:4000]}...

        다음과 같이 개선해주세요:
        1. 딱딱한 문체를 자연스러운 대화체로 바꿔주세요 (마치 동료 개발자와 이야기하듯이)
        2. 너무 긴 문장은 짧고 명확한 문장으로 나눠주세요
        3. 불필요한 반복이나 형식적인 문구는 제거해주세요
        4. 전문 용어는 간단한 설명을 자연스럽게 추가해주세요
        5. 코드 예제의 주석이 있다면 실제 개발자가 작성한 것처럼 자연스럽게 수정해주세요
        6. 필요한 경우 실무 경험을 바탕으로 한 예시나 일화를 추가해주세요
        7. 마크다운 형식은 유지하되, 과도한 형식 표시(*, #)는 자연스럽게 조정해주세요
        8. 읽는 사람에게 직접 말을 걸듯이 대화체를 적절히 사용해주세요
        
        글의 내용과 구조는 최대한 보존하되, 더 자연스럽고 개발자가 직접 쓴 것처럼 보이게 해주세요.
        """
        
        readability_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
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
        
        # SEO 최적화
        seo_prompt = f"""
        다음 기술 블로그 콘텐츠에 대한 SEO 최적화 제안을 해주세요:

        {improved_content[:4000]}...

        다음 항목을 포함한 제안을 해주세요:

        1. SEO 제목 - 60자 이내의 검색 최적화된 제목 (원래 주제: {topic})
        2. 메타 설명 - 150-160자 이내의 검색 결과에 표시될 설명
        3. 주요 키워드 - 5-8개의 관련성 높은 키워드
        4. 소제목 개선 - 더 검색 친화적인 소제목 구조 제안
        5. 콘텐츠 개선 - 키워드 배치, 내부/외부 링크, 이미지 최적화 등에 대한 간략한 제안
        
        모든 제안은 실제 적용 가능하고 자연스러워야 합니다. 과도한 키워드 삽입은 피해주세요.
        """
        
        seo_response = client.chat.completions.create(
            model="gpt-3.5-turbo",
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
    schedule.every().thursday.at("10:00").do(run_content_pipeline)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

# 프로그램 실행
if __name__ == "__main__":
    # 즉시 실행 (테스트용)
    #run_content_pipeline(
    #    topic="파이썬 기초", 
    #    description="파이썬 기초 개념 설명"
    #)
    
    # 정기적 실행 스케줄링
    schedule_pipeline()
