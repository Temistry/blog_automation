import requests
import os
from dotenv import load_dotenv
import json

# 환경 변수 로드
load_dotenv()

# WordPress 설정 가져오기
WORDPRESS_URL = os.getenv("WORDPRESS_URL")
WORDPRESS_USERNAME = os.getenv("WORDPRESS_USERNAME")
WORDPRESS_PASSWORD = os.getenv("WORDPRESS_PASSWORD")
WORDPRESS_APP_PASSWORD = os.getenv("WORDPRESS_APP_PASSWORD", "false").lower() == "true"

def test_create_post():
    """WordPress REST API를 사용하여 테스트 게시물을 생성합니다."""
    
    # API 엔드포인트
    api_url = f"{WORDPRESS_URL}/wp-json/wp/v2/posts"
    
    # 인증 정보
    auth = (WORDPRESS_USERNAME, WORDPRESS_PASSWORD)
    
    # 요청 헤더
    headers = {
        'Content-Type': 'application/json'
    }
    
    # 게시물 데이터
    post_data = {
        'title': '테스트 포스트 - 파이썬 API 테스트',
        'content': '이 글은 WordPress REST API 테스트를 위한 자동 생성 글입니다.',
        'status': 'draft',  # 임시 저장
        'categories': [13]  # Exp 카테고리 ID
    }
    
    print(f"WordPress REST API 테스트 시작...")
    print(f"URL: {api_url}")
    print(f"사용자: {WORDPRESS_USERNAME}")
    print(f"Application Password 사용: {WORDPRESS_APP_PASSWORD}")
    
    try:
        # REST API 요청
        response = requests.post(api_url, auth=auth, headers=headers, json=post_data)
        
        # 응답 처리
        if response.status_code in [200, 201]:
            post_id = response.json().get('id')
            print(f"성공! 포스트 ID: {post_id}")
            print(f"응답 상태 코드: {response.status_code}")
            print(f"응답 내용: {json.dumps(response.json(), indent=2, ensure_ascii=False)[:200]}...")
            return True
        else:
            print(f"실패: 상태 코드 {response.status_code}")
            print(f"응답 내용: {response.text}")
            return False
            
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return False

if __name__ == "__main__":
    test_create_post() 