#!/bin/bash

# WordPress 설정
WP_URL="https://gameblit.kr"
WP_USER="gameblitplay"
WP_APP_PASSWORD="QcUM tFOv CMCR sR3z k0H9 ml4a"  # Application Password

echo "WordPress REST API 테스트 시작..."
echo "URL: $WP_URL"
echo "사용자: $WP_USER"

# 테스트 포스트 생성
curl -X POST "$WP_URL/wp-json/wp/v2/posts" \
  -u "$WP_USER:$WP_APP_PASSWORD" \
  -H "Content-Type: application/json" \
  -d '{"title":"테스트 포스트 - curl 테스트", "content":"이 글은 WordPress REST API curl 테스트를 위한 자동 생성 글입니다.", "status":"draft", "categories":[13]}'

echo -e "\n테스트 완료" 