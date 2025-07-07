#!/usr/bin/env python3
"""
vLLM 서버와 MCP 기능 테스트 스크립트
"""

import requests
import json
import time

def test_vllm_connection():
    """vLLM 서버 연결 테스트"""
    try:
        response = requests.get("http://localhost:8000/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ vLLM 서버 연결 성공")
            models = response.json()
            for model in models['data']:
                print(f"   - 모델: {model['id']}")
                print(f"   - 최대 길이: {model['max_model_len']}")
            return True
        else:
            print("❌ vLLM 서버 응답 오류")
            return False
    except Exception as e:
        print(f"❌ vLLM 서버 연결 실패: {e}")
        return False

def test_vllm_chat():
    """vLLM 채팅 기능 테스트"""
    try:
        url = "http://localhost:8000/v1/chat/completions"
        data = {
            "model": "Qwen/Qwen3-8B",
            "messages": [
                {"role": "user", "content": "안녕하세요! 간단한 인사말을 해주세요."}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = requests.post(url, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print("✅ vLLM 채팅 테스트 성공")
            print(f"응답: {result['choices'][0]['message']['content']}")
            
            # reasoning 기능 확인
            if 'reasoning_content' in result['choices'][0]['message']:
                print("✅ Reasoning 기능 활성화됨")
                print(f"추론 과정: {result['choices'][0]['message']['reasoning_content']}")
            
            return True
        else:
            print(f"❌ vLLM 채팅 테스트 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ vLLM 채팅 테스트 실패: {e}")
        return False

def test_weather_query():
    """날씨 관련 질문 테스트"""
    try:
        url = "http://localhost:8000/v1/chat/completions"
        data = {
            "model": "Qwen/Qwen3-8B",
            "messages": [
                {"role": "system", "content": "당신은 한국 날씨 정보를 제공하는 AI 어시스턴트입니다. 사용자가 날씨 정보를 요청하면 korea_weather MCP 서버의 도구들을 사용하여 날씨 정보를 조회합니다."},
                {"role": "user", "content": "서울 날씨 알려줘"}
            ],
            "max_tokens": 200,
            "temperature": 0.7
        }
        
        response = requests.post(url, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print("✅ 날씨 질문 테스트 성공")
            print(f"응답: {result['choices'][0]['message']['content']}")
            return True
        else:
            print(f"❌ 날씨 질문 테스트 실패: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 날씨 질문 테스트 실패: {e}")
        return False

def main():
    print("=== vLLM + MCP 테스트 ===\n")
    
    # 1. vLLM 서버 연결 테스트
    print("1. vLLM 서버 연결 테스트...")
    if not test_vllm_connection():
        print("vLLM 서버를 시작하세요:")
        print("vllm serve Qwen/Qwen3-8B --enable-reasoning --reasoning-parser deepseek_r1 --tensor-parallel-size 4")
        return
    
    # 2. vLLM 채팅 기능 테스트
    print("\n2. vLLM 채팅 기능 테스트...")
    if not test_vllm_chat():
        print("vLLM 채팅 기능에 문제가 있습니다.")
        return
    
    # 3. 날씨 질문 테스트
    print("\n3. 날씨 질문 테스트...")
    if not test_weather_query():
        print("날씨 질문 처리에 문제가 있습니다.")
        return
    
    print("\n🎉 모든 테스트가 성공했습니다!")
    print("이제 Streamlit 앱을 실행하여 전체 시스템을 테스트할 수 있습니다:")
    print("streamlit run qwen3_mcp_chatbot_streamlit.py")

if __name__ == "__main__":
    main() 