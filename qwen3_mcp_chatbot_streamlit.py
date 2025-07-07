import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import torch
from typing import List, Tuple, Generator
import threading
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.utils.output_beautify import typewriter_print
import requests

llm_cfg = {
    'model': 'Qwen/Qwen3-8B',
    'model_server': 'http://localhost:8000/v1',  # vLLM API endpoint
    'api_key': 'EMPTY',
    'generate_cfg': {
        'top_p': 0.8,
        'temperature': 0.7,
        'max_tokens': 2048
    }
}


# Step 3: Create an agent with MCP tools for korea_weather
system_instruction = '''당신은 한국 날씨 정보를 제공하는 AI 어시스턴트입니다. 

사용자가 날씨 정보를 요청하면:
1. korea_weather MCP 서버의 도구들을 사용하여 날씨 정보를 조회합니다
2. 조회된 정보를 한국어로 친절하게 설명합니다
3. 필요한 경우 위도/경도 정보를 요청하거나 서울(37.5665, 126.9780) 등의 주요 도시 좌표를 사용합니다

사용 가능한 날씨 도구들:
- get_nowcast_observation: 현재 날씨 관측 정보 (기온, 강수량, 습도, 풍속)
- get_nowcast_forecast: 초단기 예보 (6시간 이내)
- get_short_term_forecast: 단기 예보 (3일 이내)

항상 한국어로 응답하고, 날씨 정보를 명확하고 이해하기 쉽게 설명해주세요.

만약 날씨와 관계 없는 질문이라면 도구 사용 없이 일반적인 대화로 응답해주세요.

'''

# MCP 서버 설정
tools = [
    {
        "mcpServers": {
            "korea_weather": {
                "command": "uv",
                "args": [
                    "run",
                    "korea_weather.py"
                ]
            },
        }
    }
]

bot = Assistant(llm=llm_cfg,
                system_message=system_instruction,
                function_list=tools)

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


def main():
    st.set_page_config(
        page_title="Qwen3-8B + vLLM + MCP Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Qwen3-8B + vLLM + MCP Chatbot")
    st.markdown("Qwen3-8B 모델과 vLLM 서버, MCP 도구를 사용한 AI 챗봇입니다.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar for controls
    with st.sidebar:
        st.header("설정")
        
        # 서버 상태 확인
        st.subheader("서버 상태")
        if test_vllm_connection():
            st.success("✅ vLLM 서버 연결 성공")
            st.info("""
            **vLLM 서버 정보:**
            - 모델: Qwen/Qwen3-8B
            - 엔드포인트: http://localhost:8000
            - 상태: 실행 중
            """)
        else:
            st.error("❌ vLLM 서버 연결 실패")
            st.info("""
            **vLLM 서버 시작 방법:**
            ```bash
            vllm serve Qwen/Qwen3-8B --enable-reasoning --reasoning-parser deepseek_r1 --tensor-parallel-size 4
            ```
            """)
        
        # MCP 도구 정보
        st.subheader("MCP 도구")
        st.info("""
        **사용 가능한 도구:**
        - korea_weather: 한국 날씨 정보
        - 실시간 날씨 데이터 조회
        
        **사용 예시:**
        - "서울 날씨 알려줘"
        - "부산 오늘 날씨는?"
        """)
        
        if st.button("새 채팅", type="secondary"):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("메시지를 입력하세요..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response with streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Stream the response
            response_plain_text = ""
            full_response = ""
            
            try:
                for response in bot.run(messages=st.session_state.messages):
                    # Streaming output
                    response_plain_text = typewriter_print(response, response_plain_text)
                    message_placeholder.markdown(response_plain_text + "▌")
                    full_response = response_plain_text
                
                # Final response without cursor
                message_placeholder.markdown(full_response)
                
                # Append the bot responses to the chat history
                st.session_state.messages.extend(response)
                
            except Exception as e:
                error_msg = f"오류가 발생했습니다: {str(e)}"
                message_placeholder.markdown(error_msg)
                full_response = error_msg
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()