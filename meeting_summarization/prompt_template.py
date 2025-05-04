import os
import logging
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def build_prompt(meeting_transcript: str) -> str:
    """
    Build the prompt for summarizing meeting transcript

    Args:
        meeting_transcript(str): The text of the conversation from the meeting.

    Returns:
        str: The formatted prompt for summarizing the meeting content.

    """
    try:
        template = {
            'system_role': "당신은 회의 전체 내용을 카테고리별로 요약해주는 요약 AI입니다."
            'output_format':[
                '다음 회의 날짜와 시간: OOO",
                '아이템: OOO",
                '해야 할 일: OOO",
                '팀원들의 의견: OOO",
                '멘토의 피드백: OOO",
            ]
        }

        #프롬프트 생성
        prompt = f"""
{template['system_role']}
아래 회의 전체 내용을 다음 형식으로 요약하세요:

[출력 예시]
{chr(10).join(template['output_format'])}

[회의 전체 내용]
{meeting_transcript}

[정리된 결과]
"""
        return prompt
    
    except Exception as e:
        logging.error(f"Failed to build prompt: {str(e)}")
        raise