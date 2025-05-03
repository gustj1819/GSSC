import streamlit as st
from googletrans import Translator
from transformers import pipeline
from googletrans import LANGUAGES
import umap
from bertopic import BERTopic

st.set_page_config(page_title="Realtime Translation & Summary", layout="wide")

# 세션 상태 초기화
if "app_lang" not in st.session_state:
    st.session_state.app_lang = "ko"

if "call_ended" not in st.session_state:
    st.session_state.call_ended = False

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# 언어 선택 매핑
lang_code_map = {
    "영어 (en)": "en",
    "한국어 (ko)": "ko",
    "프랑스어 (fr)": "fr",
    "일본어 (ja)": "ja",
    "중국어 간체 (zh-cn)": "zh-cn",
    "독일어 (de)": "de"
}

# 설정 사이드바
with st.sidebar:
    st.header("⚙️ Settings")
    app_lang_label = st.selectbox("🌍 앱 언어 설정", options=list(lang_code_map.keys()))
    st.session_state.app_lang = lang_code_map[app_lang_label]

# 요약 모델 파이프라인 초기화
summary_pipeline = pipeline("summarization", model="ctu-aic/mbart25-multilingual-summarization-multilarge-cs")

def multilingual_summary(texts, target_lang="ko"):
    # 텍스트 번역
    translator = Translator()
    translated_texts = [translator.translate(text).text for text in texts]
    
    # UMAP 모델 구성
    n_components = min(5, max(2, len(texts) // 5))
    umap_model = umap.UMAP(n_neighbors=min(10, len(texts)-1), n_components=n_components, min_dist=0.1, random_state=42)
    topic_model = BERTopic(umap_model=umap_model)
    topics, _ = topic_model.fit_transform(translated_texts)
    
    # 각 주제별로 텍스트 요약
    topic_summaries = {}
    for topic in set(topics):
        topic_texts = [translated_texts[i] for i in range(len(topics)) if topics[i] == topic]
        if topic_texts:
            topic_combined_text = " ".join(topic_texts[:3])
            topic_summary = summary_pipeline(topic_combined_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
            topic_summaries[topic] = topic_summary
    
    # 요약된 텍스트를 원하는 언어로 번역
    translated_summaries = {}
    for topic, summary in topic_summaries.items():
        translated_summary = translator.translate(summary, dest=target_lang).text
        translated_summaries[topic] = translated_summary
    
    return translated_summaries

# 텍스트 레이블 정의
labels = {
    "title": {
        "en": "📹 Realtime Translation & Summary",
        "ko": "📹 실시간 번역 및 요약",
        "fr": "📹 Traduction et résumé en temps réel",
        "ja": "📹 リアルタイム翻訳と要約",
        "zh-cn": "📹 实时翻译和摘要",
        "de": "📹 Echtzeitübersetzung und Zusammenfassung"
    },
    "video_call": {
        "en": "🔴 Video Call",
        "ko": "🔴 화상 통화",
        "fr": "🔴 Appel vidéo",
        "ja": "🔴 ビデオ通話",
        "zh-cn": "🔴 视频通话",
        "de": "🔴 Videoanruf"
    },
    "input_text": {
        "en": "📝 Enter text from speech recognition",
        "ko": "📝 음성으로부터 받아온 텍스트 입력",
        "fr": "📝 Entrez le texte de la reconnaissance vocale",
        "ja": "📝 音声認識からのテキストを入力",
        "zh-cn": "📝 输入语音识别文本",
        "de": "📝 Text aus Spracherkennung eingeben"
    },
    "translated": {
        "en": "🌐 Translated Text",
        "ko": "🌐 번역된 텍스트",
        "fr": "🌐 Texte traduit",
        "ja": "🌐 翻訳されたテキスト",
        "zh-cn": "🌐 翻译后的文本",
        "de": "🌐 Übersetzter Text"
    },
    "end_call": {
        "en": "🔚 End Call",
        "ko": "🔚 통화 종료",
        "fr": "🔚 Terminer l'appel",
        "ja": "🔚 通話終了",
        "zh-cn": "🔚 结束通话",
        "de": "🔚 Anruf beenden"
    },
    "call_ended": {
        "en": "Call Ended",
        "ko": "통화가 종료되었습니다",
        "fr": "Appel terminé",
        "ja": "通話が終了しました",
        "zh-cn": "通话已结束",
        "de": "Anruf beendet"
    },
    "summary_button": {
        "en": "View Summary",
        "ko": "전체 요약 보기",
        "fr": "Voir le résumé",
        "ja": "概要を見る",
        "zh-cn": "查看总结",
        "de": "Zusammenfassung ansehen"
    }
}

# 현재 언어
target_lang = st.session_state.app_lang

# UI 시작
st.title(labels["title"][target_lang])

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(labels["video_call"][target_lang])
    zoom_url = "https://zoom.us/wc/join/1234567890"
    st.markdown(f'<iframe src="{zoom_url}" width="100%" height="600px" frameborder="0"></iframe>', unsafe_allow_html=True)

with col2:
    st.subheader(labels["input_text"][target_lang])
    user_input = st.text_area("STT 결과 텍스트를 여기에 입력하세요...", value=st.session_state.user_input)

    if user_input != st.session_state.user_input:
        st.session_state.user_input = user_input

        with st.spinner('번역 중...'):
            # 텍스트를 번역 및 요약 처리
            translated = multilingual_summary([user_input], target_lang)
            
            # 번역된 텍스트 출력
            st.subheader(labels["translated"][target_lang])
            st.write(translated)

            # 요약된 텍스트 출력
            for topic, summary in translated.items():
                st.subheader(f"🧠 요약 {topic}")
                st.write(summary)

# 통화 종료 후 "전체 요약 보기" 버튼 표시
if st.session_state.call_ended:
    if st.button(labels["summary_button"][target_lang]):
        st.write("전체 요약을 아래에서 확인하세요.")
        for topic, summary in translated.items():
            st.subheader(f"🧠 요약 {topic}")
            st.write(summary)

# 통화 종료 버튼
if st.button(labels["end_call"][target_lang]):
    st.session_state.call_ended = True
    st.success(labels["call_ended"][target_lang])
    st.experimental_rerun()  # 페이지 강제 리로드







