import streamlit as st
import json
import pandas as pd
import numpy as np
from googletrans import Translator
import re
from datetime import datetime
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Add the parent directory to Python path to import from standalone_eval
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from standalone_eval.utils import compute_average_precision_detection, compute_temporal_iou_batch_cross, compute_temporal_iou_batch_paired
def plot_temporal_visualization(pred_windows, gt_windows, duration, pred_saliency_scores, gt_data=None):
    """시간축 기반 시각화"""
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Predicted Windows', 'Ground Truth Windows', 'Predicted Saliency Scores', 'Ground Truth Saliency Scores'),
        vertical_spacing=0.12,
        row_heights=[0.25, 0.25, 0.25, 0.25]
    )

st.set_page_config(
    page_title="QD-DETR Audio-Only Model Performance Viewer", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #fafafa;
    }
    .query-container {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin-bottom: 1rem;
    }
    .prediction-container {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #ff9800;
        margin-bottom: 1rem;
    }
    .ground-truth-container {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #4caf50;
        margin-bottom: 1rem;
    }
    .performance-high {
        background-color: #c8e6c9;
        border-left: 4px solid #4caf50;
    }
    .performance-medium {
        background-color: #fff9c4;
        border-left: 4px solid #ff9800;
    }
    .performance-low {
        background-color: #ffcdd2;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_jsonl_data(file_path):
    """JSONL 파일을 로드하는 함수"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except Exception as e:
        st.error(f"파일 로드 중 오류 발생: {e}")
        return []
    return data

def calculate_iou(pred_window, gt_window):
    """두 윈도우 간의 IoU를 계산 (eval.py와 동일한 방식)"""
    pred_start, pred_end = pred_window[0], pred_window[1]
    gt_start, gt_end = gt_window[0], gt_window[1]
    
    # compute_temporal_iou_batch_paired와 동일한 로직
    intersection = max(0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = max(pred_end, gt_end) - min(pred_start, gt_start)
    
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_recall_at_1(pred_windows, gt_windows, iou_threshold=0.5):
    """Recall@1 계산 (eval.py와 동일한 방식)"""
    if not gt_windows or not pred_windows:
        return 0.0
    
    # 상위 1개 예측 윈도우만 선택
    top_pred = pred_windows[0][:2]  # [start, end]만 사용
    
    # GT 윈도우들과 IoU 계산하여 최대값 구하기 (eval.py의 compute_mr_r1과 동일)
    pred_array = np.array([top_pred])
    gt_array = np.array(gt_windows)
    ious = compute_temporal_iou_batch_cross(pred_array, gt_array)[0]
    max_iou = np.max(ious)
    
    return 1.0 if max_iou >= iou_threshold else 0.0

def calculate_ap_using_official_method(pred_windows, gt_windows, iou_thresholds=None):
    """eval.py의 compute_average_precision_detection을 사용한 AP 계산"""
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]
    
    if not gt_windows or not pred_windows:
        return {f'AP@{th}': 0.0 for th in iou_thresholds}
    
    # 데이터를 eval.py format으로 변환
    ground_truth = []
    for i, gt_window in enumerate(gt_windows):
        ground_truth.append({
            'video-id': 'dummy_id',
            't-start': gt_window[0],
            't-end': gt_window[1]
        })
    
    prediction = []
    for pred_window in pred_windows:
        prediction.append({
            'video-id': 'dummy_id',
            't-start': pred_window[0],
            't-end': pred_window[1],
            'score': pred_window[2] if len(pred_window) > 2 else 1.0
        })
    
    ap_results = {}
    
    # 각 IoU threshold에 대해 AP 계산
    for iou_threshold in iou_thresholds:
        ap_scores = compute_average_precision_detection(
            ground_truth, 
            prediction, 
            tiou_thresholds=np.array([iou_threshold])
        )
        ap_results[f'AP@{iou_threshold}'] = ap_scores[0]
    
    return ap_results

def calculate_ap(pred_windows, gt_windows, iou_thresholds=None):
    """eval.py와 동일한 방식으로 AP 계산"""
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]
    
    # 기본 AP@0.5, AP@0.75 계산
    ap_results = calculate_ap_using_official_method(pred_windows, gt_windows, iou_thresholds)
    
    # AP@Avg 계산 (0.5부터 0.95까지 10개 threshold의 평균)
    if iou_thresholds == [0.5, 0.75]:  # 기본 호출인 경우에만 AP@Avg 계산
        avg_thresholds = np.linspace(0.5, 0.95, 10)
        ap_avg_results = calculate_ap_using_official_method(pred_windows, gt_windows, avg_thresholds)
        avg_ap_values = [ap_avg_results[f'AP@{th}'] for th in avg_thresholds]
        ap_results['AP@Avg'] = np.mean(avg_ap_values)
    
    return ap_results
    
    return ap_results

def process_predictions(pred_data, gt_data):
    """예측 데이터와 실제 데이터를 매칭하고 성능 지표 계산"""
    results = []
    
    # GT 데이터를 딕셔너리로 변환
    gt_dict = {item['qid']: item for item in gt_data}
    
    for pred_item in pred_data:
        qid = pred_item['qid']
        
        if qid not in gt_dict:
            continue
        
        gt_item = gt_dict[qid]
        
        # 예측 윈도우 파싱
        pred_windows = pred_item.get('pred_relevant_windows', [])
        
        # 실제 윈도우 파싱
        gt_windows = gt_item.get('relevant_windows', [])
        
        # 성능 지표 계산
        recall_1_05 = calculate_recall_at_1(pred_windows, gt_windows, iou_threshold=0.5)
        recall_1_07 = calculate_recall_at_1(pred_windows, gt_windows, iou_threshold=0.7)
        ap_scores = calculate_ap(pred_windows, gt_windows, iou_thresholds=[0.5, 0.75])
        
        result = {
            'qid': qid,
            'query': pred_item.get('query', ''),
            'vid': pred_item.get('vid', ''),
            'duration': gt_item.get('duration', 0),
            'pred_windows': pred_windows,
            'gt_windows': gt_windows,
            'recall_1_05': recall_1_05,
            'recall_1_07': recall_1_07,
            'ap_05': ap_scores['AP@0.5'],
            'ap_075': ap_scores['AP@0.75'],
            'ap_avg': ap_scores['AP@Avg'],
            'pred_saliency_scores': pred_item.get('pred_saliency_scores', [])
        }
        
        results.append(result)
    
    return results

def get_youtube_embed_url(youtube_id, start_time=None):
    """YouTube 임베드 URL을 생성하는 함수"""
    if youtube_id:
        base_url = f"https://www.youtube.com/embed/{youtube_id}"
        if start_time:
            base_url += f"?start={int(start_time)}&autoplay=1"
        return base_url
    return None

def seconds_to_mmss(seconds):
    """초를 mm:ss 형식으로 변환"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def extract_youtube_id_from_vid(vid):
    """vid에서 YouTube ID를 추출"""
    parts = vid.split('_')
    if len(parts) >= 2:
        return parts[0]
    return None

def get_youtube_url(youtube_id):
    """YouTube URL 생성"""
    if youtube_id:
        return f"https://www.youtube.com/watch?v={youtube_id}"
    return None

def get_video_start_time(vid):
    """vid에서 비디오 시작 시간을 추출"""
    parts = vid.split('_')
    if len(parts) >= 2:
        try:
            return float(parts[1])
        except ValueError:
            return 0.0
    return 0.0

@st.cache_data
def translate_text(text, target_language='ko'):
    """Google Translate를 사용하여 텍스트를 번역"""
    try:
        translator = Translator()
        result = translator.translate(text, dest=target_language)
        return result.text
    except Exception as e:
        return f"번역 실패: {text}"

def plot_temporal_visualization(pred_windows, gt_windows, duration, pred_saliency_scores, gt_data=None):
    """시간축 기반 시각화"""
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Predicted Windows', 'Ground Truth Windows', 'Predicted Saliency Scores', 'Ground Truth Saliency Scores'),
        vertical_spacing=0.12,
        row_heights=[0.23, 0.23, 0.27, 0.27]
    )
    
    # 예측 윈도우 시각화 (상위 10개)
    for i, window in enumerate(pred_windows[:10]):
        start, end = window[0], window[1]
        confidence = window[2] if len(window) > 2 else 1.0
        
        fig.add_trace(
            go.Scatter(
                x=[start, end, end, start, start],
                y=[i, i, i+0.8, i+0.8, i],
                fill="toself",
                fillcolor=f"rgba(255, 99, 71, {confidence})",
                line=dict(color="red", width=2),
                name=f"Pred {i+1} ({confidence * 100:.2f})",
                mode="lines"
            ),
            row=1, col=1
        )
    
    # 실제 윈도우 시각화  
    for i, window in enumerate(gt_windows):
        start, end = window[0], window[1]
        
        fig.add_trace(
            go.Scatter(
                x=[start, end, end, start, start],
                y=[i, i, i+0.8, i+0.8, i],
                fill="toself",
                fillcolor="rgba(50, 205, 50, 0.7)",
                line=dict(color="green", width=2),
                name=f"GT {i+1}",
                mode="lines"
            ),
            row=2, col=1
        )
    
    # 예측 Saliency 점수 시각화
    if pred_saliency_scores:
        # 0~75 범위의 time points 생성 (2초 간격으로 총 75개)
        time_points = [i * 2 for i in range(len(pred_saliency_scores))]
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=pred_saliency_scores,
                mode="lines+markers",
                name="Predicted Saliency",
                line=dict(color="blue", width=2),
                marker=dict(size=4)
            ),
            row=3, col=1
        )
    
    # GT Saliency 점수 시각화
    if gt_data and 'saliency_scores' in gt_data:
        gt_saliency_scores = gt_data['saliency_scores']
        if gt_saliency_scores:
            # GT saliency scores를 평균값으로 계산
            avg_gt_saliency = [np.mean(scores) for scores in gt_saliency_scores]
            
            # relevant_clip_ids가 있으면 해당 time points 사용, 없으면 연속적으로 생성
            if 'relevant_clip_ids' in gt_data and gt_data['relevant_clip_ids']:
                time_points_gt = [clip_id * 2 for clip_id in gt_data['relevant_clip_ids']]
            else:
                time_points_gt = [i * 2 for i in range(len(avg_gt_saliency))]
            
            fig.add_trace(
                go.Scatter(
                    x=time_points_gt,
                    y=avg_gt_saliency,
                    mode="lines+markers",
                    name="GT Saliency (avg)",
                    line=dict(color="green", width=2),
                    marker=dict(size=4)
                ),
                row=4, col=1
            )
    
    fig.update_layout(
        height=1000,
        title_text="Temporal Moment Detection Visualization",
        showlegend=True,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    # x축 범위 설정
    fig.update_xaxes(range=[0, 150], title_text="Time (seconds)", row=1, col=1)
    fig.update_xaxes(range=[0, 150], title_text="Time (seconds)", row=2, col=1)
    fig.update_xaxes(range=[0, 75], title_text="Time (seconds)", row=3, col=1)  # Saliency는 0~75
    fig.update_xaxes(range=[0, 75], title_text="Time (seconds)", row=4, col=1)  # GT Saliency도 0~75
    
    fig.update_yaxes(title_text="Windows", row=1, col=1)
    fig.update_yaxes(title_text="Windows", row=2, col=1)
    fig.update_yaxes(title_text="Saliency", row=3, col=1)
    fig.update_yaxes(title_text="GT Saliency", row=4, col=1)
    
    return fig

def display_result_item(result, idx, sort_by, gt_data_dict):
    """결과 항목을 표시"""
    qid = result['qid']
    query = result['query']
    vid = result['vid']
    duration = result['duration']
    pred_windows = result['pred_windows']
    gt_windows = result['gt_windows']
    recall_1_05 = result['recall_1_05']
    recall_1_07 = result['recall_1_07']
    ap_05 = result['ap_05']
    ap_075 = result['ap_075']
    ap_avg = result['ap_avg']
    pred_saliency_scores = result['pred_saliency_scores']
    
    # GT 데이터 가져오기
    gt_data_item = gt_data_dict.get(qid, {})
    
    # YouTube 정보
    youtube_id = extract_youtube_id_from_vid(vid)
    youtube_url = get_youtube_url(youtube_id)
    video_start_time = get_video_start_time(vid)
    
    # 현재 정렬 기준에 따른 점수 계산
    if sort_by == "AP@Avg":
        current_score = ap_avg
        score_label = "AP@Avg"
    elif sort_by == "Recall@1 (0.5)":
        current_score = recall_1_05
        score_label = "Recall@1 (0.5)"
    elif sort_by == "Recall@1 (0.7)":
        current_score = recall_1_07
        score_label = "Recall@1 (0.7)"
    elif sort_by == "AP@0.5":
        current_score = ap_05
        score_label = "AP@0.5"
    elif sort_by == "AP@0.75":
        current_score = ap_075
        score_label = "AP@0.75"
    else:
        current_score = ap_avg
        score_label = "AP@Avg"
    
    # 성능 분류
    if current_score >= 0.7:
        perf_class = "performance-high"
        perf_emoji = "🟢"
    elif current_score >= 0.4:
        perf_class = "performance-medium"
        perf_emoji = "🟡"
    else:
        perf_class = "performance-low"
        perf_emoji = "🔴"
    
    # 메인 헤더
    st.markdown(f"""
    <div class="metric-container {perf_class}">
        <h3>{perf_emoji} 쿼리 #{idx} (QID: {qid}) - {score_label}: {current_score * 100:.2f}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # 컬럼 분할
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 원본 쿼리
        st.markdown(f"""
        <div class="query-container">
            <h4>🔍 원본 쿼리</h4>
            <p>{query}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 번역된 쿼리
        with st.spinner("번역 중..."):
            translated_query = translate_text(query, 'ko')
        
        st.markdown(f"""
        <div class="query-container">
            <h4>🌐 한국어 번역</h4>
            <p>{translated_query}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 성능 지표
        st.markdown("#### 📊 성능 지표")
        
        col_r05, col_r07, col_ap05 = st.columns(3)
        
        with col_r05:
            st.metric(
                "Recall@1 (0.5)", 
                f"{recall_1_05 * 100:.2f}",
                help="IoU 0.5에서 상위 1개 예측의 정확도"
            )
        
        with col_r07:
            st.metric(
                "Recall@1 (0.7)", 
                f"{recall_1_07 * 100:.2f}",
                help="IoU 0.7에서 상위 1개 예측의 정확도"
            )
        
        with col_ap05:
            st.metric(
                "AP@0.5", 
                f"{ap_05 * 100:.2f}",
                help="IoU 0.5에서의 Average Precision"
            )
        
        col_ap075, col_apavg, col_empty = st.columns(3)
        
        with col_ap075:
            st.metric(
                "AP@0.75", 
                f"{ap_075 * 100:.2f}",
                help="IoU 0.75에서의 Average Precision"
            )
        
        with col_apavg:
            st.metric(
                "AP@Avg", 
                f"{ap_avg * 100:.2f}",
                help="평균 Average Precision"
            )
        
        # 비디오 정보
        st.markdown(f"""
        <div class="video-info">
            <h4>📹 비디오 정보</h4>
            <p><strong>VID:</strong> <code>{vid}</code></p>
            <p><strong>YouTube ID:</strong> <code>{youtube_id}</code></p>
            <p><strong>비디오 길이:</strong> {duration}초</p>
            <p><strong>YouTube 링크:</strong> <a href="{youtube_url}" target="_blank">여기서 보기</a></p>
        </div>
        """, unsafe_allow_html=True)
        
        # 정답 모멘트 구간
        st.markdown("**⏰ 정답 모멘트 구간:**")
        
        if gt_windows:
            for j, window in enumerate(gt_windows):
                if len(window) >= 2:
                    start_in_clip = window[0]
                    end_in_clip = window[1]
                    
                    # 전체 비디오에서의 실제 시간 계산
                    actual_start = video_start_time + start_in_clip
                    actual_end = video_start_time + end_in_clip
                    
                    st.markdown(f"""
                    <div class="ground-truth-container">
                        <strong>구간 {j+1}:</strong><br>
                        &nbsp;&nbsp;• 클립 내: {seconds_to_mmss(start_in_clip)} ~ {seconds_to_mmss(end_in_clip)}<br>
                        &nbsp;&nbsp;• 전체 영상: {seconds_to_mmss(actual_start)} ~ {seconds_to_mmss(actual_end)}
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="ground-truth-container">
                정답 구간이 없습니다.
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # YouTube 비디오 임베드
        if youtube_id:
            st.markdown("#### 🎥 YouTube 비디오")
            
            # VID의 시작 시간으로 비디오 시작
            embed_start_time = video_start_time
            
            embed_url = get_youtube_embed_url(youtube_id, embed_start_time)
            
            # iframe을 사용하여 YouTube 비디오 임베드
            st.markdown(f"""
            <iframe width="100%" height="315" 
                    src="{embed_url}" 
                    frameborder="0" 
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                    allowfullscreen>
            </iframe>
            """, unsafe_allow_html=True)
            
            # 시간 점프 버튼들
            if gt_windows:
                st.markdown("**⏯️ 구간 바로가기:**")
                for j, window in enumerate(gt_windows):
                    if len(window) >= 2:
                        jump_time = video_start_time + window[0]
                        jump_url = get_youtube_url(youtube_id) + f"&t={int(jump_time)}s"
                        st.markdown(f"🔗 [구간 {j+1} 바로가기]({jump_url})")
        else:
            st.warning("⚠️ YouTube 비디오를 찾을 수 없습니다.")
    
    # 시간축 시각화
    fig = plot_temporal_visualization(pred_windows, gt_windows, duration, pred_saliency_scores, gt_data_item)
    st.plotly_chart(fig, use_container_width=True)
    
    # 상세 예측 정보
    st.markdown("#### 📋 모델 예측 상세 (상위 10개)")
    
    for i, window in enumerate(pred_windows[:10]):
        start, end = window[0], window[1]
        confidence = window[2] if len(window) > 2 else 1.0
        
        # 전체 비디오에서의 실제 시간
        actual_start = video_start_time + start
        actual_end = video_start_time + end
        
        st.markdown(f"""
        <div class="prediction-container">
            <strong>예측 {i+1}:</strong> {seconds_to_mmss(start)} ~ {seconds_to_mmss(end)} 
            (신뢰도: {confidence * 100:.2f})<br>
            <em>전체 영상: {seconds_to_mmss(actual_start)} ~ {seconds_to_mmss(actual_end)}</em>
        </div>
        """, unsafe_allow_html=True)

def main():
    # 메인 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🎧 QD-DETR Audio-Only Model Performance Viewer</h1>
        <p>오디오 전용 모델의 성능을 분석하고 최고 성능 쿼리들을 확인하세요</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.title("⚙️ 설정")
        
        # 파일 경로 설정
        pred_file = st.text_input(
            "🤖 예측 결과 파일:",
            value="results/audio-seed2018/best_hl_val_preds.jsonl",
            help="모델의 예측 결과 파일 경로"
        )
        
        gt_file = st.text_input(
            "✅ 실제 정답 파일:",
            value="data/highlight_val_release.jsonl",
            help="Validation 정답 데이터 파일 경로"
        )
        
        # 정렬 옵션
        sort_by = st.selectbox(
            "📈 정렬 기준:",
            ["AP@Avg", "Recall@1 (0.5)", "Recall@1 (0.7)", "AP@0.5", "AP@0.75"],
            help="결과를 정렬할 기준을 선택하세요"
        )
        
        # 필터 옵션
        min_performance = st.slider(
            "🎯 최소 성능 임계값:",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            help="이 값 이상의 성능을 가진 쿼리만 표시 (백분율)"
        )
        
        # 페이지당 아이템 수
        items_per_page = st.slider(
            "📄 페이지당 아이템 수:",
            min_value=1,
            max_value=20,
            value=1
        )
        
        # 검색
        search_query = st.text_input(
            "🔍 쿼리 검색:",
            placeholder="검색어를 입력하세요...",
            help="쿼리 내용으로 검색"
        )
    
    # 데이터 로드
    with st.spinner("📊 데이터를 로드하고 성능을 계산하는 중..."):
        pred_data = load_jsonl_data(pred_file)
        gt_data = load_jsonl_data(gt_file)
        
        if not pred_data or not gt_data:
            st.error("❌ 데이터를 로드할 수 없습니다. 파일 경로를 확인해주세요.")
            return
        
        # 성능 계산
        results = process_predictions(pred_data, gt_data)
        
        # GT 데이터를 딕셔너리로 변환 (display_result_item에서 사용)
        gt_data_dict = {item['qid']: item for item in gt_data}
    
    if not results:
        st.error("❌ 매칭되는 데이터가 없습니다.")
        return
    
    # 전체 통계
    avg_recall_1_05 = np.mean([r['recall_1_05'] for r in results])
    avg_recall_1_07 = np.mean([r['recall_1_07'] for r in results])
    avg_ap_05 = np.mean([r['ap_05'] for r in results])
    avg_ap_075 = np.mean([r['ap_075'] for r in results])
    avg_ap_avg = np.mean([r['ap_avg'] for r in results])
    
    st.markdown("#### 📊 전체 성능 통계")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("평균 Recall@1 (0.5)", f"{avg_recall_1_05 * 100:.2f}")
    with col2:
        st.metric("평균 Recall@1 (0.7)", f"{avg_recall_1_07 * 100:.2f}")
    with col3:
        st.metric("평균 AP@0.5", f"{avg_ap_05 * 100:.2f}")
    with col4:
        st.metric("평균 AP@0.75", f"{avg_ap_075 * 100:.2f}")
    with col5:
        st.metric("평균 AP@Avg", f"{avg_ap_avg * 100:.2f}")
    
    # 총 쿼리 수 표시
    st.metric("총 쿼리 수", len(results))
    
    # 성능 분포 히스토그램
    fig_dist = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Recall@1 (0.5)', 'Recall@1 (0.7)', 'AP@0.5', 'AP@0.75', 'AP@Avg', '')
    )
    
    fig_dist.add_trace(
        go.Histogram(x=[r['recall_1_05'] for r in results], name="Recall@1 (0.5)", nbinsx=20),
        row=1, col=1
    )
    fig_dist.add_trace(
        go.Histogram(x=[r['recall_1_07'] for r in results], name="Recall@1 (0.7)", nbinsx=20),
        row=1, col=2
    )
    fig_dist.add_trace(
        go.Histogram(x=[r['ap_05'] for r in results], name="AP@0.5", nbinsx=20),
        row=1, col=3
    )
    fig_dist.add_trace(
        go.Histogram(x=[r['ap_075'] for r in results], name="AP@0.75", nbinsx=20),
        row=2, col=1
    )
    fig_dist.add_trace(
        go.Histogram(x=[r['ap_avg'] for r in results], name="AP@Avg", nbinsx=20),
        row=2, col=2
    )
    
    fig_dist.update_layout(height=600, title_text="Performance Distribution")
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # 결과 필터링 및 정렬
    filtered_results = results
    
    # 성능 필터
    if min_performance > 0:
        if sort_by == "AP@Avg":
            filtered_results = [r for r in filtered_results if r['ap_avg'] * 100 >= min_performance]
        elif sort_by == "Recall@1 (0.5)":
            filtered_results = [r for r in filtered_results if r['recall_1_05'] * 100 >= min_performance]
        elif sort_by == "Recall@1 (0.7)":
            filtered_results = [r for r in filtered_results if r['recall_1_07'] * 100 >= min_performance]
        elif sort_by == "AP@0.5":
            filtered_results = [r for r in filtered_results if r['ap_05'] * 100 >= min_performance]
        elif sort_by == "AP@0.75":
            filtered_results = [r for r in filtered_results if r['ap_075'] * 100 >= min_performance]
    
    # 검색 필터
    if search_query:
        filtered_results = [
            r for r in filtered_results 
            if search_query.lower() in r['query'].lower()
        ]
    
    # 정렬
    if sort_by == "AP@Avg":
        filtered_results.sort(key=lambda x: x['ap_avg'], reverse=True)
    elif sort_by == "Recall@1 (0.5)":
        filtered_results.sort(key=lambda x: x['recall_1_05'], reverse=True)
    elif sort_by == "Recall@1 (0.7)":
        filtered_results.sort(key=lambda x: x['recall_1_07'], reverse=True)
    elif sort_by == "AP@0.5":
        filtered_results.sort(key=lambda x: x['ap_05'], reverse=True)
    elif sort_by == "AP@0.75":
        filtered_results.sort(key=lambda x: x['ap_075'], reverse=True)
    
    # 페이지네이션
    total_items = len(filtered_results)
    total_pages = max(1, (total_items - 1) // items_per_page + 1)
    
    if 'page' not in st.session_state:
        st.session_state.page = 1
    
    st.markdown(f"**🎯 필터링된 결과: {total_items}개 (전체 {len(results)}개 중)**")
    
    # 페이지 네비게이션
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("⬅️ 이전", disabled=st.session_state.page <= 1):
            st.session_state.page -= 1
            st.rerun()
    
    with col2:
        if st.button("⏮️ 처음"):
            st.session_state.page = 1
            st.rerun()
    
    with col3:
        page_options = list(range(1, total_pages + 1))
        current_page_index = min(st.session_state.page - 1, len(page_options) - 1)
        current_page_index = max(0, current_page_index)
        
        page = st.selectbox(
            "페이지:",
            page_options,
            index=current_page_index,
            key='page_select'
        )
        if page != st.session_state.page:
            st.session_state.page = page
            st.rerun()
    
    with col4:
        if st.button("⏭️ 마지막"):
            st.session_state.page = total_pages
            st.rerun()
    
    with col5:
        if st.button("➡️ 다음", disabled=st.session_state.page >= total_pages):
            st.session_state.page += 1
            st.rerun()
    
    # 현재 페이지 데이터
    start_idx = (st.session_state.page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    current_results = filtered_results[start_idx:end_idx]
    
    st.markdown(f"**📊 페이지 {st.session_state.page} / {total_pages}** (아이템 {start_idx + 1}-{end_idx} / {total_items})")
    st.markdown("---")
    
    # 결과 표시
    for i, result in enumerate(current_results):
        actual_idx = start_idx + i + 1
        display_result_item(result, actual_idx, sort_by, gt_data_dict)
        
        if i < len(current_results) - 1:
            st.markdown("---")

if __name__ == "__main__":
    main()
