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

# 페이지 설정
st.set_page_config(
    page_title="QD-DETR Model Comparison Viewer", 
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
    .comparison-header {
        background: linear-gradient(90deg, #ff6b6b 0%, #feca57 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .model1-better {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .model2-better {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
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
    .performance-diff-high {
        background-color: #c8e6c9;
        border-left: 4px solid #4caf50;
    }
    .performance-diff-medium {
        background-color: #fff9c4;
        border-left: 4px solid #ff9800;
    }
    .performance-diff-low {
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

@st.cache_data
def process_predictions(pred_data, gt_data):
    """예측 데이터와 실제 데이터를 매칭하고 성능 지표 계산 (캐시됨)"""
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

@st.cache_data
def compare_models(results1, results2):
    """두 모델의 결과를 비교 (캐시됨)"""
    # 공통 QID 찾기
    qids1 = {r['qid'] for r in results1}
    qids2 = {r['qid'] for r in results2}
    common_qids = qids1.intersection(qids2)
    
    # 결과를 딕셔너리로 변환
    results1_dict = {r['qid']: r for r in results1}
    results2_dict = {r['qid']: r for r in results2}
    
    comparisons = []
    
    for qid in common_qids:
        r1 = results1_dict[qid]
        r2 = results2_dict[qid]
        
        comparison = {
            'qid': qid,
            'query': r1['query'],
            'vid': r1['vid'],
            'duration': r1['duration'],
            'gt_windows': r1['gt_windows'],
            'model1_ap_avg': r1['ap_avg'],
            'model2_ap_avg': r2['ap_avg'],
            'ap_diff': r1['ap_avg'] - r2['ap_avg'],  # Model1 - Model2
            'model1_pred_windows': r1['pred_windows'],
            'model2_pred_windows': r2['pred_windows'],
            'model1_saliency': r1['pred_saliency_scores'],
            'model2_saliency': r2['pred_saliency_scores'],
            'model1_recall_1_05': r1['recall_1_05'],
            'model2_recall_1_05': r2['recall_1_05'],
            'model1_recall_1_07': r1['recall_1_07'],
            'model2_recall_1_07': r2['recall_1_07'],
            'model1_ap_05': r1['ap_05'],
            'model2_ap_05': r2['ap_05'],
            'model1_ap_075': r1['ap_075'],
            'model2_ap_075': r2['ap_075'],
        }
        
        comparisons.append(comparison)
    
    return comparisons

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

@st.cache_data(ttl=3600)  # 1시간 캐시
def process_and_compare_models(model1_file, model2_file, gt_file):
    """모든 모델 처리와 비교를 한 번에 수행 (메인 캐시 함수)"""
    # 데이터 로드
    model1_data = load_jsonl_data(model1_file)
    model2_data = load_jsonl_data(model2_file)
    gt_data = load_jsonl_data(gt_file)
    
    if not model1_data or not model2_data or not gt_data:
        return None, None, None
    
    # 성능 계산
    results1 = process_predictions(model1_data, gt_data)
    results2 = process_predictions(model2_data, gt_data)
    
    # 모델 비교
    comparisons = compare_models(results1, results2)
    
    # GT 데이터를 딕셔너리로 변환
    gt_data_dict = {item['qid']: item for item in gt_data}
    
    return comparisons, gt_data_dict, len(gt_data)

@st.cache_data(ttl=3600)  # 1시간 캐시
def translate_text_cached(text, target_language='ko'):
    """캐시된 번역 함수"""
    try:
        translator = Translator()
        result = translator.translate(text, dest=target_language)
        return result.text
    except Exception as e:
        return f"번역 실패: {text}"

@st.cache_data
def plot_comparison_visualization(comparison, gt_data_dict, model1_name, model2_name):
    """모델 비교 시각화 (캐시됨)"""
    qid = comparison['qid']
    gt_data_item = gt_data_dict.get(qid, {})
    
    fig = make_subplots(
        rows=6, cols=1,
        subplot_titles=(
            f'{model1_name} Predicted Windows', 
            f'{model2_name} Predicted Windows',
            'Ground Truth Windows', 
            f'{model1_name} Saliency Scores', 
            f'{model2_name} Saliency Scores',
            'Ground Truth Saliency Scores'
        ),
        vertical_spacing=0.08,
        row_heights=[0.15, 0.15, 0.15, 0.18, 0.18, 0.19]
    )
    
    # Model 1 예측 윈도우 시각화 (상위 10개)
    model1_windows = comparison['model1_pred_windows'][:10]
    for i, window in enumerate(model1_windows):
        start, end = window[0], window[1]
        confidence = window[2] if len(window) > 2 else 1.0
        
        fig.add_trace(
            go.Scatter(
                x=[start, end, end, start, start],
                y=[i, i, i+0.8, i+0.8, i],
                fill="toself",
                fillcolor=f"rgba(255, 99, 71, {confidence})",
                line=dict(color="red", width=2),
                name=f"{model1_name} Pred {i+1} ({confidence * 100:.2f})",
                mode="lines",
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Model 2 예측 윈도우 시각화 (상위 10개)
    model2_windows = comparison['model2_pred_windows'][:10]
    for i, window in enumerate(model2_windows):
        start, end = window[0], window[1]
        confidence = window[2] if len(window) > 2 else 1.0
        
        fig.add_trace(
            go.Scatter(
                x=[start, end, end, start, start],
                y=[i, i, i+0.8, i+0.8, i],
                fill="toself",
                fillcolor=f"rgba(54, 162, 235, {confidence})",
                line=dict(color="blue", width=2),
                name=f"{model2_name} Pred {i+1} ({confidence * 100:.2f})",
                mode="lines",
                showlegend=False
            ),
            row=2, col=1
        )
    
    # 실제 윈도우 시각화  
    gt_windows = comparison['gt_windows']
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
                mode="lines",
                showlegend=False
            ),
            row=3, col=1
        )
    
    # Model 1 Saliency 점수 시각화
    model1_saliency = comparison['model1_saliency']
    if model1_saliency:
        time_points = [i * 2 for i in range(len(model1_saliency))]
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=model1_saliency,
                mode="lines+markers",
                name=f"{model1_name} Saliency",
                line=dict(color="red", width=2),
                marker=dict(size=4),
                showlegend=False
            ),
            row=4, col=1
        )
    
    # Model 2 Saliency 점수 시각화
    model2_saliency = comparison['model2_saliency']
    if model2_saliency:
        time_points = [i * 2 for i in range(len(model2_saliency))]
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=model2_saliency,
                mode="lines+markers",
                name=f"{model2_name} Saliency",
                line=dict(color="blue", width=2),
                marker=dict(size=4),
                showlegend=False
            ),
            row=5, col=1
        )
    
    # GT Saliency 점수 시각화
    if gt_data_item and 'saliency_scores' in gt_data_item:
        gt_saliency_scores = gt_data_item['saliency_scores']
        if gt_saliency_scores:
            # GT saliency scores를 평균값으로 계산
            avg_gt_saliency = [np.mean(scores) for scores in gt_saliency_scores]
            
            # relevant_clip_ids가 있으면 해당 time points 사용, 없으면 연속적으로 생성
            if 'relevant_clip_ids' in gt_data_item and gt_data_item['relevant_clip_ids']:
                time_points_gt = [clip_id * 2 for clip_id in gt_data_item['relevant_clip_ids']]
            else:
                time_points_gt = [i * 2 for i in range(len(avg_gt_saliency))]
            
            fig.add_trace(
                go.Scatter(
                    x=time_points_gt,
                    y=avg_gt_saliency,
                    mode="lines+markers",
                    name="GT Saliency (avg)",
                    line=dict(color="green", width=2),
                    marker=dict(size=4),
                    showlegend=False
                ),
                row=6, col=1
            )
    
    fig.update_layout(
        height=1200,
        title_text=f"Model Comparison: {model1_name} vs {model2_name}",
        showlegend=False,
        margin=dict(l=50, r=50, t=100, b=50)
    )
    
    # x축 범위 설정 (윈도우는 0~150, saliency는 0~75)
    for row in [1, 2, 3]:
        fig.update_xaxes(range=[0, 150], title_text="Time (seconds)", row=row, col=1)
    for row in [4, 5, 6]:
        fig.update_xaxes(range=[0, 75], title_text="Time (seconds)", row=row, col=1)
    
    # y축 제목 설정
    for row in [1, 2, 3]:
        fig.update_yaxes(title_text="Windows", row=row, col=1)
    for row in [4, 5, 6]:
        fig.update_yaxes(title_text="Saliency", row=row, col=1)
    
    return fig

def display_comparison_item(comparison, idx, model1_name, model2_name, gt_data_dict):
    """비교 결과 항목을 표시"""
    qid = comparison['qid']
    query = comparison['query']
    vid = comparison['vid']
    duration = comparison['duration']
    gt_windows = comparison['gt_windows']
    
    model1_ap_avg = comparison['model1_ap_avg']
    model2_ap_avg = comparison['model2_ap_avg']
    ap_diff = comparison['ap_diff']
    
    # YouTube 정보
    youtube_id = extract_youtube_id_from_vid(vid)
    youtube_url = get_youtube_url(youtube_id)
    video_start_time = get_video_start_time(vid)
    
    # 성능 차이 분류
    if ap_diff >= 0.3:
        perf_class = "performance-diff-high"
        perf_emoji = "🟢"
        diff_desc = f"{model1_name} 우위"
    elif ap_diff >= 0.1:
        perf_class = "performance-diff-medium"
        perf_emoji = "🟡"
        diff_desc = f"{model1_name} 약간 우위"
    elif ap_diff <= -0.3:
        perf_class = "performance-diff-low"
        perf_emoji = "🔴"
        diff_desc = f"{model2_name} 우위"
    elif ap_diff <= -0.1:
        perf_class = "performance-diff-medium"
        perf_emoji = "🟡"
        diff_desc = f"{model2_name} 약간 우위"
    else:
        perf_class = "performance-diff-medium"
        perf_emoji = "⚪"
        diff_desc = "비슷한 성능"
    
    # 메인 헤더
    st.markdown(f"""
    <div class="comparison-header">
        <h3>{perf_emoji} 쿼리 #{idx} (QID: {qid}) - AP@Avg 차이: {ap_diff * 100:.2f} ({diff_desc})</h3>
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
            translated_query = translate_text_cached(query, 'ko')
        
        st.markdown(f"""
        <div class="query-container">
            <h4>🌐 한국어 번역</h4>
            <p>{translated_query}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 모델 비교 성능 지표
        st.markdown(f"#### 📊 성능 비교: {model1_name} vs {model2_name}")
        
        # AP@Avg 비교
        col_ap1, col_ap2, col_diff = st.columns(3)
        
        with col_ap1:
            st.metric(
                f"{model1_name} AP@Avg", 
                f"{model1_ap_avg * 100:.2f}",
                help=f"{model1_name}의 Average Precision"
            )
        
        with col_ap2:
            st.metric(
                f"{model2_name} AP@Avg", 
                f"{model2_ap_avg * 100:.2f}",
                help=f"{model2_name}의 Average Precision"
            )
        
        with col_diff:
            delta_color = "normal" if abs(ap_diff) < 0.1 else ("inverse" if ap_diff < 0 else "normal")
            st.metric(
                "차이 (1-2)", 
                f"{ap_diff * 100:.2f}",
                delta=f"{ap_diff * 100:.2f}",
                help="Model1 - Model2의 성능 차이"
            )
        
        # 추가 성능 지표들
        st.markdown("##### 세부 성능 지표")
        
        metrics_data = {
            "지표": ["Recall@1 (0.5)", "Recall@1 (0.7)", "AP@0.5", "AP@0.75"],
            f"{model1_name}": [
                f"{comparison['model1_recall_1_05'] * 100:.2f}",
                f"{comparison['model1_recall_1_07'] * 100:.2f}",
                f"{comparison['model1_ap_05'] * 100:.2f}",
                f"{comparison['model1_ap_075'] * 100:.2f}"
            ],
            f"{model2_name}": [
                f"{comparison['model2_recall_1_05'] * 100:.2f}",
                f"{comparison['model2_recall_1_07'] * 100:.2f}",
                f"{comparison['model2_ap_05'] * 100:.2f}",
                f"{comparison['model2_ap_075'] * 100:.2f}"
            ]
        }
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
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
    
    # 시간축 시각화
    fig = plot_comparison_visualization(comparison, gt_data_dict, model1_name, model2_name)
    st.plotly_chart(fig, use_container_width=True)
    
    # 상세 예측 정보 비교
    st.markdown(f"#### 📋 모델 예측 비교 (상위 5개씩)")
    
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.markdown(f"**{model1_name} 예측**")
        for i, window in enumerate(comparison['model1_pred_windows'][:5]):
            start, end = window[0], window[1]
            confidence = window[2] if len(window) > 2 else 1.0
            
            # 전체 비디오에서의 실제 시간
            actual_start = video_start_time + start
            actual_end = video_start_time + end
            
            st.markdown(f"""
            <div class="model1-better">
                <strong>예측 {i+1}:</strong> {seconds_to_mmss(start)} ~ {seconds_to_mmss(end)} 
                (신뢰도: {confidence * 100:.2f})<br>
                <em>전체 영상: {seconds_to_mmss(actual_start)} ~ {seconds_to_mmss(actual_end)}</em>
            </div>
            """, unsafe_allow_html=True)
    
    with col_m2:
        st.markdown(f"**{model2_name} 예측**")
        for i, window in enumerate(comparison['model2_pred_windows'][:5]):
            start, end = window[0], window[1]
            confidence = window[2] if len(window) > 2 else 1.0
            
            # 전체 비디오에서의 실제 시간
            actual_start = video_start_time + start
            actual_end = video_start_time + end
            
            st.markdown(f"""
            <div class="model2-better">
                <strong>예측 {i+1}:</strong> {seconds_to_mmss(start)} ~ {seconds_to_mmss(end)} 
                (신뢰도: {confidence * 100:.2f})<br>
                <em>전체 영상: {seconds_to_mmss(actual_start)} ~ {seconds_to_mmss(actual_end)}</em>
            </div>
            """, unsafe_allow_html=True)

def main():
    # 메인 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🔄 QD-DETR Model Comparison Viewer</h1>
        <p>두 모델의 성능을 비교하고 차이를 분석하세요</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.title("⚙️ 설정")
        
        # 파일 경로 설정
        model1_file = st.text_input(
            "🤖 모델 1 예측 파일:",
            value="results/audio-seed2018/best_hl_val_preds.jsonl",
            help="모델 1의 예측 결과 파일 경로"
        )
        
        model1_name = st.text_input(
            "📝 모델 1 이름:",
            value="Audio-Only Model",
            help="모델 1의 표시 이름"
        )
        
        model2_file = st.text_input(
            "🤖 모델 2 예측 파일:",
            value="results/both-seed2018/best_hl_val_preds.jsonl",
            help="모델 2의 예측 결과 파일 경로"
        )
        
        model2_name = st.text_input(
            "📝 모델 2 이름:",
            value="Video_Audio Model",
            help="모델 2의 표시 이름"
        )
        
        gt_file = st.text_input(
            "✅ 실제 정답 파일:",
            value="data/highlight_val_release.jsonl",
            help="Validation 정답 데이터 파일 경로"
        )
        
        # 필터 옵션
        comparison_type = st.selectbox(
            "🎯 비교 타입:",
            ["Model1이 더 좋은 경우", "Model2가 더 좋은 경우", "모든 비교"],
            help="어떤 비교 결과를 보고 싶은지 선택하세요"
        )
        
        min_diff = st.slider(
            "📊 최소 성능 차이 (AP@Avg):",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=1.0,
            help="이 값 이상의 성능 차이를 가진 쿼리만 표시 (백분율)"
        )
        
        # 페이지당 아이템 수
        items_per_page = st.slider(
            "📄 페이지당 아이템 수:",
            min_value=1,
            max_value=10,
            value=1
        )
        
        # 검색
        search_query = st.text_input(
            "🔍 쿼리 검색:",
            placeholder="검색어를 입력하세요...",
            help="쿼리 내용으로 검색"
        )
    
    # 데이터 로드 및 처리 (캐시됨)
    with st.spinner("📊 데이터를 로드하고 성능을 계산하는 중... (첫 번째 실행 시에만 시간이 걸립니다)"):
        result = process_and_compare_models(model1_file, model2_file, gt_file)
        
        if result[0] is None:
            st.error("❌ 데이터를 로드할 수 없습니다. 파일 경로를 확인해주세요.")
            return
        
        comparisons, gt_data_dict, total_gt_count = result
    
    if not comparisons:
        st.error("❌ 비교할 데이터가 없습니다.")
        return
    
    # 전체 통계
    model1_avg_ap = np.mean([c['model1_ap_avg'] for c in comparisons])
    model2_avg_ap = np.mean([c['model2_ap_avg'] for c in comparisons])
    avg_diff = np.mean([c['ap_diff'] for c in comparisons])
    
    model1_better_count = sum(1 for c in comparisons if c['ap_diff'] > 0.1)
    model2_better_count = sum(1 for c in comparisons if c['ap_diff'] < -0.1)
    similar_count = sum(1 for c in comparisons if abs(c['ap_diff']) <= 0.1)
    
    st.markdown("#### 📊 전체 비교 통계")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(f"{model1_name} 평균 AP@Avg", f"{model1_avg_ap * 100:.2f}")
    with col2:
        st.metric(f"{model2_name} 평균 AP@Avg", f"{model2_avg_ap * 100:.2f}")
    with col3:
        st.metric("평균 차이", f"{avg_diff * 100:.2f}")
    with col4:
        st.metric(f"{model1_name} 우위", f"{model1_better_count}개")
    with col5:
        st.metric(f"{model2_name} 우위", f"{model2_better_count}개")
    
    # 총 비교 수와 비슷한 성능 수
    col_total, col_similar = st.columns(2)
    with col_total:
        st.metric("총 비교 쿼리 수", len(comparisons))
    with col_similar:
        st.metric("비슷한 성능", f"{similar_count}개")
    
    # 성능 차이 분포 히스토그램 (캐시됨)
    @st.cache_data
    def create_performance_distribution_plot(comparisons_data):
        fig_dist = go.Figure()
        fig_dist.add_trace(
            go.Histogram(
                x=[c['ap_diff'] * 100 for c in comparisons_data], 
                name="AP@Avg 차이 분포", 
                nbinsx=30,
                marker_color='skyblue'
            )
        )
        fig_dist.update_layout(
            title="AP@Avg 성능 차이 분포 (Model1 - Model2)",
            xaxis_title="성능 차이 (%)",
            yaxis_title="쿼리 수",
            height=400
        )
        return fig_dist
    
    fig_dist = create_performance_distribution_plot(comparisons)
    st.plotly_chart(fig_dist, use_container_width=True)
    
    # 결과 필터링 및 정렬
    filtered_comparisons = comparisons
    
    # 비교 타입 필터
    if comparison_type == "Model1이 더 좋은 경우":
        filtered_comparisons = [c for c in filtered_comparisons if c['ap_diff'] > min_diff / 100]
    elif comparison_type == "Model2가 더 좋은 경우":
        filtered_comparisons = [c for c in filtered_comparisons if c['ap_diff'] < -min_diff / 100]
    else:  # 모든 비교
        filtered_comparisons = [c for c in filtered_comparisons if abs(c['ap_diff']) >= min_diff / 100]
    
    # 검색 필터
    if search_query:
        filtered_comparisons = [
            c for c in filtered_comparisons 
            if search_query.lower() in c['query'].lower()
        ]
    
    # 정렬 (성능 차이 절댓값 기준으로 내림차순)
    if comparison_type == "Model1이 더 좋은 경우":
        filtered_comparisons.sort(key=lambda x: x['ap_diff'], reverse=True)
    elif comparison_type == "Model2가 더 좋은 경우":
        filtered_comparisons.sort(key=lambda x: x['ap_diff'])
    else:
        filtered_comparisons.sort(key=lambda x: abs(x['ap_diff']), reverse=True)
    
    # 페이지네이션
    total_items = len(filtered_comparisons)
    total_pages = max(1, (total_items - 1) // items_per_page + 1)
    
    if 'page' not in st.session_state:
        st.session_state.page = 1
    
    st.markdown(f"**🎯 필터링된 결과: {total_items}개 (전체 {len(comparisons)}개 중)**")
    
    # 페이지 네비게이션
    col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
    
    with col1:
        if st.button("⬅️ 이전", disabled=st.session_state.page <= 1):
            st.session_state.page -= 1
    
    with col2:
        if st.button("⏮️ 처음"):
            st.session_state.page = 1
    
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
    
    with col4:
        if st.button("⏭️ 마지막"):
            st.session_state.page = total_pages
    
    with col5:
        if st.button("➡️ 다음", disabled=st.session_state.page >= total_pages):
            st.session_state.page += 1
    
    # 현재 페이지 데이터
    start_idx = (st.session_state.page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    current_comparisons = filtered_comparisons[start_idx:end_idx]
    
    st.markdown(f"**📊 페이지 {st.session_state.page} / {total_pages}** (아이템 {start_idx + 1}-{end_idx} / {total_items})")
    st.markdown("---")
    
    # 결과 표시
    for i, comparison in enumerate(current_comparisons):
        actual_idx = start_idx + i + 1
        display_comparison_item(comparison, actual_idx, model1_name, model2_name, gt_data_dict)
        
        if i < len(current_comparisons) - 1:
            st.markdown("---")

if __name__ == "__main__":
    main()
