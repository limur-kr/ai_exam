import json
import os
from collections import defaultdict
from datetime import date

from django.db.models import Avg, Sum
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from .models import AiAnalysisReport, Channel, Video, VideoStats


# ─── EDA 요약 데이터 빌드 ─────────────────────────────────────────────────────

def _build_eda_summary(channel_id: str) -> dict | None:
    if channel_id == 'all':
        channels = Channel.objects.all()
        videos   = Video.objects.all()
    else:
        channels = Channel.objects.filter(channel_id=channel_id)
        videos   = Video.objects.filter(channel__channel_id=channel_id)

    total_videos = videos.count()
    if total_videos == 0:
        return None

    stats_qs    = VideoStats.objects.filter(video__in=videos)
    total_views = stats_qs.aggregate(s=Sum('view_count'))['s'] or 0
    avg_views   = total_views // total_videos if total_videos else 0

    # 최근 12개월 업로드 추이
    today = date.today()
    month_list = []
    for i in range(11, -1, -1):
        m, y = today.month - i, today.year
        while m <= 0:
            m += 12
            y -= 1
        month_list.append((y, m))

    monthly: dict[tuple, int] = defaultdict(int)
    for v in videos.values('published_at'):
        if v['published_at']:
            monthly[(v['published_at'].year, v['published_at'].month)] += 1

    upload_trend = {
        f'{str(y)[-2:]}.{m:02d}': monthly.get((y, m), 0)
        for y, m in month_list
    }

    # 조회수 분포
    vc_list = list(stats_qs.values_list('view_count', flat=True))
    dist_labels = ['~1만', '1~5만', '5~10만', '10~50만', '50~100만', '100만+']
    dist = [0] * 6
    for vc in vc_list:
        vc = vc or 0
        if   vc < 10_000:    dist[0] += 1
        elif vc < 50_000:    dist[1] += 1
        elif vc < 100_000:   dist[2] += 1
        elif vc < 500_000:   dist[3] += 1
        elif vc < 1_000_000: dist[4] += 1
        else:                dist[5] += 1

    # Shorts 비율
    shorts_cnt  = videos.filter(is_short=True).count()
    regular_cnt = total_videos - shorts_cnt
    shorts_pct  = round(shorts_cnt / total_videos * 100, 1) if total_videos else 0

    # 요일별 업로드
    weekday_counts = [0] * 7
    for v in videos.values('published_at'):
        if v['published_at']:
            weekday_counts[v['published_at'].weekday()] += 1
    weekday_names = ['월', '화', '수', '목', '금', '토', '일']
    weekday_dict  = dict(zip(weekday_names, weekday_counts))
    best_day      = max(weekday_dict, key=lambda k: weekday_dict[k])

    # TOP 5 영상 (프롬프트 크기 절감)
    top5 = []
    for i, st in enumerate(
        stats_qs.select_related('video').order_by('-view_count')[:5]
    ):
        v = st.video
        top5.append({
            'rank':     i + 1,
            'title':    v.title[:60],
            'views':    st.view_count or 0,
            'likes':    st.like_count or 0,
            'comments': st.comment_count or 0,
            'is_short': v.is_short,
        })

    # 참여도
    agg = stats_qs.aggregate(
        al=Avg('like_count'), ac=Avg('comment_count'), av=Avg('view_count')
    )
    avg_likes    = round(agg['al'] or 0)
    avg_comments = round(agg['ac'] or 0)
    avg_view_r   = agg['av'] or 1
    like_rate    = round(avg_likes / avg_view_r * 100, 2)

    # Shorts vs 일반 좋아요율
    def _like_rate(qs):
        a = qs.aggregate(al=Avg('like_count'), av=Avg('view_count'))
        return round((a['al'] or 0) / (a['av'] or 1) * 100, 2)

    shorts_like_rate = _like_rate(stats_qs.filter(video__is_short=True))
    normal_like_rate = _like_rate(stats_qs.filter(video__is_short=False))

    return {
        'channel_count':     channels.count(),
        'total_videos':      total_videos,
        'total_views':       total_views,
        'avg_views':         avg_views,
        'upload_trend':      upload_trend,
        'views_dist':        dict(zip(dist_labels, dist)),
        'shorts_ratio':      {'regular': regular_cnt, 'shorts': shorts_cnt, 'shorts_pct': shorts_pct},
        'weekday':           weekday_dict,
        'best_upload_day':   best_day,
        'top5_videos':       top5,
        'engagement': {
            'avg_likes':        avg_likes,
            'avg_comments':     avg_comments,
            'like_rate':        like_rate,
            'shorts_like_rate': shorts_like_rate,
            'normal_like_rate': normal_like_rate,
        },
    }


# ─── Gemini API 호출 ──────────────────────────────────────────────────────────

def _call_gemini(eda: dict) -> dict:
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError(
            'google-generativeai 패키지가 설치되지 않았습니다. '
            'pip install google-generativeai 을 실행하세요.'
        )

    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key or api_key.startswith('여기에'):
        raise RuntimeError(
            'GEMINI_API_KEY가 설정되지 않았습니다. '
            'backend/.env 파일에 GEMINI_API_KEY=AIza... 를 입력하세요.'
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')

    prompt = f"""당신은 YouTube 채널 데이터 전문 분석가입니다. 아래의 실제 채널 통계 데이터를 분석하여 크리에이터에게 유용한 인사이트를 제공하세요.

## 분석 데이터
{json.dumps(eda, ensure_ascii=False, indent=2)}

## 응답 형식 (순수 JSON만 응답, 코드블록/마크다운 없음)
{{
  "summary": "2~3문장의 종합 분석 요약 (실제 수치 포함, 한국어)",
  "insights": [
    {{
      "icon": "📈",
      "label": "성장 트렌드",
      "badge": "긍정 또는 부정 또는 중립",
      "points": ["구체적 수치 기반 인사이트", "인사이트 2", "인사이트 3"]
    }},
    {{
      "icon": "🕐",
      "label": "최적 업로드 패턴",
      "badge": "추천",
      "points": ["인사이트 1", "인사이트 2", "인사이트 3"]
    }},
    {{
      "icon": "🎯",
      "label": "콘텐츠 전략 제안",
      "badge": "제안",
      "points": ["인사이트 1", "인사이트 2", "인사이트 3"]
    }},
    {{
      "icon": "⚠️",
      "label": "이상값 · 주의 사항",
      "badge": "주의",
      "points": ["인사이트 1", "인사이트 2", "인사이트 3"]
    }}
  ]
}}

규칙:
- 반드시 제공된 수치(숫자, 퍼센트)를 문장에 포함하세요.
- 모든 텍스트는 한국어로 작성하세요.
- JSON 외 다른 텍스트는 절대 작성하지 마세요."""

    response = model.generate_content(prompt)
    raw = response.text.strip()

    # 코드블록으로 감싼 경우 제거
    if raw.startswith('```'):
        lines = raw.split('\n')
        raw = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])

    return json.loads(raw)


# ─── View ─────────────────────────────────────────────────────────────────────

def _report_to_dict(report: AiAnalysisReport) -> dict:
    return {
        'summary':    report.summary,
        'insights':   report.insights_json,
        'created_at': report.created_at.strftime('%Y-%m-%d %H:%M'),
        'model_name': report.model_name,
    }


@csrf_exempt
def ai_report(request):

    # ── GET: 저장된 최신 AI 리포트 반환 ─────────────────────────────────────
    if request.method == 'GET':
        channel_id = request.GET.get('channel_id', 'all')
        report = AiAnalysisReport.objects.filter(channel_scope=channel_id).first()
        if report:
            return JsonResponse({'success': True, 'has_report': True,  'report': _report_to_dict(report)})
        return JsonResponse({'success': True,  'has_report': False, 'report': None})

    # ── POST: Gemini 호출 → DB 저장 → 결과 반환 ──────────────────────────────
    if request.method == 'POST':
        try:
            body = json.loads(request.body or b'{}')
            channel_id = body.get('channel_id', 'all')
        except Exception:
            channel_id = 'all'

        eda = _build_eda_summary(channel_id)
        if eda is None:
            return JsonResponse(
                {'success': False, 'error': '분석할 데이터가 없습니다. 먼저 채널 데이터를 수집하세요.'},
                status=400,
            )

        try:
            result = _call_gemini(eda)

            # 채널 FK 해결
            channel_obj = None
            if channel_id != 'all':
                try:
                    channel_obj = Channel.objects.get(channel_id=channel_id)
                except Channel.DoesNotExist:
                    pass

            saved = AiAnalysisReport.objects.create(
                channel=channel_obj,
                channel_scope=channel_id,
                summary=result['summary'],
                insights_json=result['insights'],
                model_name='gemini-2.5-flash',
            )

            return JsonResponse({
                'success':    True,
                'created_at': saved.created_at.strftime('%Y-%m-%d %H:%M'),
                'model_name': saved.model_name,
                **result,
            })

        except json.JSONDecodeError as e:
            return JsonResponse({'success': False, 'error': f'AI 응답 파싱 오류: {e}'}, status=500)
        except RuntimeError as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=500)
        except Exception as e:
            return JsonResponse({'success': False, 'error': f'AI 분석 중 오류 발생: {e}'}, status=500)

    return JsonResponse({'success': False, 'error': 'Method not allowed'}, status=405)
