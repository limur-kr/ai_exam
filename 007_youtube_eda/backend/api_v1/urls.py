from django.urls import path
from . import views_api_youtube  as yt
from . import views_api_channel  as ch
from . import views_api_eda      as eda
from . import views_api_ai       as ai

urlpatterns = [
    # ── YouTube 데이터 수집 ──────────────────────────────────────────────────
    path('youtube/fetch-channel',               yt.fetch_channel,    name='fetch_channel'),
    path('youtube/fetch-videos',                yt.fetch_videos,     name='fetch_videos'),
    path('youtube/collection-log/<int:log_id>', yt.collection_status, name='collection_status'),

    # ── EDA 분석 ────────────────────────────────────────────────────────────
    path('eda',                             eda.eda_data,        name='eda_data'),
    path('ai-report',                       ai.ai_report,        name='ai_report'),

    # ── 채널 관리 ────────────────────────────────────────────────────────────
    path('channel/list',                        ch.channel_list,     name='channel_list'),
    path('channel/logs',                        ch.collection_logs,  name='collection_logs'),
    path('channel/<str:channel_id>',            ch.channel_detail,   name='channel_detail'),
    path('channel/<str:channel_id>/delete',     ch.channel_delete,   name='channel_delete'),
]
