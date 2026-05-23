import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api_v1', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='AiAnalysisReport',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                (
                    'channel',
                    models.ForeignKey(
                        blank=True, null=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name='ai_reports',
                        to='api_v1.channel',
                        verbose_name='채널',
                    ),
                ),
                ('channel_scope', models.CharField(db_index=True, default='all', max_length=100, verbose_name='분석 범위')),
                ('summary', models.TextField(verbose_name='종합 요약')),
                ('insights_json', models.JSONField(verbose_name='인사이트 JSON')),
                ('model_name', models.CharField(default='gemini-2.5-flash', max_length=100, verbose_name='AI 모델')),
                ('created_at', models.DateTimeField(auto_now_add=True, verbose_name='생성일시')),
            ],
            options={
                'verbose_name': 'AI 분석 리포트',
                'verbose_name_plural': 'AI 분석 리포트 목록',
                'db_table': 'ai_analysis_report',
                'ordering': ['-created_at'],
            },
        ),
    ]
