from django.urls import path, include
from django.contrib import admin
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.top_results_home, name='top_results_home'),
    path('evaluation_results/<str:strat_name>/<str:asset_ticker>/<int:perm_id>/',views.individual_results,name='Evaluation Results'),
    path('backtests/<str:strat_name>/<str:asset_ticker>/', views.strategy_backtests, name='backtests'),
    path('methodology/', views.methodology, name='methodology'),
    path('about/', views.about, name='about'),
    path('interpreting_results/', views.interpreting_results, name='interpreting_results'),
    path('disclaimer/', views.disclaimer, name='disclaimer'),
    path('__debug__/', include('debug_toolbar.urls')),
]