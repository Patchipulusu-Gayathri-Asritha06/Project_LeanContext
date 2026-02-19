from django.urls import path
from . import views

urlpatterns= [
    path("upload/", views.upload_and_process, name= "rag-upload"),
    path("chat/<int:document_id>/", views.chat, name="rag-chat"),
    path("history/", views.session_history, name="session-history"),
    path("history/<int:session_id>/", views.session_detail, name="session-detail"),
    path("session/<int:session_id>/delete/", views.delete_session, name="delete-session"),
    path("sessions/delete-all/", views.delete_all_sessions, name="delete-all-sessions"),
    path("submit-feedback/", views.submit_feedback, name="submit-feedback"),
]