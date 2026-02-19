from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import Document, ChatMessage, ChatSession
from .utils import extract_text_from_pdf, process_pdf, answer_question, update_reward_from_feedback
from django.http import JsonResponse
from django.utils import timezone


@login_required
def upload_and_process(request):
    if request.method == "POST":
        pdf_file = request.FILES.get("pdf")

        if pdf_file:
            document = Document.objects.create(
                file=pdf_file,
                user=request.user
            )

            # --- RAG PIPELINE ---
            text = extract_text_from_pdf(document.file.path)
            process_pdf(
                text=text,
                user_id=request.user.id,
                document_id=document.id
            )

            document.processed = True
            document.save()

            return redirect("rag-chat", document_id=document.id)
    else:
        request.session.pop("chat_session_id", None)
    return render(request, "ragapp/upload.html", {"title": "Upload PDF"})


@login_required
def chat(request, document_id):
    document = Document.objects.get(id=document_id, user=request.user)

    session_id = request.session.get("chat_session_id")

    if session_id:
        session = ChatSession.objects.filter(
            id=session_id, user=request.user
        ).first()
    else:
        session = None

    if not session:
        session = ChatSession.objects.create(
            user=request.user,
            document=document,
            title=f"Chat on {document.file.name}"
        )
        request.session["chat_session_id"] = session.id

    chats = ChatMessage.objects.filter(
        session=session
    ).order_by("created_at")

    if request.method == "POST":
        question = request.POST.get("question")

        answer = answer_question(
            question=question,
            user_id=request.user.id,
            document_id=document.id,
            chat_history=chats
        )

        ChatMessage.objects.create(
            user=request.user,
            document=document,
            session=session,
            question=question,
            answer=answer,
            rl_state=answer.get("rl_state"),
            rl_action=answer.get("rl_action")
        )


        return redirect("rag-chat", document_id=document.id)
    
    print(chats)

    return render(request, "ragapp/chat.html", {
        "chats": chats,
        "document_id": document_id,
        "session": session,
        "title": "Chat Page"
    })


@login_required
def session_history(request):
    sessions = ChatSession.objects.filter(
        user=request.user
    ).order_by("-created_at")

    return render(request, "ragapp/session_history.html", {
        "sessions": sessions,
        "title": "Session History"
    })


@login_required
def session_detail(request, session_id):
    session = ChatSession.objects.get(
        id=session_id,
        user=request.user
    )

    chats = ChatMessage.objects.filter(
        session=session
    ).order_by("created_at")

    return render(request, "ragapp/session_detail.html", {
        "session": session,
        "chats": chats,
        "title": "Session Detail"
    })

@login_required
def delete_session(request, session_id):
    session = get_object_or_404(
        ChatSession,
        id=session_id,
        user=request.user  # 🔒 security check
    )

    if request.method == "POST":
        session.delete()

    return redirect("session-history")

@login_required
def delete_all_sessions(request):
    if request.method == "POST":
        ChatSession.objects.filter(user=request.user).delete()

    return redirect("session-history")
@login_required
def submit_feedback(request):
    if request.method == "POST":
        message_id = request.POST.get("message_id")
        rating = request.POST.get("rating")

        if not message_id or not rating:
            return JsonResponse({"error": "Invalid data"}, status=400)

        rating = int(rating)

        message = get_object_or_404(
            ChatMessage,
            id=message_id,
            session__user=request.user
        )

        if message.rating is not None:
            return JsonResponse({"error": "Already rated"}, status=400)

        # Save rating
        message.rating = rating
        message.rated_at = timezone.now()
        message.save()

        # 🔥 Update RL reward using human feedback
        update_reward_from_feedback(message, rating)

        return JsonResponse({"success": True})

    return JsonResponse({"error": "Invalid request"}, status=400)
