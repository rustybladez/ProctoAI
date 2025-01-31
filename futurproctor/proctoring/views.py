# Django imports
from django.shortcuts import render, redirect  # For rendering templates and redirecting users
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse, HttpResponseRedirect  # For different types of HTTP responses
from django.contrib import messages  # For displaying success/error messages
from django.contrib.auth.decorators import login_required  # To restrict views to logged-in users
from django.contrib.auth.models import User  # For accessing the User model
from django.urls import reverse  # For generating URLs dynamically
from django.views.decorators.csrf import csrf_exempt  # For disabling CSRF protection on certain views

# Model imports
from .models import Student, Exam, CheatingEvent  # Models for handling student data, exam results, and cheating events
from django.utils.timezone import now  # For getting the current time in timezone-aware format
from django.core.files.base import ContentFile  # For working with file content in memory

# External imports
import base64  # For encoding and decoding base64 data (e.g., for image handling)
import cv2  # OpenCV for image and video processing
import numpy as np  # For working with arrays, especially in image processing
import face_recognition  # For face recognition operations
import json  # For parsing and handling JSON data

# Machine learning model imports
from .ml_models.object_detection import detectObject  # Object detection model for cheating detection

# Other imports
import threading  # For handling concurrent tasks (e.g., camera streaming or object detection)
from datetime import datetime  # For working with date and time objects
from datetime import datetime as now
from django.utils import timezone
import os
import cv2
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from.ml_models.audio_detection import audio_detection
from.ml_models.head_pose_detection import head_pose_detection
from.ml_models.object_detection import detectObject
from.ml_models.gaze_tracking import gaze_tracking



# Home page view
def home(request):
    return render(request, 'home.html')  # Render the home page


from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User
from django.core.files.base import ContentFile
from django.contrib.auth.hashers import make_password
import base64
import numpy as np
import cv2
from .models import Student


def registration(request):
    if request.method == 'POST':  # Process the form if submitted
        # Capture form data
        name = request.POST['name']
        address = request.POST['address']
        email = request.POST['email']
        password = request.POST['password']
        captured_photo = request.POST.get('photo_data')  # Captured photo in base64 format

        # Decode the base64 image
        try:
            # Remove the base64 prefix and decode the image
            img_data = base64.b64decode(captured_photo.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Extract face encoding
            face_encoding = get_face_encoding(image)  # Make sure this function returns a valid encoding
            if face_encoding is None:  # No face detected
                messages.error(request, "No face detected. Please try again.")
                return redirect('registration')
        except Exception as e:
            messages.error(request, f"Error processing image: {e}")
            return redirect('registration')

        # Check if email already exists
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already exists.")
            return redirect('registration')

        # Create the User instance
        try:
            user = User.objects.create(
                username=email,
                email=email,
                first_name=name.split(' ')[0],  # Use the first part of the name as first name
                last_name=' '.join(name.split(' ')[1:]) if ' ' in name else '',  # Remaining part as last name
                password=make_password(password),  # Hash the password
            )

            # Save the Student instance linked to the user
            student = Student(
                user=user,
                name=name,
                address=address,
                email=email,
                photo=ContentFile(img_data, name=f"{name}_photo.jpg"),  # Save the photo
                face_encoding=face_encoding.tolist(),  # Save the face encoding
            )
            student.save()

            # Set session data
            request.session['user_id'] = user.id
            request.session['user_name'] = user.first_name

            messages.success(request, "Registration successful!")
            return redirect('login')  # Redirect to login page

        except Exception as e:
            messages.error(request, f"Error creating user: {e}")
            return redirect('registration')

    return render(request, 'registration.html')  # Render the registration page


# Helper function to extract face encoding
def get_face_encoding(image):
    face_locations = face_recognition.face_locations(image)  # Detect faces in the image
    if not face_locations:
        return None  # Return None if no faces are detected
    return face_recognition.face_encodings(image, face_locations)[0]  # Return the first face encoding


# Helper function to match face encodings
def match_face_encodings(captured_encoding, stored_encoding):
    return face_recognition.compare_faces([stored_encoding], captured_encoding)[0]  # Compare encodings



from django.contrib.auth import authenticate, login as auth_login
from django.http import JsonResponse
from django.shortcuts import render
import base64
import numpy as np
import cv2
from .models import Student
# views.py
from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt  # Allow POST requests without CSRF token (for simplicity, use proper CSRF handling in production)
## Login View
def login(request):
    if request.method == "POST":
        email = request.POST.get('email')
        password = request.POST.get('password')
        captured_photo_data = request.POST.get('captured_photo')

        if not email or not password or not captured_photo_data:
            return JsonResponse({"success": False, "error": "Missing email, password, or captured photo."})

        try:
            captured_photo_data = captured_photo_data.split(',')[1]
            captured_photo = base64.b64decode(captured_photo_data)
            nparr = np.frombuffer(captured_photo, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            captured_encoding = get_face_encoding(image)

            if captured_encoding is None:
                return JsonResponse({"success": False, "error": "No face detected in the captured photo."})

            user = authenticate(request, username=email, password=password)
            if user is None:
                return JsonResponse({"success": False, "error": "Invalid email or password."})

            try:
                student = user.student
                stored_encoding = np.array(student.face_encoding)

                if match_face_encodings(captured_encoding, stored_encoding):
                    auth_login(request, user)

                    # Store student data in the session
                    request.session['student_id'] = student.id
                    request.session['student_name'] = student.name

                    return JsonResponse({
                        "success": True,
                        "redirect_url": "/dashboard/",
                        "student_name": student.name
                    })
                else:
                    return JsonResponse({"success": False, "error": "Face does not match our records."})

            except Student.DoesNotExist:
                return JsonResponse({"success": False, "error": "No student record associated with this account."})

        except Exception as e:
            return JsonResponse({"success": False, "error": f"Error processing image: {str(e)}"})

    return render(request, "login.html")


def logout_view(request):
    request.session.flush()  # Clear session data
    messages.success(request, "You have been logged out.")
    return redirect('login')


# Video feed generation for the webcam
def gen_frames():
    camera = cv2.VideoCapture(0)  # Open webcam
    while True:
        success, frame = camera.read()  # Read a frame
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)  # Encode the frame as JPEG
        frame = buffer.tobytes()

        # Yield the frame as part of a streaming response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()  # Release the webcam


# Video feed view
def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')  # Stream frames


# Stop video feed view
def stop_event (request):
    return JsonResponse({'status': 'success'})  # Dummy endpoint for stopping video

## Dashboard Views
def dashboard(request):
    user_name = request.session.get('user_name', 'Guest')
    context = {'user_name': user_name}
    return render(request, 'dashboard.html', context)



import time
from threading import Thread
# Global stop_event for thread control
stop_event = threading.Event()



# -------------------------Video Detection Thread----------------------------------
import io
import time
import logging
import json
import threading
from PIL import Image
import cv2
from django.shortcuts import render, HttpResponse, redirect
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile
from django.utils import timezone
from django.contrib import messages
from .models import CheatingEvent, CheatingImage, CheatingAudio, Exam, Student

logger = logging.getLogger(__name__)

# Global variables for warnings and background processes
warning = None
last_audio_detected_time = time.time()
stop_event = threading.Event()  # To stop background threads

# Function to process each frame
def process_frame(frame, request):
    """Process a single frame for cheating detection."""
    global warning
    labels, processed_frame, person_count, detected_objects = detectObject(frame)
    cheating_event = None  # Initialize cheating event

    # Extract object names detected in the frame
    detected_labels = [label for label, _ in labels]

    # Filter detected objects for cheating alerts
    cheating_objects = [label for label in detected_labels if label in ["cell phone", "book"]]

    # Trigger alert only if cell phone or book is detected
    if cheating_objects:
        warning = f"ALERT: {', '.join(cheating_objects)} detected!"  # Show only relevant objects
        cheating_event, _ = CheatingEvent.objects.get_or_create(
            student=request.user.student,
            cheating_flag=True,
            event_type="object_detected"
        )
        save_cheating_event(frame, request, cheating_event, cheating_objects)  # Save only relevant objects

# Alert only if more than 1 persons are detected
    if person_count > 1:
        warning = "ALERT: Multiple persons detected!"
        cheating_event, _ = CheatingEvent.objects.get_or_create(
            student=request.user.student,
            cheating_flag=True,
            event_type="multiple_persons"
        )
        save_cheating_event(frame, request, cheating_event, ["person"])

    # Check if the candidate is not looking at the screen
    gaze = gaze_tracking(frame)
    if gaze["gaze"] != "center":
        warning = "ALERT: Candidate not looking at the screen!"
        cheating_event, _ = CheatingEvent.objects.get_or_create(
            student=request.user.student,
            cheating_flag=True,
            event_type="gaze_detected"
        )
        save_cheating_event(frame, request, cheating_event, ["gaze_not_center"])  # Save only gaze-related info

    return processed_frame  # Ensure the processed frame is returned
# Function to process audio
def process_audio(request):
    """Continuously process audio for cheating detection."""
    global last_audio_detected_time, warning

    while True:
        audio = audio_detection()
        if audio["audio_detected"]:
            warning = "ALERT: Suspicious audio detected!"
            cheating_event, _ = CheatingEvent.objects.get_or_create(
                student=request.user.student,
                cheating_flag=True,
                event_type="audio_detected"
            )
            save_cheating_event(None, request, cheating_event, audio_data=audio["audio_data"])
            last_audio_detected_time = time.time()

        if time.time() - last_audio_detected_time > 5:
            warning = None

        time.sleep(1)

# Background processing for video
def background_processing(request):
    """Runs video processing in the background."""
    cap = cv2.VideoCapture(0)
    frame_count = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 2 == 0:
            process_frame(frame, request)
        
        frame_count += 1
        time.sleep(0.5)
    
    cap.release()

def save_cheating_event(frame, request, cheating_event, detected_objects=None, audio_data=None):
    """Save cheating event along with images and audio in the database."""
    try:
        
        # Save detected objects
        if detected_objects:
            cheating_event.detected_objects = detected_objects  # Save as JSON
            cheating_event.save()
        # Save up to 10 sample images per event
        if frame is not None and cheating_event.cheating_images.count() < 10:
            try:
                image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image_io = io.BytesIO()
                image_pil.save(image_io, format="JPEG", quality=85)
                image_content = image_io.getvalue()
                
                cheating_image = CheatingImage(event=cheating_event)
                cheating_image.image.save(
                    f"cheating_{time.time()}.jpg", 
                    ContentFile(image_content), 
                    save=True
                )
            except Exception as e:
                logger.error(f"Error processing image: {e}")
        
        # Save audio data
        if audio_data:
            try:
                cheating_audio = CheatingAudio(event=cheating_event)
                cheating_audio.audio.save(
                    f"cheating_audio_{time.time()}.wav", 
                    ContentFile(audio_data), 
                    save=True
                )
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
        
        logger.info(f"Cheating event saved for student {request.user.student.id}")
    
    except Exception as e:
        logger.error(f"Error saving cheating event: {e}")

@login_required
def exam(request):
    """Start the exam and initialize proctoring."""
    try:
        # Get the Student instance associated with the logged-in user
        student = request.user.student
    except Student.DoesNotExist:
        # Handle the case where the user does not have a linked Student instance
        return HttpResponse("Student profile not found. Please contact support.", status=404)

    # Get the tab switch count from the CheatingEvent model
    violations = CheatingEvent.objects.filter(student=student).first()
    tab_count = violations.tab_switch_count if violations else 0

    # Load exam questions from the JSON file
    try:
        with open("D://Futurproctor//futurproctor//proctoring//dummy_data//ai.json") as file:
            data = json.load(file)
        questions = data.get("questions", [])
    except FileNotFoundError:
        return HttpResponse("Error: Questions file not found!", status=404)
    except json.JSONDecodeError:
        return HttpResponse("Error: Failed to parse the questions file!", status=400)

    # Start background processing threads for video and audio monitoring
    global stop_event
    stop_event.clear()  # Reset the stop event flag
    threading.Thread(target=background_processing, args=(request,), daemon=True).start()
    threading.Thread(target=process_audio, args=(request,), daemon=True).start()

    # Render the exam template with questions and tab count
    return render(request, 'exam.html', {
        'questions': questions,
        'warning': warning,
        'tab_count': tab_count,
    })

# Submit exam
@login_required
def submit_exam(request):
    if request.method == 'POST':
        # Stop the background threads
        global stop_event
        stop_event.set()

        user = request.user

        # Load questions from ai.json
        try:
            with open('D:\\Futurproctor\\futurproctor\\proctoring\\dummy_data\\ai.json', 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            return HttpResponse("Error: Questions file not found!", status=404)
        except json.JSONDecodeError:
            return HttpResponse("Error: Failed to parse the questions file!", status=400)

        questions = data.get('questions', [])
        total_questions = len(questions)
        correct_answers = 0

        # Check answers
        for question in questions:
            question_id = question['id']
            user_answer = request.POST.get(f'answer_{question_id}')
            if user_answer == question['correct_answer']:
                correct_answers += 1

        # Save exam result
        exam = Exam(
            student=user.student,
            total_questions=total_questions,
            correct_answers=correct_answers,
            timestamp=timezone.now()
        )
        exam.save()

        # Redirect to success page
        messages.success(request, 'You have successfully completed the exam!')
        return redirect('exam_submission_success')

    return HttpResponse("Invalid request method.", status=400)

# Tab switch tracking
from django.http import JsonResponse
from django.shortcuts import redirect
from django.contrib.auth.decorators import login_required
from .models import CheatingEvent
import threading

stop_event = threading.Event()

@login_required
def record_tab_switch(request):
    if request.method == "POST":
        violations, created = CheatingEvent.objects.get_or_create(student=request.user.student)

        # Increment the tab switch count
        violations.tab_switch_count += 1

        # Set cheating_flag based on tab_switch_count
        violations.cheating_flag = violations.tab_switch_count > 0
        violations.save()

        # If tab switches exceed 5, take action
        if violations.tab_switch_count > 5:
            stop_event.set()  # Stop background threads
            return redirect('home')  # Redirect to home page

        return JsonResponse({
            "status": "updated",
            "count": violations.tab_switch_count,
            "cheating_flag": violations.cheating_flag,
            "message": f"Tab switch detected! Total switches: {violations.tab_switch_count}"
        }, status=200)

    return JsonResponse({"error": "Invalid request"}, status=400)

# Exam submission success page
def exam_submission_success(request):
    return render(request, 'exam_submission_success.html')

# Result page
@login_required
def result(request):
    user = request.user
    try:
        exam = Exam.objects.filter(student=user.student).latest('timestamp')
    except Exam.DoesNotExist:
        return HttpResponse("No exam found for this user", status=404)

    total_questions = exam.total_questions
    correct_answers = exam.correct_answers
    percentage = (correct_answers / total_questions) * 100

    context = {
        'user_name': user.username,
        'score': correct_answers,
        'total_questions': total_questions,
        'percentage': round(percentage, 2),
    }

    return render(request, 'result.html', context)

from django.http import JsonResponse

# Fetch warnings
@csrf_exempt
def get_warning(request):
    """Fetch real-time warnings for the exam page."""
    global warning
    return JsonResponse({'warning': warning})

# Streaming notifications to the proctor
def proctor_notifications(request):
    """Stream real-time cheating events to the proctor."""
    def event_stream():
        while True:
            events = CheatingEvent.objects.filter(cheating_flag=True).order_by('-timestamp')[:5]
            if events:
                yield f"data: {json.dumps([str(event) for event in events])}\n\n"
            time.sleep(5)
    
    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')


from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import user_passes_test
from django.shortcuts import render

def logout(request):
    return render(request,'home.html')

def is_admin(user):
    """Check if the user is an admin (staff or superuser)."""
    return user.is_authenticated and (user.is_staff or user.is_superuser)

def access_denied(request):
    return render(request, 'access_denied.html')

@user_passes_test(is_admin, login_url='/admin/login/')  # Redirect non-admin users to the admin login page
def admin_dashboard(request):
    students = Student.objects.all().prefetch_related('exams')
    for student in students:
        cheating_events = CheatingEvent.objects.filter(student=student).count()
        student.trust_score = 100 - (cheating_events * 10) # Example trust score calculation
    return render(request, 'admin_dashboard.html', {'students': students})

from collections import defaultdict

def report_page(request, student_id):
    student = get_object_or_404(Student, id=student_id)
    exam = student.exams.first()  # Assuming one exam per student for simplicity
    cheating_events = CheatingEvent.objects.filter(student=student)
    cheating_images = CheatingImage.objects.filter(event__student=student)
    cheating_audios = CheatingAudio.objects.filter(event__student=student)

    # Summarize cheating events
    cheating_events_summary = defaultdict(lambda: {
        'count': 0,
        'first_timestamp': None,
        'last_timestamp': None,
        'detected_objects': set(),
    })

    for event in cheating_events:
        summary = cheating_events_summary[event.event_type]
        summary['count'] += 1
        if not summary['first_timestamp']:
            summary['first_timestamp'] = event.timestamp
        summary['last_timestamp'] = event.timestamp
        summary['detected_objects'].update(event.detected_objects)

    # Convert sets to lists for template rendering
    for event_type, details in cheating_events_summary.items():
        details['detected_objects'] = list(details['detected_objects'])

    return render(request, 'report_page.html', {
        'student': student,
        'exam': exam,
        'cheating_events_summary': dict(cheating_events_summary),  # Convert defaultdict to a regular dict
        'cheating_images': cheating_images,
        'cheating_audios': cheating_audios,
    })