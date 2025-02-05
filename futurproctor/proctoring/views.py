# Django Core Imports
from django.shortcuts import render, redirect, get_object_or_404  # Rendering templates, redirecting, and fetching objects
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse, HttpResponseRedirect  # Handling HTTP responses
from django.contrib import messages  # Displaying success/error messages
from django.contrib.auth.decorators import login_required, user_passes_test  # Restricting views to logged-in users
from django.contrib.auth.models import User  # Accessing Django's built-in User model
from django.contrib.auth.hashers import make_password  # Hashing passwords securely
from django.contrib.auth import authenticate, login as auth_login  # Handling user authentication
from django.urls import reverse  # Generating dynamic URLs
from django.views.decorators.csrf import csrf_exempt  # Disabling CSRF protection for certain views (Use cautiously)
from django.utils.timezone import now  # Getting timezone-aware current time
from django.core.files.base import ContentFile  # Handling in-memory file storage

# Models
from .models import Student, Exam, CheatingEvent, CheatingImage, CheatingAudio  # Importing custom models

# External Library Imports
import os  # Operating system utilities (e.g., file handling)
import json  # JSON handling (e.g., parsing request data)
import threading  # Running concurrent tasks (e.g., real-time monitoring)
import base64  # Encoding and decoding base64 (used for image handling)
import numpy as np  # Numerical operations, especially for image processing
import cv2  # OpenCV for computer vision tasks (e.g., face recognition)
import logging  # Logging errors and system activity
import time  # Time-based operations (e.g., timestamps)
from PIL import Image  # Image processing using the Pillow library
import io  # Handling in-memory file operations

# Machine Learning Imports (Custom AI Models for Proctoring)
from .ml_models.object_detection import detectObject  # Detecting objects in the exam environment
from .ml_models.audio_detection import audio_detection  # Detecting external sounds for cheating detection
from .ml_models.facial_detections import detectFace  # Monitoring head movements (e.g., looking away)
# from .ml_models.gaze_tracking import gaze_tracking  # Tracking eye gaze to detect focus and distractions

# Fix: Import face_recognition (Previously missing)
import face_recognition  # Used for facial recognition, comparing student faces with stored images

# Fix: Proper datetime handling for Nepal Time Zone (Asia/Kathmandu)
import pytz  # For timezone handling
from datetime import datetime  # Standard date and time handling

# Define Nepal Time Zone
NEPAL_TZ = pytz.timezone('Asia/Kathmandu')

# Function to get Nepal's current time
def get_nepal_time():
    """
    Returns the current time in Nepal's timezone.
    This ensures all timestamps are consistent with the local time.
    """
    return datetime.now(NEPAL_TZ)


# Home page view
def home(request):
    """
    Renders the home page of the application.
    This is the entry point for users visiting the site.
    """
    return render(request, 'home.html')  # Render the home page


# Registration View
def registration(request):
    """
    Handles user registration, including:
    - Capturing form data (name, address, email, password, and photo)
    - Decoding and processing a base64-encoded image
    - Extracting face encoding using face recognition
    - Creating a new User and Student instance
    - Handling errors and displaying messages
    """
    if request.method == 'POST':  # Check if form is submitted
        # Retrieve form data
        name = request.POST['name']
        address = request.POST['address']
        email = request.POST['email']
        password = request.POST['password']
        captured_photo = request.POST.get('photo_data')  # Base64 image data

        try:
            # Decode the base64 image (photo_data comes in "data:image/png;base64,ENCODED_DATA")
            img_data = base64.b64decode(captured_photo.split(',')[1])
            nparr = np.frombuffer(img_data, np.uint8)  # Convert to numpy array
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Convert to OpenCV image

            # Extract face encoding from the image
            face_encoding = get_face_encoding(image)  # Function should return a list or None
            if face_encoding is None:  # No face detected
                messages.error(request, "No face detected. Please try again.")
                return redirect('registration')
        except Exception as e:
            messages.error(request, f"Error processing image: {e}")
            return redirect('registration')

        # Check if the email is already registered
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already exists.")
            return redirect('registration')

        try:
            # Create a new User instance
            user = User.objects.create(
                username=email,  # Use email as username for uniqueness
                email=email,
                first_name=name.split(' ')[0],  # Extract first name
                last_name=' '.join(name.split(' ')[1:]) if ' ' in name else '',  # Extract last name if available
                password=make_password(password),  # Hash password for security
            )

            # Create a linked Student instance
            student = Student(
                user=user,
                name=name,
                address=address,
                email=email,
                photo=ContentFile(img_data, name=f"{name}_photo.jpg"),  # Save the uploaded image
                face_encoding=face_encoding.tolist(),  # Convert NumPy array to list
            )
            student.save()

            # Store user session data
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
    """
    Extracts face encoding from an image using the face_recognition library.
    - Detects faces in the image.
    - Returns the encoding of the first face found.
    - Returns None if no faces are detected.
    """
    face_locations = face_recognition.face_locations(image)  # Detect faces in the image
    if not face_locations:
        return None  # Return None if no faces are detected
    return face_recognition.face_encodings(image, face_locations)[0]  # Return the first face encoding

# Helper function to match face encodings
def match_face_encodings(captured_encoding, stored_encoding):
    return face_recognition.compare_faces([stored_encoding], captured_encoding)[0]  # Compare encodings


#Login View
@csrf_exempt  # Allow POST requests without CSRF token (for simplicity, use proper CSRF handling in production)
def login(request):
    """
    Handles user login with email, password, and facial recognition.
    - Authenticates the user using email and password.
    - Compares the captured photo with the stored face encoding.
    - Logs the user in if all checks pass.
    - Returns JSON responses for success or failure.
    """
    if request.method == "POST":
        # Retrieve form data
        email = request.POST.get('email')
        password = request.POST.get('password')
        captured_photo_data = request.POST.get('captured_photo')

        # Validate required fields
        if not email or not password or not captured_photo_data:
            return JsonResponse({"success": False, "error": "Missing email, password, or captured photo."})

        try:
            # Decode the base64 image (remove the "data:image/png;base64," prefix)
            captured_photo_data = captured_photo_data.split(',')[1]
            captured_photo = base64.b64decode(captured_photo_data)

            # Convert the image to a NumPy array and decode it using OpenCV
            nparr = np.frombuffer(captured_photo, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Extract face encoding from the captured image
            captured_encoding = get_face_encoding(image)
            if captured_encoding is None:
                return JsonResponse({"success": False, "error": "No face detected in the captured photo."})

            # Authenticate the user using email and password
            user = authenticate(request, username=email, password=password)
            if user is None:
                return JsonResponse({"success": False, "error": "Invalid email or password."})

            try:
                # Fetch the associated student record
                student = user.student
                stored_encoding = np.array(student.face_encoding)

                # Compare the captured face encoding with the stored encoding
                if match_face_encodings(captured_encoding, stored_encoding):
                    # Log the user in
                    auth_login(request, user)

                    # Store student data in the session for future use
                    request.session['student_id'] = student.id
                    request.session['student_name'] = student.name

                    # Return a success response with redirect URL and student name
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
            # Handle any unexpected errors during the login process
            return JsonResponse({"success": False, "error": f"Error processing image: {str(e)}"})

    # Render the login page for GET requests
    return render(request, "login.html")

# Logout View 
def logout_view(request):
    """
    Handles user logout.
    - Clears all session data.
    - Displays a success message.
    - Redirects the user to the home page.
    """
    request.session.flush()  # Clear all session data
    messages.success(request, "You have been logged out.")  # Display a success message
    return redirect('home')  # Redirect to the home page

# Video feed generation for the webcam
def gen_frames():
    """
    Generates a live video feed from the webcam.
    - Captures frames from the webcam using OpenCV.
    - Encodes each frame as a JPEG image.
    - Yields the frames as a streaming response for real-time display in the browser.
    """
    camera = cv2.VideoCapture(0)  # Open the default webcam (index 0)
    if not camera.isOpened():  # Check if the webcam was successfully opened
        raise RuntimeError("Could not open webcam.")

    while True:
        success, frame = camera.read()  # Read a frame from the webcam
        if not success:
            break  # Exit the loop if the frame cannot be read

        # Encode the frame as a JPEG image
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()  # Convert the frame to bytes

        # Yield the frame as part of a streaming response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the webcam when the loop ends
    camera.release()


# Video feed view
def video_feed(request):
    """
    Streams the live video feed to the browser.
    - Uses the `gen_frames` generator to fetch frames from the webcam.
    - Returns a `StreamingHttpResponse` with the appropriate content type for real-time video streaming.
    """
    return StreamingHttpResponse(
        gen_frames(),  # Use the generator to stream frames
        content_type='multipart/x-mixed-replace; boundary=frame'  # Required for live video streaming
    )


# Stop video feed view
def stop_event(request):
    """
    Dummy endpoint for stopping the video feed.
    - Can be extended to handle cleanup or other actions when the video feed is stopped.
    - Returns a JSON response indicating success.
    """
    return JsonResponse({'status': 'success'})  # Simple response for stopping the video feed

#Dashboard View
@login_required
def dashboard(request):
    """
    Renders the dashboard page for authenticated users.
    - Retrieves the user's name from the session.
    - Displays personalized content on the dashboard.
    - Handles cases where the user is not logged in (defaults to 'Guest').
    """
    # Retrieve the user's name from the session (default to 'Guest' if not found)
    user_name = request.session.get('user_name', 'Guest')

    # Prepare context data to pass to the template
    context = {
        'user_name': user_name,  # Pass the user's name to the template
    }

    # Render the dashboard template with the context data
    return render(request, 'dashboard.html', context)



# -------------------------Video Detection Thread----------------------------------

from django.utils import timezone
# Define Nepal Time Zone
NEPAL_TZ = pytz.timezone('Asia/Kathmandu')

# Helper function to get Nepal time
def get_nepal_time():
    return timezone.now().astimezone(NEPAL_TZ)

# Global variables
warning = None
last_audio_detected_time = time.time()
stop_event = threading.Event()  # Event to stop background processing
frame_buffer = []  # Buffer to store frames for processing
logger = logging.getLogger(__name__)

# Frame Processing for Cheating Detection
def process_frame(frame, request):
    """
    Process a single frame for cheating detection.
    - Detects multiple persons.
    - Detects cheating objects (cell phone, book).
    - Saves cheating events to the database.
    - Returns the processed frame.
    """
    global warning

    # Detect cheating objects
    labels, processed_frame = detectObject(frame)
    detected_labels = [label for label, _ in labels]
    cheating_objects = [label for label in detected_labels if label in ["cell phone", "book"]]

    # Detect faces and multiple persons
    faceCount, annotated_frame = detectFace(frame)

    # Trigger alert if multiple faces are detected
    if faceCount > 1:
        warning = "ALERT: Multiple Faces Detected!"
        cheating_event, _ = CheatingEvent.objects.get_or_create(
            student=request.user.student,
            cheating_flag=True,
            event_type="multiple_faces_detected",
            timestamp=get_nepal_time()
        )
        save_cheating_event(frame, request, cheating_event)

    # Trigger alert if cell phone or book is detected
    if cheating_objects:
        warning = f"ALERT: {', '.join(cheating_objects)} detected!"
        cheating_event, _ = CheatingEvent.objects.get_or_create(
            student=request.user.student,
            cheating_flag=True,
            event_type="object_detected",
            timestamp=get_nepal_time()
        )
        save_cheating_event(frame, request, cheating_event, detected_objects=cheating_objects)

    return annotated_frame  # Return the frame with all detections

# Audio Processing for Cheating Detection
def process_audio(request):
    """
    Continuously process audio for cheating detection.
    - Detects suspicious audio and saves it to the database.
    """
    global last_audio_detected_time, warning

    while not stop_event.is_set():
        audio = audio_detection()  # Detect audio
        if audio["audio_detected"]:
            warning = "ALERT: Suspicious audio detected!"
            cheating_event, _ = CheatingEvent.objects.get_or_create(
                student=request.user.student,
                cheating_flag=True,
                event_type="audio_detected",
                timestamp=get_nepal_time()
            )
            save_cheating_event(None, request, cheating_event, audio_data=audio["audio_data"])
            last_audio_detected_time = time.time()

        time.sleep(1)  # Add a delay to reduce CPU usage

# Background processing for video
def background_processing(request):
    """
    Runs video processing in the background.
    - Captures frames from the webcam.
    - Processes every alternate frame for cheating detection.
    - Releases the webcam when the thread is stopped.
    """
    global frame_buffer

    # Open the webcam (index 0 is the default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open webcam.")
        return

    frame_count = 0  # Counter to track frames

    while not stop_event.is_set():  # Continue until the stop event is triggered
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            logger.error("Failed to capture frame from webcam.")
            break  # Exit the loop if the frame cannot be read

        # Add frame to buffer
        frame_buffer.append(frame)
        if len(frame_buffer) > 10:  # Limit buffer size to 10 frames
            frame_buffer.pop(0)

        # Process every alternate frame to reduce CPU usage
        if frame_count % 2 == 0:
            process_frame(frame, request)  # Process the frame for cheating detection

        frame_count += 1  # Increment the frame counter
        time.sleep(1)  # Add a small delay to reduce CPU usage

    # Release the webcam when the loop ends
    cap.release()
    logger.info("Webcam released and background processing stopped.")

# Saving the Cheating Event to the database with image and Audio Data
def save_cheating_event(frame, request, cheating_event, detected_objects=None, audio_data=None):
    """
    Save cheating event along with images and audio in the database.
    - Stores up to 10 images per event.
    - Handles both image and audio data.
    """
    try:
        # Save detected objects
        if detected_objects:
            cheating_event.detected_objects = json.dumps(detected_objects)  # Save as JSON
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
                print(f"Audio saved: {cheating_audio.audio.path}") 
            except Exception as e:
                logger.error(f"Error processing audio: {e}")

        logger.info(f"Cheating event saved for student {request.user.student.id}")

    except Exception as e:
        logger.error(f"Error saving cheating event: {e}")

# Exam View
@login_required
def exam(request):
    """
    Start the exam and initialize proctoring.
    - Loads exam questions.
    - Starts background threads for video and audio monitoring.
    - Renders the exam template with questions and tab count.
    """
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
            timestamp=get_nepal_time()
        )
        exam.save()

        # Redirect to success page
        messages.success(request, 'You have successfully completed the exam!')
        return redirect('exam_submission_success')

    return HttpResponse("Invalid request method.", status=400)

stop_event = threading.Event()


# Set up logging
logger = logging.getLogger(__name__)

@login_required
def record_tab_switch(request):
    if request.method == "POST":
        # Get the current student
        student = request.user.student
        logger.info(f"Student: {student}")

        # Get the active exam for the student
        active_exam = Exam.objects.filter(student=student, status='ongoing').first()
        if not active_exam:
            logger.error("No active exam found for the student")
            return JsonResponse({"error": "No active exam found for the student"}, status=400)

        logger.info(f"Active Exam: {active_exam}")

        # Get or create a CheatingEvent for the student and exam
        cheating_event, created = CheatingEvent.objects.get_or_create(
            student=student,
            exam=active_exam,
            event_type='tab_switch',  # Specify the event type
            defaults={
                'cheating_flag': False,
                'tab_switch_count': 0,
            }
        )

        logger.info(f"Cheating Event: {cheating_event}, Created: {created}")

        # Increment the tab switch count
        cheating_event.tab_switch_count += 1
        logger.info(f"Updated Tab Switch Count: {cheating_event.tab_switch_count}")

        # Set cheating_flag based on tab_switch_count
        cheating_event.cheating_flag = cheating_event.tab_switch_count > 0
        logger.info(f"Cheating Flag: {cheating_event.cheating_flag}")

        # Save the updated CheatingEvent
        cheating_event.save()
        logger.info("Cheating Event saved successfully")

        # If tab switches exceed 5, take action
        if cheating_event.tab_switch_count > 5:
            stop_event.set()  # Stop background threads (ensure stop_event is defined)
            logger.info("Tab switches exceeded 5, stopping exam")
            return redirect('home')  # Redirect to home page

        # Return a JSON response with the updated count and flag
        return JsonResponse({
            "status": "updated",
            "count": cheating_event.tab_switch_count,
            "cheating_flag": cheating_event.cheating_flag,
            "message": f"Tab switch detected! Total switches: {cheating_event.tab_switch_count}"
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



def logout(request):
    return render(request,'home.html')

def is_admin(user):
    """Check if the user is an admin (staff or superuser)."""
    return user.is_authenticated and (user.is_staff or user.is_superuser)

def access_denied(request):
    return render(request, 'access_denied.html')

def exam_score(exam):
    """Calculate the exam score percentage."""
    if exam.total_questions and exam.total_questions > 0 and exam.correct_answers is not None:
        return round((exam.correct_answers / exam.total_questions) * 100, 2)  # Round to 2 decimal places
    return 0.0  # Default score if no data is available


from django.db.models import Count, Sum
from django.contrib.auth.decorators import user_passes_test

@user_passes_test(lambda u: u.is_superuser, login_url='/admin/login/')
def admin_dashboard(request):
    # Fetch students with their exam and cheating event counts
    students = Student.objects.annotate(
        exam_count=Count('exams'),
        cheating_event_count=Count('exams__cheating_events')
    ).prefetch_related('exams')

    # Calculate trust score and exam score for each student
    for student in students:
        student.trust_score = max(0, 100 - (student.cheating_event_count * 10))  # Trust score logic
        for exam in student.exams.all():
            if not exam.percentage_score:
                exam.percentage_score = exam_score(exam)
                exam.save()

    return render(request, 'admin_dashboard.html', {'students': students})


import base64
from django.shortcuts import get_object_or_404

@user_passes_test(lambda u: u.is_superuser, login_url='/admin/login/')
def report_page(request, student_id):
    # Fetch the student and their exam
    student = get_object_or_404(Student, id=student_id)
    exam = student.exams.first()

    # Fetch cheating events, images, and audios for the student
    cheating_events = CheatingEvent.objects.filter(student=student).prefetch_related('cheating_images', 'cheating_audios')
    cheating_images = CheatingImage.objects.filter(event__student=student)
    cheating_audios = CheatingAudio.objects.filter(event__student=student)

    # Calculate tab switch count and gaze count
    tab_switch_count = sum(event.tab_switch_count for event in cheating_events if event.event_type == "tab_switch")
    gaze_count = sum(1 for event in cheating_events if event.event_type == "gaze")

    # Process detected objects
    detected_objects = {}
    for event in cheating_events:
        for obj in event.detected_objects:
            detected_objects[obj] = detected_objects.get(obj, 0) + 1

    # Determine cheating status
    cheating = (
        tab_switch_count > 0
        or gaze_count > 100
        or any(obj in detected_objects for obj in ["cell phone", "book", "multiple_person"])
    )

    # Process audio files (convert bytes to base64 for playback)
    audio_files = []
    for audio in cheating_audios:
        if audio.audio:
            with open(audio.audio.path, "rb") as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
                audio_files.append({
                    "id": audio.id,
                    "base64": audio_base64,
                    "timestamp": audio.timestamp,
                })

    # Prepare data for the template
    context = {
        'student': student,
        'exam': exam,
        'cheating': cheating,
        'tab_switch_count': tab_switch_count,
        'gaze_count': gaze_count,
        'detected_objects': detected_objects,
        'detected_objects_count': sum(detected_objects.values()),
        'cheating_images': cheating_images,
        'audio_files': audio_files,
        'cheating_events': cheating_events,
    }

    return render(request, 'report_page.html', context)