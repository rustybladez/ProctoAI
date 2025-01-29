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
from.ml_models.head_pose_estimation import head_pose_detection



# Home page view
def home(request):
    return render(request, 'home.html')  # Render the home page


from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect
from django.contrib import messages
import base64
import numpy as np
import cv2
## Registration View
def registration(request):
    if request.method == 'POST':  # Process the form if submitted
        name = request.POST['name']
        address = request.POST['address']
        email = request.POST['email']
        password = request.POST['password']
        captured_photo = request.POST.get('photo_data')  # Captured photo in base64 format

        # Decode the base64 image
        try:
            img_data = base64.b64decode(captured_photo.split(',')[1])  # Remove prefix and decode
            nparr = np.frombuffer(img_data, np.uint8)  # Convert to NumPy array
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Convert to an image

            # Extract face encoding
            face_encoding = get_face_encoding(image)
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

        # Create a User instance
        try:
            user = User.objects.create(
                username=email,
                email=email,
                first_name=name.split(' ')[0],  # Use the first part of the name as first name
                last_name=' '.join(name.split(' ')[1:]) if ' ' in name else '',  # Remaining part as last name
                password=make_password(password),  # Hash the password
            )

            # Save Student instance linked to the user
            student = Student(
                user=user,
                name=name,
                address=address,
                email=email,
                photo=ContentFile(img_data, name=f"{name}_photo.jpg"),
                face_encoding=face_encoding.tolist(),  # Save face encoding as a list
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
## Login View
def login(request):
    if request.method == "POST":
        # Retrieve form data
        email = request.POST.get('email')
        password = request.POST.get('password')
        captured_photo_data = request.POST.get('captured_photo')  # Captured photo in base64

        # Ensure all required fields are provided
        if not email or not password or not captured_photo_data:
            return JsonResponse({"success": False, "error": "Missing email, password, or captured photo."})

        try:
            # Decode the captured photo from base64
            captured_photo_data = captured_photo_data.split(',')[1]  # Remove the prefix (e.g., "data:image/jpeg;base64,")
            captured_photo = base64.b64decode(captured_photo_data)

            # Convert the decoded photo to a NumPy array for processing
            nparr = np.frombuffer(captured_photo, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Extract face encoding from the captured photo
            captured_encoding = get_face_encoding(image)
            if captured_encoding is None:  # Handle cases where no face is detected
                return JsonResponse({"success": False, "error": "No face detected in the captured photo."})
            
            # Authenticate user by email and password using Django's User model
            user = authenticate(request, username=email, password=password)
            if user is None:  # Invalid email or password
                return JsonResponse({"success": False, "error": "Invalid email or password."})

            # Retrieve the associated Student object
            try:
                student = user.student  # Access the related Student instance
                stored_encoding = np.array(student.face_encoding)  # Convert stored encoding to a NumPy array

                # Compare the captured face encoding with the stored encoding
                if match_face_encodings(captured_encoding, stored_encoding):
                    # Log the user in and store session data upon successful login
                    auth_login(request, user)

                    # Store the student data in the session
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

    # Render the login page for GET requests
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
import json
import time
import threading
import datetime
import asyncio
import websockets
from django.shortcuts import render
from django.http import HttpResponse




def exam(request):
    # if request.method == 'POST':
        async def proctoring_system(username):
            cap = cv2.VideoCapture(0)  # Open webcam
            cheat_count = 0
            consecutive_cheat_events = 0
            alert_threshold = 3  # Threshold for alerting cheating
            real_time_alert = False
            frame_skip = 30  # Process one frame per second (assuming 30 FPS)
            frame_count = 0
            head_pose_buffer = []  # Buffer to track head pose over time
            buffer_duration = 30  # Number of frames to track for head pose (30 sec)

            async def send_ws_message(message):
                """Send a real-time message to the frontend via WebSocket."""
                async with websockets.connect('ws://localhost:8000/ws/exam_detections/') as websocket:
                    await websocket.send(json.dumps(message))

            def detect_cheating(detected_objects, head_pose, person_count, audio_detected):
                nonlocal cheat_count, consecutive_cheat_events, real_time_alert, head_pose_buffer
                cheated = False

                if any(obj in ['cell phone', 'phone', 'book'] for obj, _ in detected_objects):
                    cheat_count += 1
                    consecutive_cheat_events += 1
                    cheated = True
                    asyncio.run(send_ws_message({"alert": "Prohibited object detected!"}))

                if person_count > 1:
                    cheat_count += 1
                    consecutive_cheat_events += 1
                    cheated = True
                    asyncio.run(send_ws_message({"alert": "Multiple faces detected!"}))

                if audio_detected == "Suspicious audio detected":
                    cheat_count += 1
                    consecutive_cheat_events += 1
                    cheated = True
                    asyncio.run(send_ws_message({"alert": "Suspicious sound detected!"}))

                if head_pose in ['left', 'right', 'down']:
                    head_pose_buffer.append(head_pose)
                    if len(head_pose_buffer) > buffer_duration:
                        head_pose_buffer.pop(0)
                    if all(h == head_pose for h in head_pose_buffer):
                        asyncio.run(send_ws_message({"alert": "Please focus on the screen!"}))
                        head_pose_buffer.clear()
                else:
                    head_pose_buffer.clear()

                if not cheated:
                    consecutive_cheat_events = 0

                if consecutive_cheat_events >= alert_threshold:
                    real_time_alert = True
                    asyncio.run(send_ws_message({"alert": "Cheating detected!"}))

                return cheated

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame from camera.")
                    break

                frame_count += 1
                if frame_count % frame_skip == 0:  # Process every 30th frame
                    current_time = datetime.datetime.now().strftime("%H:%M:%S")

                    labels, processed_frame, person_count = detectObject(frame)

                    head_pose = head_pose_detection(frame) if person_count > 0 else "No head detected."
                    detected_objects, _ = detectObject(frame)
                    audio_status = audio_detection()

                    cheating = detect_cheating(detected_objects, head_pose, person_count, audio_status)

                    message = {
                        "timestamp": current_time,
                        "cheating": cheating,
                        "objects": detected_objects,
                        "head_pose": head_pose,
                        "person_count": person_count,
                        "audio": audio_status
                    }
                    asyncio.run(send_ws_message(message))

                key = cv2.waitKey(1)
                if key == 27:  # Press 'Esc' to exit
                    break

            cap.release()
            cv2.destroyAllWindows()

        proctoring_thread = threading.Thread(target=lambda: asyncio.run(proctoring_system("student")))
        proctoring_thread.start()

        try:
            with open("D://Futurproctor//futurproctor//proctoring//dummy_data//ai.json") as file:
                data = json.load(file)
            questions = data.get("questions", [])
            return render(request, 'exam.html', {'questions': questions, 'message': "Exam Started!"})
        except FileNotFoundError:
            return HttpResponse("Error: Questions file not found!", status=404)
        except json.JSONDecodeError:
            return HttpResponse("Error: Failed to parse the questions file!", status=400)

    # If the request method is GET, return a default response
    # return HttpResponse("Welcome to the exam page!", status=200)




# Ensures only logged-in users can submit an exam
def submit_exam(request):
    if request.method == 'POST':
        # Stop the camera capture thread
        global stop_event
        stop_event.set()  # Signal the thread to stop

        user = request.user  # Get the logged-in user directly

        
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

        # Iterate over questions and check the answers submitted
        for question in questions:
            question_id = question['id']
            user_answer = request.POST.get(f'answer_{question_id}')
            if user_answer == question['correct_answer']:
                correct_answers += 1

        
        # Save exam information to the database
        exam = Exam(
            student=user.student,  # Link to the Student profile via OneToOne relation
            total_questions=total_questions,
            correct_answers=correct_answers,
            timestamp=timezone.now()

        )
        exam.save()

        # Display a success message and redirect to the submission result page
        messages.success(request, 'You have successfully completed the exam!')
        return redirect('exam_submission_success')  # Redirect to the success page

    return HttpResponse("Invalid request method.", status=400)


## exam_submission_success Page:
def exam_submission_success(request):
    # Here we use the messages framework to display a success message
    return render(request, 'exam_submission_success.html')


# @login_required  # Ensures only logged-in users can access their result
def result(request):
    user = request.user  # Retrieve logged-in user

    
    # Get the latest exam result for the logged-in user
    try:
        exam = Exam.objects.filter(student=user.student).latest('timestamp')
    except Exam.DoesNotExist:
        return HttpResponse("No exam found for this user", status=404)

    # Calculate the percentage
    total_questions = exam.total_questions
    correct_answers = exam.correct_answers
    percentage = (correct_answers / total_questions) * 100

    # Context to pass to the result template
    context = {
        'user_name': user.username,  # Display logged-in user's username
        'score': correct_answers,  # Number of correct answers
        'total_questions': total_questions,  # Total number of questions in the exam
        'percentage': round(percentage, 2),  # Percentage with 2 decimal places
    }

    return render(request, 'result.html', context)