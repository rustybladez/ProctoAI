from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import datetime as now

# Student model
class Student(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='student', null=True, blank=True)  # Temporarily allow null for User
    name = models.CharField(max_length=255)
    address = models.TextField()
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=255, null=True)  # Consider using User model's password
    photo = models.ImageField(upload_to='student_photos/')  # Stores photo as an image file
    face_encoding = models.JSONField(null=True, blank=True)  # Store face encoding as binary data (numpy array converted to binary)
    timestamp = models.DateTimeField(default=timezone.now)
    feedback = models.TextField(null=True, blank=True, max_length=1000)  # Optional feedback from the student

    def __str__(self):
        return self.name


# Exam model
class Exam(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='exams', null=True, blank=True)
    exam_name = models.CharField(max_length=255, default='Default Exam Name')  # Provide a default name
    total_questions = models.IntegerField(null=True, blank=True)
    correct_answers = models.IntegerField(null=True, blank=True)
    timestamp = models.DateTimeField(default=now)
    status = models.CharField(
        max_length=50,
        choices=[('ongoing', 'Ongoing'), ('completed', 'Completed'), ('cancelled', 'Cancelled')],
        default='ongoing'
    )

    def calculate_percentage(self):
        if self.total_questions and self.total_questions > 0:
            return round((self.correct_answers / self.total_questions) * 100, 2)
        return 0.0

    def __str__(self):
        return f"{self.exam_name} - {self.student.name}"


# CheatingEvent model
class CheatingEvent(models.Model):
    student = models.ForeignKey(User, on_delete=models.CASCADE,null=True, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    detected_faces = models.IntegerField(null=True, blank=True)
    face_details = models.JSONField(null=True, blank=True)
    head_pose = models.JSONField(null=True, blank=True)
    detected_objects = models.JSONField(null=True, blank=True)
    audio_detected = models.BooleanField(null=True, blank=True)
    cheating_flag = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.student.username} - {self.timestamp}"
