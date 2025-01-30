from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone

# Student model
class Student(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='student', null=True, blank=True)
    name = models.CharField(max_length=255)
    address = models.TextField(null=True, blank=True)
    email = models.EmailField(unique=True)  # Use Django's built-in email validation
    photo = models.ImageField(upload_to='student_photos/')
    face_encoding = models.JSONField(null=True, blank=True)  # Store facial encoding as JSON
    timestamp = models.DateTimeField(default=timezone.now)
    feedback = models.TextField(null=True, blank=True, max_length=1000)

    def __str__(self):
        return self.name

# Exam model
class Exam(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE, related_name='exams', null=True, blank=True)
    exam_name = models.CharField(max_length=255, default='Default Exam Name')
    total_questions = models.IntegerField(null=True, blank=True)
    correct_answers = models.IntegerField(null=True, blank=True)
    timestamp = models.DateTimeField(default=timezone.now)
    status = models.CharField(
        max_length=50,
        choices=[('ongoing', 'Ongoing'), ('completed', 'Completed'), ('cancelled', 'Cancelled')],
        default='ongoing'
    )
    percentage_score = models.FloatField(null=True, blank=True)  # Cache percentage score

    def calculate_percentage(self):
        if self.total_questions > 0:
            self.percentage_score = round((self.correct_answers / self.total_questions) * 100, 2)
        else:
            self.percentage_score = 0.0
        self.save()

    def __str__(self):
        return f"{self.exam_name} - {self.student.name}"

from django.db import models

class CheatingEvent(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE,blank=True, null=True)
    cheating_flag = models.BooleanField(default=False)
    event_type = models.CharField(max_length=50, blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    tab_switch_count = models.IntegerField(default=0)
    detected_objects = models.JSONField(default=list)  # Store detected objects as JSON

class CheatingImage(models.Model):
    event = models.ForeignKey(CheatingEvent, on_delete=models.CASCADE, related_name='cheating_images')
    image = models.ImageField(upload_to='cheating_images/')
    timestamp = models.DateTimeField(auto_now_add=True)

class CheatingAudio(models.Model):
    event = models.ForeignKey(CheatingEvent, on_delete=models.CASCADE, related_name='cheating_audios')
    audio = models.FileField(upload_to='cheating_audios/',blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)