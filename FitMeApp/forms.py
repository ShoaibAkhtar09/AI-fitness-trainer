from .models import *
from django.forms import ModelForm


class TaskForm(ModelForm):
    class Meta:
        model = Task
        fields = ["note", "task_to_give"]
        
        
from django.contrib.auth.models import User
from django import forms

from django.contrib.auth.forms import UserCreationForm
 
class SignUpForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('username', 'password1', 'password2', )
        
        
        
class VideoForm(forms.ModelForm):
    class Meta:
        model= Videos
        fields= ["video"]

        


   
