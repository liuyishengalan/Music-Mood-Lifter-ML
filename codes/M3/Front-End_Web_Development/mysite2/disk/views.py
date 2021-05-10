from django.shortcuts import render
from django import forms
from django.http import HttpResponse
from disk.models import User
# Create your views here.

class UserForm(forms.Form):
    headImg = forms.FileField()

def register(request):
    if request.method == "POST":
        uf = UserForm(request.POST,request.FILES)
        if uf.is_valid():
            #Request data form
            headImg = uf.cleaned_data['headImg']
            #Weite to DataSet
            user = User()
            user.headImg = headImg
            user.save()
            return HttpResponse('Uploaded Successfully')
    else:
        uf = UserForm()
    return render(request, 'register.html',{'uf':uf})