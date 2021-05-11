from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.shortcuts import render
from .forms import UploadFileForm
from .models import ImageModel
from django.conf import settings
from app.classifier import classifier
from PIL import Image, ImageOps,ImageFilter
import random 
import os.path
from app.music_model import MusicMoodClassifier

def index(request):
    context ={}
    # if this is a POST request we need to process the form data
    if request.method == 'POST':
        # create a form instance and populate it with data from the request:
        form = UploadFileForm(request.POST, request.FILES)

        if form.is_valid():
            print(request.FILES['File'])
            name = handle_uploaded_file(request.FILES['File'])
            outputfilename = "static/images/"+ name
            prediction = process_upload(name)
            emotion =prediction[0]
            if prediction[1] is not -1:
                rf = MusicMoodClassifier()
                if(prediction[1] == 0):
                    label = 0 # if angry detected, suggest calm music.
                elif(prediction[1] == 1):
                    label = 0 # if disgusted detected, suggest calm music.
                elif(prediction[1] == 2):
                    label = 0 # if fearful detected, suggest calm music.
                elif(prediction[1] == 3):
                    label = 2 # if happy detected, suggest happy music.
                elif(prediction[1] == 4):
                    label = 2 # if neutral detected, suggest happy music.
                elif(prediction[1] == 5):
                    label = 3 # if sad detected, suggest sad music.
                elif(prediction[1] == 6):
                    label = 1 # if surprised detected, suggest energetic music.
                prob= rf.getTypicalTracks(label)
            else:
                prob= 'N/A'            
            print("Root Path: " , settings.ROOT_PATH)
            return render(request, 'result.html',{'file': outputfilename, 'emotion': emotion, 'Music': prob[0], 'URL': prob[1], 'Album': prob[2]})

    # if a GET (or any other method) we'll create a blank form
    else:
        form = UploadFileForm()
    return render(request, 'index.html', {'form': form})

def handle_uploaded_file(f):
    context={}
    uploadfilename= os.path.abspath(os.path.dirname(__file__))+ '/static/images/'+f.name
    with open(uploadfilename, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    return f.name

def process_upload(f):
    print("In process upload function")
    c = classifier()
    return c.make_prediction(f)