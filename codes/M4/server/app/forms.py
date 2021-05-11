from django import forms

class UploadFileForm(forms.Form):
    File  = forms.FileField()