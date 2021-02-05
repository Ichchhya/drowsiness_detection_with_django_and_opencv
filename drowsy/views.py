from django.shortcuts import render, redirect,HttpResponse
from drowsy.camera import VideoCamera
from drow.drowsinessdetection import runmethod
# Create your views here.
from django.http.response import StreamingHttpResponse


def index(request):
    return render(request, 'drowsy/index.html')

def startsystem(request):
    runmethod()
    return HttpResponse("system is starting")

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video_feed(request):
    return StreamingHttpResponse(gen(VideoCamera()),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
