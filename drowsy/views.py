from django.shortcuts import render, redirect
from drowsy.camera import VideoCamera

# Create your views here.
from django.http import StreamingHttpResponse


def index(request):
    return render(request, 'drowsy/index.html')


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video_feed():
    return StreamingHttpResponse(gen(VideoCamera()), content_type='multipart/x-mixed-replace; boundary=frame')
