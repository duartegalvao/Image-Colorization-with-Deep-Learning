from datetime import timedelta

from django.http import HttpResponse, HttpResponseForbidden, HttpResponseBadRequest
from django.shortcuts import render
from ipware import get_client_ip

from .models import TuringTest


def index_view(request):
    return render(request, 'turing_tests/solo.html')


def ajax_submit_test(request):
    if request.is_ajax() and request.method == 'POST':
        test = TuringTest()
        try:
            time = int(request.POST['t'])
            if time < 1000:
                return HttpResponse(str(1))
            test.image = int(request.POST['i'])
            test.set = int(request.POST['si'])
            test.is_true = bool(request.POST['it'])
            test.is_correct = bool(request.POST['ic'])
            test.time = timedelta(milliseconds=time)
            test.ip_address, _ = get_client_ip(request)
            test.save()
            return HttpResponse(str(0))
        except KeyError or ValueError:
            return HttpResponseBadRequest()
    else:
        return HttpResponseForbidden()
