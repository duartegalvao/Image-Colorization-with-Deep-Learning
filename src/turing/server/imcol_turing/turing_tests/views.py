import json
from datetime import timedelta

from django.shortcuts import render
from ipware import get_client_ip
from ratelimit.decorators import ratelimit
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .stats import SetStatistics
from .models import TuringTest


def index_view(request):
    return render(request, 'turing_tests/solo.html')


def stats_view(request):
    return render(request, 'turing_tests/stats.html', context={
        "unet": SetStatistics(1),
        "gan": SetStatistics(2),
        "vacgan": SetStatistics(3),
        "total": SetStatistics(),
    })


@ratelimit(key='ip', rate='60/m')
@api_view(['POST'])
def ajax_submit_test(request):
    was_limited = getattr(request, 'limited', False)
    if was_limited:
        return Response(status=status.HTTP_400_BAD_REQUEST)

    if request.is_ajax():
        test = TuringTest()
        try:
            time = request.data['t']
            if time < 1000:
                return Response(str(1), status=status.HTTP_200_OK)
            test.image = request.data['i']
            test.set = request.data['si']
            test.is_true = request.data['it']
            test.is_correct = request.data['ic']
            test.time = timedelta(milliseconds=time)
            test.ip_address, _ = get_client_ip(request)
            test.save()
            return Response(str(0), status=status.HTTP_200_OK)
        except KeyError or ValueError:
            return Response(status=status.HTTP_400_BAD_REQUEST)
    else:
        return Response(status=status.HTTP_403_FORBIDDEN)
