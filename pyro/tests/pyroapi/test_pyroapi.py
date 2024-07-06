def backend(request):
    with pyro_backend(request.param):
        yield

