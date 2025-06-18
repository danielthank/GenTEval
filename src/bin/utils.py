def run_dirs(application=None):
    applications = {
        "RE2-OB": (
            [
                "checkoutservice",
                "currencyservice",
                "emailservice",
                "productcatalogservice",
            ],
            ["cpu", "delay", "disk", "loss", "mem", "socket"],
        ),
        "RE2-TT": (
            [
                "ts-auth-service",
                "ts-order-service",
                "ts-route-service",
                "ts-train-service",
                "ts-travel-service",
            ],
            ["cpu", "delay", "disk", "loss", "mem", "socket"],
        ),
    }

    run_apps = applications
    if application:
        if application in applications:
            run_apps = {application: applications[application]}
        else:
            return

    for app_name, (service, fault) in run_apps.items():
        for s in service:
            for f in fault:
                for run in range(3):
                    yield app_name, s, f, run + 1
