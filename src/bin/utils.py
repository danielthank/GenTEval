def run_dirs():
    applications = [
        {
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
    ]

    for app in applications:
        for app_name, (service, fault) in app.items():
            for s in service:
                for f in fault:
                    for run in range(3):
                        yield app_name, s, f, run + 1
