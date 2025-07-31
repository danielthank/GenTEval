def run_dirs(applications=None, services=None, faults=None, runs=None):
    """
    Generate combinations of app, service, fault, and run.

    Args:
        applications: List of application names or None for all (default: all)
        services: List of service names or None for all (default: all)
        faults: List of fault types or None for all (default: all)
        runs: List of run numbers or None for all (default: [1, 2, 3])

    Yields:
        Tuple of (app_name, service, fault, run)
    """
    all_applications = {
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

    # Convert single values to lists
    if applications is None:
        target_apps = list(all_applications.keys())
    elif isinstance(applications, str):
        target_apps = [applications]
    else:
        target_apps = applications

    if runs is None:
        target_runs = [1, 2, 3]
    elif isinstance(runs, int):
        target_runs = [runs]
    else:
        target_runs = runs

    # Process each requested application
    for app_name in target_apps:
        if app_name not in all_applications:
            continue

        available_services, available_faults = all_applications[app_name]
        # Filter services
        if services is None:
            target_services = available_services
        elif isinstance(services, str):
            target_services = [services] if services in available_services else []
        else:
            target_services = [s for s in services if s in available_services]

        # Filter faults
        if faults is None:
            target_faults = available_faults
        elif isinstance(faults, str):
            target_faults = [faults] if faults in available_faults else []
        else:
            target_faults = [f for f in faults if f in available_faults]

        for service in target_services:
            for fault in target_faults:
                for run in target_runs:
                    yield app_name, service, fault, run
