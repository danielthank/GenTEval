import pathlib


def run_dirs(applications=None, services=None, faults=None, runs=None):
    """
    Generate combinations of app, service, fault, and run.

    Args:
        applications: List of application names or None for all (default: all)
        services: List of service names or None for all (default: all)
        faults: List of fault types or None for all (default: all)
        runs: List of run numbers or None for all (default: all)

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
            [1, 2, 3],
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
            [1, 2, 3],
        ),
        "the-agent-company-transformed": (
            [
                "20241217_OpenHands-0.14.2-gemini-1.5-pro",
                "20241217_OpenHands-0.14.2-gemini-2.0-flash",
                "20241217_OpenHands-0.14.2-gpt-4o-2024-08-06",
                "20241217_OpenHands-0.14.2-llama-3.1-405b",
                "20241217_OpenHands-0.14.2-llama-3.1-70b",
                "20241217_OpenHands-0.14.2-llama-3.3-70b",
                "20241217_OpenHands-0.14.2-nova-pro-v1:0",
                "20241217_OpenHands-0.14.2-qwen2-72b",
                "20241217_OpenHands-0.14.2-qwen2.5-72b",
                "20241217_OpenHands-0.14.2-sonnet-20241022",
                "20250510_OpenHands-0.28.1-gemini-2.5-pro",
                "20250510_OpenHands-0.28.1-sonnet-20250219",
                "20250614_OpenHands-Versa-claude-3.7-sonnet",
                "20250614_OpenHands-Versa-claude-sonnet-4",
            ],
            [None],
            [1],
        ),
    }

    # Convert single values to lists
    if applications is None:
        target_apps = list(all_applications.keys())
    elif isinstance(applications, str):
        target_apps = [applications]
    else:
        target_apps = applications

    # Process each requested application
    for app_name in target_apps:
        if app_name not in all_applications:
            continue

        available_services, available_faults, available_runs = all_applications[
            app_name
        ]
        # Filter services
        if services is None:
            target_services = available_services
        elif isinstance(services, str):
            target_services = [services] if services in available_services else []
        else:
            target_services = [s for s in services if s in available_services]

        # Filter runs
        if runs is None:
            target_runs = available_runs
        elif isinstance(runs, int):
            target_runs = [runs] if runs in available_runs else []
        else:
            target_runs = [r for r in runs if r in available_runs]

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


def get_dir_with_root(
    root_dir: pathlib.Path, app_name: str, service: str, fault: str | None, run: int
) -> pathlib.Path:
    if fault is not None:
        return root_dir / app_name / f"{service}_{fault}" / str(run)
    return root_dir / app_name / service / str(run)
