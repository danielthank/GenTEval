class Trace:
    def __init__(self, spans):
        self._spans = spans
        self._start_time = None
        self._is_error = None

    def __len__(self):
        return len(self._spans)

    @property
    def start_time(self):
        if self._start_time is not None:
            return self._start_time
        start_time = None
        for span in self._spans.values():
            if start_time is None or span["startTime"] < start_time:
                start_time = span["startTime"]
        self._start_time = start_time
        return self._start_time

    @property
    def is_error(self) -> bool:
        """Check if trace contains any ERROR spans (status.code=2).

        The status code is embedded in nodeName as the last segment:
        e.g., "frontend-proxy!@#POST!@#ingress!@#2" means status.code=2 (ERROR)

        Result is cached after first computation.
        """
        if self._is_error is not None:
            return self._is_error

        for span_data in self._spans.values():
            node_name = span_data.get("nodeName", "")
            if node_name.endswith("!@#2"):
                self._is_error = True
                return True

        self._is_error = False
        return False

    @property
    def spans(self):
        return self._spans
