class Trace:
    def __init__(self, spans):
        self._spans = spans
        self._start_time = None

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
    def spans(self):
        return self._spans
