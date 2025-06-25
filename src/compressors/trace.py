import copy


class Trace:
    def __init__(self, spans):
        self._spans = spans
        self._start_time = None
        self._edges = None
        self._preorder = None
        self._span_id_to_unique_name = None
        self._gap_from_parent = None

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
    def edges(self):
        if self._edges is not None:
            return self._edges
        edges = []
        for span in self._spans.values():
            if span["parentSpanId"] is not None and span["parentSpanId"] in self._spans:
                parent_span = self._spans[span["parentSpanId"]]
                edges.append((parent_span["nodeName"], span["nodeName"]))
        self._edges = sorted(edges)
        return self._edges

    @property
    def graph(self):
        return str(self.edges)

    @property
    def spans(self):
        return self._spans

    def unique_name(self, span_id):
        if self._preorder is None:
            raise ValueError(
                "Unique name cannot be generated before chains are generated."
            )

        if self._span_id_to_unique_name is not None:
            return self._span_id_to_unique_name[span_id]

        self._span_id_to_unique_name = {}
        node_name_to_span_ids = {}
        for id in self._preorder:
            node_name = self._spans[id]["nodeName"]
            if node_name not in node_name_to_span_ids:
                node_name_to_span_ids[node_name] = []
            node_name_to_span_ids[node_name].append(id)

        for node_name, ids in node_name_to_span_ids.items():
            for i, id in enumerate(ids):
                if len(ids) == 1:
                    self._span_id_to_unique_name[id] = node_name
                else:
                    self._span_id_to_unique_name[id] = f"{node_name}*{i + 1}"

        return self._span_id_to_unique_name[span_id]

    def gap_from_parent(self, span_id):
        if self._gap_from_parent is None:
            self._gap_from_parent = {}
            for id in self._spans:
                if self._spans[id]["parentSpanId"] is not None:
                    if self._spans[id]["parentSpanId"] not in self._spans:
                        gap = 0
                    else:
                        parent_span = self._spans[self._spans[id]["parentSpanId"]]
                        gap = self._spans[id]["startTime"] - parent_span["startTime"]
                        if gap < 0:
                            gap = 0  # gap is modeled as log normal distribution so it cannot be negative
                    self._gap_from_parent[id] = gap
                else:
                    self._gap_from_parent[id] = 0
        return self._gap_from_parent[span_id]

    def duration(self, span_id):
        span = self._spans[span_id]
        return span["duration"]

    def chains(self, chain_length):
        graph = {}
        for span_id in self._spans:
            graph[span_id] = []

        for child_id, child_span in self._spans.items():
            parent_id = child_span["parentSpanId"]
            if parent_id is not None and parent_id in self._spans:
                graph[parent_id].append(child_id)

        for span_id in graph:
            graph[span_id].sort(key=lambda child_id: self._spans[child_id]["nodeName"])

        all_root_nodes = sorted(
            [
                span_id
                for span_id in self._spans.keys()
                if self._spans[span_id]["parentSpanId"] is None
            ],
            key=lambda span_id: self._spans[span_id]["nodeName"],
        )

        all = []
        visited = set()
        self._preorder = []

        def dfs(node, current):
            if node in visited:
                return
            self._preorder.append(node)
            visited.add(node)
            current["chain"].append(node)

            if len(current["chain"]) == chain_length:
                all.append(copy.deepcopy(current))
                current["chain"].clear()
                current["is_root"] = False

            is_leaf = True
            for child in graph.get(node, []):
                if child not in visited:
                    if len(current["chain"]) == 0:
                        current["chain"].append(node)
                    dfs(child, current)
                is_leaf = False
            if is_leaf and len(current["chain"]) > 0:
                all.append(copy.deepcopy(current))
                current["chain"].clear()
                current["is_root"] = False

        for node in all_root_nodes:
            if node not in visited:
                new_chain = {"chain": [], "is_root": True}
                dfs(node, new_chain)

        return all
