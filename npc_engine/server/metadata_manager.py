"""Module that implements lifetime and discoverability of the services."""
from typing import Dict
import os

from npc_engine import services
from collections import namedtuple
import ntpath
import yaml

from npc_engine.server.utils import build_ipc_uri


ServiceDescriptor = namedtuple(
    "ServiceDescriptor",
    ["id", "type", "path", "uri", "api_name", "api_methods", "dependencies"],
)


class MetadataManager:
    """Class that manages service name resolution and metadata."""

    def __init__(self, path: str, port: str):
        """Create model manager and load models from the given path."""
        self.services = self._scan_path(path)
        self.models_path = path
        self.port = port

    def resolve_service(self, id_or_type, method=None):
        """Resolve service id or type to service id."""
        service_id = None
        if id_or_type == "control":
            service_id = "control"
        else:
            if id_or_type in self.services:
                service_id = id_or_type
            if service_id is None:
                for service_key, service in self.services.items():
                    if service.type == id_or_type or service.api_name == id_or_type:
                        service_id = service_key
                        break
            if service_id is None and method is not None:
                service_id = self.resolve_by_method(method)
            elif service_id is None:
                raise ValueError(f"Service {id_or_type} not found")

        return service_id

    def get_services_metadata(self):
        """List the models in the folder."""
        return [self.get_metadata(service) for service in self.services]

    def resolve_by_method(self, method_name):
        """Resolve service id by method name."""
        for service_id, service in self.services.items():
            if method_name in service.api_methods:
                return service_id
        raise ValueError(f"Service with method {method_name} not found")

    def check_dependency_cycles(self):
        """Check if there are any dependency cycles."""
        # build graph
        graph = {}
        for service_id, service in self.services.items():
            graph[service_id] = [
                self.resolve_service(dep) for dep in service.dependencies
            ]
        # Tarjan's algorithm
        visited = {}
        llvs = {}
        stack = []
        sccs = []
        self.idx = 0
        for node in graph:
            if node not in visited:
                self.__scc(graph, node, llvs, visited, stack, sccs)
        cycles = [scc for scc in sccs if len(scc) > 1]
        if len(cycles) > 0:
            to_str = "\n".join([" -> ".join(cycle + [cycle[0]]) for cycle in cycles])
            raise ValueError(f"There are dependency cycles: {to_str} ")
        del self.idx

    def __scc(self, graph, node, llvs, visited, stack, sccs):
        visited[node] = self.idx
        llvs[node] = self.idx
        self.idx += 1
        stack.append(node)
        for v in graph[node]:
            if v not in visited:
                self.__scc(graph, v, llvs, visited, stack, sccs)
                llvs[node] = min(llvs[node], llvs[v])
            elif v in stack:
                llvs[node] = min(llvs[node], visited.get(v, -1))
        if llvs[node] == visited[node]:
            sccs.append([stack.pop()])
            while sccs[-1][-1] != node:
                sccs[-1].append(stack.pop())

    def _scan_path(self, path: str) -> Dict[str, ServiceDescriptor]:
        """Scan services defined in the given path."""
        norm_path = ntpath.normpath(path).replace("\\", os.path.sep)
        paths = [
            f.path
            for f in os.scandir(norm_path)
            if f.is_dir() and os.path.exists(os.path.join(f, "config.yml"))
        ]
        svcs = {}
        for path in paths:
            with open(os.path.join(path, "config.yml")) as f:
                config_dict = yaml.safe_load(f)
                uri = build_ipc_uri(os.path.basename(path))
                cls = getattr(
                    services,
                    config_dict.get("model_type", config_dict.get("type", None)),
                )
                svcs[os.path.basename(path)] = ServiceDescriptor(
                    id=os.path.basename(path),
                    type=config_dict.get("model_type", config_dict.get("type", None)),
                    path=path,
                    uri=uri,
                    api_name=cls.get_api_name(),
                    api_methods=cls.API_METHODS,
                    dependencies=[],
                )

        return svcs

    def get_metadata(self, service_id: str) -> Dict[str, str]:
        """Print the model from the path."""
        service_id = self.resolve_service(service_id)
        config_path = os.path.join(self.services[service_id].path, "config.yml")
        readme_path = os.path.join(self.services[service_id].path, "README.md")
        with open(config_path) as f:
            config_dict = yaml.load(f, Loader=yaml.Loader)
        try:
            with open(readme_path) as f:
                readme = f.read().split("---")[-1]
        except FileNotFoundError:
            readme = ""
        cls = getattr(
            services,
            config_dict.get("model_type", config_dict.get("type", None)),
        )
        return {
            "id": self.services[service_id].id,
            "service": self.services[service_id].type,
            "api_name": self.services[service_id].api_name,
            "path": self.services[service_id].path,
            "service_short_description": cls.models[
                self.services[service_id].type
            ].__doc__.split("\n\n")[0],
            "service_description": cls.models[self.services[service_id].type].__doc__,
            "readme": readme,
        }
