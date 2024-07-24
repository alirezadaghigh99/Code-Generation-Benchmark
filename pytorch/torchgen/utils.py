def from_namespaced_entity(
        namespaced_entity: str, max_level: int = 2
    ) -> NamespaceHelper:
        """
        Generate helper from nested namespaces as long as class/function name. E.g.: "torch::lazy::add"
        """
        names = namespaced_entity.split("::")
        entity_name = names[-1]
        namespace_str = "::".join(names[:-1])
        return NamespaceHelper(
            namespace_str=namespace_str, entity_name=entity_name, max_level=max_level
        )

class NamespaceHelper:
    """A helper for constructing the namespace open and close strings for a nested set of namespaces.

    e.g. for namespace_str torch::lazy,

    prologue:
    namespace torch {
    namespace lazy {

    epilogue:
    } // namespace lazy
    } // namespace torch
    """

    def __init__(
        self, namespace_str: str, entity_name: str = "", max_level: int = 2
    ) -> None:
        # cpp_namespace can be a colon joined string such as torch::lazy
        cpp_namespaces = namespace_str.split("::")
        assert (
            len(cpp_namespaces) <= max_level
        ), f"Codegen doesn't support more than {max_level} level(s) of custom namespace. Got {namespace_str}."
        self.cpp_namespace_ = namespace_str
        self.prologue_ = "\n".join([f"namespace {n} {{" for n in cpp_namespaces])
        self.epilogue_ = "\n".join(
            [f"}} // namespace {n}" for n in reversed(cpp_namespaces)]
        )
        self.namespaces_ = cpp_namespaces
        self.entity_name_ = entity_name

    @staticmethod
    def from_namespaced_entity(
        namespaced_entity: str, max_level: int = 2
    ) -> NamespaceHelper:
        """
        Generate helper from nested namespaces as long as class/function name. E.g.: "torch::lazy::add"
        """
        names = namespaced_entity.split("::")
        entity_name = names[-1]
        namespace_str = "::".join(names[:-1])
        return NamespaceHelper(
            namespace_str=namespace_str, entity_name=entity_name, max_level=max_level
        )

    @property
    def prologue(self) -> str:
        return self.prologue_

    @property
    def epilogue(self) -> str:
        return self.epilogue_

    @property
    def entity_name(self) -> str:
        return self.entity_name_

    # Only allow certain level of namespaces
    def get_cpp_namespace(self, default: str = "") -> str:
        """
        Return the namespace string from joining all the namespaces by "::" (hence no leading "::").
        Return default if namespace string is empty.
        """
        return self.cpp_namespace_ if self.cpp_namespace_ else default

