class ListType(Type):
    elem: Type
    size: int | None

    def __str__(self) -> str:
        size = f"{self.size}" if self.size else ""
        return f"{self.elem}[{size}]"

    def is_base_ty_like(self, base_ty: BaseTy) -> bool:
        return self.elem.is_base_ty_like(base_ty)

    def is_symint_like(self) -> bool:
        return self.elem.is_symint_like()

    def is_nullable(self) -> bool:
        return self.elem.is_nullable()

    def is_list_like(self) -> ListType | None:
        return self

class Location:
    file: str
    line: int

    def __str__(self) -> str:
        return f"{self.file}:{self.line}"

