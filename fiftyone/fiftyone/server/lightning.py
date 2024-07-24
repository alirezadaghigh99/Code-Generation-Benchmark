class LightningPathInput:
    path: str

    exclude: t.Optional[t.List[str]] = gql.field(
        description="exclude these values from results", default=None
    )
    first: t.Optional[int] = foc.LIST_LIMIT
    search: t.Optional[str] = None

