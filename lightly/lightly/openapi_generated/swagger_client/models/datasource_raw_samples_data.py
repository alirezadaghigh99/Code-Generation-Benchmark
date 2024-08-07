class DatasourceRawSamplesData(BaseModel):
    """
    DatasourceRawSamplesData
    """
    has_more: StrictBool = Field(..., alias="hasMore", description="Set to `false` if end of list is reached. Otherwise `true`.")
    cursor: StrictStr = Field(..., description="A cursor that indicates the current position in the list. Must be passed to future requests to continue reading from the same list. ")
    data: conlist(DatasourceRawSamplesDataRow) = Field(..., description="Array containing the sample objects")
    __properties = ["hasMore", "cursor", "data"]

    class Config:
        """Pydantic configuration"""
        allow_population_by_field_name = True
        validate_assignment = True
        use_enum_values = True
        extra = Extra.forbid

    def to_str(self, by_alias: bool = False) -> str:
        """Returns the string representation of the model"""
        return pprint.pformat(self.dict(by_alias=by_alias))

    def to_json(self, by_alias: bool = False) -> str:
        """Returns the JSON representation of the model"""
        return json.dumps(self.to_dict(by_alias=by_alias))

    @classmethod
    def from_json(cls, json_str: str) -> DatasourceRawSamplesData:
        """Create an instance of DatasourceRawSamplesData from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in data (list)
        _items = []
        if self.data:
            for _item in self.data:
                if _item:
                    _items.append(_item.to_dict(by_alias=by_alias))
            _dict['data' if by_alias else 'data'] = _items
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DatasourceRawSamplesData:
        """Create an instance of DatasourceRawSamplesData from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return DatasourceRawSamplesData.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in DatasourceRawSamplesData) in the input: " + str(obj))

        _obj = DatasourceRawSamplesData.parse_obj({
            "has_more": obj.get("hasMore"),
            "cursor": obj.get("cursor"),
            "data": [DatasourceRawSamplesDataRow.from_dict(_item) for _item in obj.get("data")] if obj.get("data") is not None else None
        })
        return _obj

