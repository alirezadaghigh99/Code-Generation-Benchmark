class DatasourceProcessedUntilTimestampResponse(BaseModel):
    """
    DatasourceProcessedUntilTimestampResponse
    """
    processed_until_timestamp: conint(strict=True, ge=0) = Field(..., alias="processedUntilTimestamp", description="unix timestamp in milliseconds")
    __properties = ["processedUntilTimestamp"]

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
    def from_json(cls, json_str: str) -> DatasourceProcessedUntilTimestampResponse:
        """Create an instance of DatasourceProcessedUntilTimestampResponse from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DatasourceProcessedUntilTimestampResponse:
        """Create an instance of DatasourceProcessedUntilTimestampResponse from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return DatasourceProcessedUntilTimestampResponse.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in DatasourceProcessedUntilTimestampResponse) in the input: " + str(obj))

        _obj = DatasourceProcessedUntilTimestampResponse.parse_obj({
            "processed_until_timestamp": obj.get("processedUntilTimestamp")
        })
        return _obj

