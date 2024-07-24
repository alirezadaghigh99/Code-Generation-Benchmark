class TagBitMaskResponse(BaseModel):
    """
    TagBitMaskResponse
    """
    bit_mask_data: constr(strict=True) = Field(..., alias="bitMaskData", description="BitMask as a base16 (hex) string")
    __properties = ["bitMaskData"]

    @validator('bit_mask_data')
    def bit_mask_data_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^0x[a-f0-9]+$", value):
            raise ValueError(r"must validate the regular expression /^0x[a-f0-9]+$/")
        return value

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
    def from_json(cls, json_str: str) -> TagBitMaskResponse:
        """Create an instance of TagBitMaskResponse from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> TagBitMaskResponse:
        """Create an instance of TagBitMaskResponse from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return TagBitMaskResponse.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in TagBitMaskResponse) in the input: " + str(obj))

        _obj = TagBitMaskResponse.parse_obj({
            "bit_mask_data": obj.get("bitMaskData")
        })
        return _obj

