class JobStatusDataResult(BaseModel):
    """
    JobStatusDataResult
    """
    type: JobResultType = Field(...)
    data: Optional[Any] = Field(None, description="Depending on the job type, this can be anything")
    __properties = ["type", "data"]

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
    def from_json(cls, json_str: str) -> JobStatusDataResult:
        """Create an instance of JobStatusDataResult from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        # set to None if data (nullable) is None
        # and __fields_set__ contains the field
        if self.data is None and "data" in self.__fields_set__:
            _dict['data' if by_alias else 'data'] = None

        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> JobStatusDataResult:
        """Create an instance of JobStatusDataResult from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return JobStatusDataResult.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in JobStatusDataResult) in the input: " + str(obj))

        _obj = JobStatusDataResult.parse_obj({
            "type": obj.get("type"),
            "data": obj.get("data")
        })
        return _obj

