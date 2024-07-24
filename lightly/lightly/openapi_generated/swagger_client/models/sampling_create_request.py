class SamplingCreateRequest(BaseModel):
    """
    SamplingCreateRequest
    """
    new_tag_name: constr(strict=True, min_length=3) = Field(..., alias="newTagName", description="The name of the tag")
    method: SamplingMethod = Field(...)
    config: SamplingConfig = Field(...)
    preselected_tag_id: Optional[constr(strict=True)] = Field(None, alias="preselectedTagId", description="MongoDB ObjectId")
    query_tag_id: Optional[constr(strict=True)] = Field(None, alias="queryTagId", description="MongoDB ObjectId")
    score_type: Optional[constr(strict=True, min_length=1)] = Field(None, alias="scoreType", description="Type of active learning score")
    row_count: Optional[Union[StrictFloat, StrictInt]] = Field(None, alias="rowCount", description="temporary rowCount until the API/DB is aware how many they are..")
    __properties = ["newTagName", "method", "config", "preselectedTagId", "queryTagId", "scoreType", "rowCount"]

    @validator('new_tag_name')
    def new_tag_name_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9 .:;=@_-]+$", value):
            raise ValueError(r"must validate the regular expression /^[a-zA-Z0-9][a-zA-Z0-9 .:;=@_-]+$/")
        return value

    @validator('preselected_tag_id')
    def preselected_tag_id_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if value is None:
            return value

        if not re.match(r"^[a-f0-9]{24}$", value):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return value

    @validator('query_tag_id')
    def query_tag_id_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if value is None:
            return value

        if not re.match(r"^[a-f0-9]{24}$", value):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return value

    @validator('score_type')
    def score_type_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if value is None:
            return value

        if not re.match(r"^[a-zA-Z0-9_+=,.@:\/-]*$", value):
            raise ValueError(r"must validate the regular expression /^[a-zA-Z0-9_+=,.@:\/-]*$/")
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
    def from_json(cls, json_str: str) -> SamplingCreateRequest:
        """Create an instance of SamplingCreateRequest from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of config
        if self.config:
            _dict['config' if by_alias else 'config'] = self.config.to_dict(by_alias=by_alias)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> SamplingCreateRequest:
        """Create an instance of SamplingCreateRequest from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return SamplingCreateRequest.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in SamplingCreateRequest) in the input: " + str(obj))

        _obj = SamplingCreateRequest.parse_obj({
            "new_tag_name": obj.get("newTagName"),
            "method": obj.get("method"),
            "config": SamplingConfig.from_dict(obj.get("config")) if obj.get("config") is not None else None,
            "preselected_tag_id": obj.get("preselectedTagId"),
            "query_tag_id": obj.get("queryTagId"),
            "score_type": obj.get("scoreType"),
            "row_count": obj.get("rowCount")
        })
        return _obj

