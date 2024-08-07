class JobStatusData(BaseModel):
    """
    JobStatusData
    """
    id: constr(strict=True) = Field(..., description="MongoDB ObjectId")
    dataset_id: Optional[constr(strict=True)] = Field(None, alias="datasetId", description="MongoDB ObjectId")
    status: JobState = Field(...)
    meta: Optional[JobStatusMeta] = None
    wait_time_till_next_poll: StrictInt = Field(..., alias="waitTimeTillNextPoll", description="The time in seconds the client should wait before doing the next poll.")
    created_at: conint(strict=True, ge=0) = Field(..., alias="createdAt", description="unix timestamp in milliseconds")
    last_modified_at: Optional[conint(strict=True, ge=0)] = Field(None, alias="lastModifiedAt", description="unix timestamp in milliseconds")
    finished_at: Optional[conint(strict=True, ge=0)] = Field(None, alias="finishedAt", description="unix timestamp in milliseconds")
    error: Optional[StrictStr] = None
    result: Optional[JobStatusDataResult] = None
    __properties = ["id", "datasetId", "status", "meta", "waitTimeTillNextPoll", "createdAt", "lastModifiedAt", "finishedAt", "error", "result"]

    @validator('id')
    def id_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^[a-f0-9]{24}$", value):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return value

    @validator('dataset_id')
    def dataset_id_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if value is None:
            return value

        if not re.match(r"^[a-f0-9]{24}$", value):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
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
    def from_json(cls, json_str: str) -> JobStatusData:
        """Create an instance of JobStatusData from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of meta
        if self.meta:
            _dict['meta' if by_alias else 'meta'] = self.meta.to_dict(by_alias=by_alias)
        # override the default output from pydantic by calling `to_dict()` of result
        if self.result:
            _dict['result' if by_alias else 'result'] = self.result.to_dict(by_alias=by_alias)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> JobStatusData:
        """Create an instance of JobStatusData from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return JobStatusData.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in JobStatusData) in the input: " + str(obj))

        _obj = JobStatusData.parse_obj({
            "id": obj.get("id"),
            "dataset_id": obj.get("datasetId"),
            "status": obj.get("status"),
            "meta": JobStatusMeta.from_dict(obj.get("meta")) if obj.get("meta") is not None else None,
            "wait_time_till_next_poll": obj.get("waitTimeTillNextPoll"),
            "created_at": obj.get("createdAt"),
            "last_modified_at": obj.get("lastModifiedAt"),
            "finished_at": obj.get("finishedAt"),
            "error": obj.get("error"),
            "result": JobStatusDataResult.from_dict(obj.get("result")) if obj.get("result") is not None else None
        })
        return _obj

