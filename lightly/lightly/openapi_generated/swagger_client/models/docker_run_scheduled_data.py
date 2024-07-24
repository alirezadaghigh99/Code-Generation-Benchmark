class DockerRunScheduledData(BaseModel):
    """
    DockerRunScheduledData
    """
    id: constr(strict=True) = Field(..., description="MongoDB ObjectId")
    dataset_id: constr(strict=True) = Field(..., alias="datasetId", description="MongoDB ObjectId")
    user_id: Optional[StrictStr] = Field(None, alias="userId")
    config_id: constr(strict=True) = Field(..., alias="configId", description="MongoDB ObjectId")
    priority: DockerRunScheduledPriority = Field(...)
    runs_on: conlist(StrictStr) = Field(..., alias="runsOn", description="The labels used for specifying the run-worker-relationship")
    state: DockerRunScheduledState = Field(...)
    created_at: conint(strict=True, ge=0) = Field(..., alias="createdAt", description="unix timestamp in milliseconds")
    last_modified_at: conint(strict=True, ge=0) = Field(..., alias="lastModifiedAt", description="unix timestamp in milliseconds")
    owner: Optional[constr(strict=True)] = Field(None, description="MongoDB ObjectId")
    __properties = ["id", "datasetId", "userId", "configId", "priority", "runsOn", "state", "createdAt", "lastModifiedAt", "owner"]

    @validator('id')
    def id_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^[a-f0-9]{24}$", value):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return value

    @validator('dataset_id')
    def dataset_id_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^[a-f0-9]{24}$", value):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return value

    @validator('config_id')
    def config_id_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^[a-f0-9]{24}$", value):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return value

    @validator('owner')
    def owner_validate_regular_expression(cls, value):
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
    def from_json(cls, json_str: str) -> DockerRunScheduledData:
        """Create an instance of DockerRunScheduledData from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DockerRunScheduledData:
        """Create an instance of DockerRunScheduledData from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return DockerRunScheduledData.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in DockerRunScheduledData) in the input: " + str(obj))

        _obj = DockerRunScheduledData.parse_obj({
            "id": obj.get("id"),
            "dataset_id": obj.get("datasetId"),
            "user_id": obj.get("userId"),
            "config_id": obj.get("configId"),
            "priority": obj.get("priority"),
            "runs_on": obj.get("runsOn"),
            "state": obj.get("state"),
            "created_at": obj.get("createdAt"),
            "last_modified_at": obj.get("lastModifiedAt"),
            "owner": obj.get("owner")
        })
        return _obj

