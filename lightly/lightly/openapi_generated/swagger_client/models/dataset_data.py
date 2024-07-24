class DatasetData(BaseModel):
    """
    DatasetData
    """
    id: constr(strict=True) = Field(..., description="MongoDB ObjectId")
    name: constr(strict=True, min_length=3) = Field(...)
    user_id: StrictStr = Field(..., alias="userId", description="The owner of the dataset")
    access_type: Optional[SharedAccessType] = Field(None, alias="accessType")
    type: DatasetType = Field(...)
    img_type: Optional[ImageType] = Field(None, alias="imgType")
    n_samples: StrictInt = Field(..., alias="nSamples")
    size_in_bytes: StrictInt = Field(..., alias="sizeInBytes")
    meta_data_configuration_id: Optional[constr(strict=True)] = Field(None, alias="metaDataConfigurationId", description="MongoDB ObjectId")
    datasources: Optional[conlist(constr(strict=True))] = None
    created_at: conint(strict=True, ge=0) = Field(..., alias="createdAt", description="unix timestamp in milliseconds")
    last_modified_at: conint(strict=True, ge=0) = Field(..., alias="lastModifiedAt", description="unix timestamp in milliseconds")
    datasource_processed_until_timestamp: Optional[conint(strict=True, ge=0)] = Field(None, alias="datasourceProcessedUntilTimestamp", description="unix timestamp in seconds")
    access_role: Optional[constr(strict=True)] = Field(None, alias="accessRole", description="AccessRole bitmask of the one accessing the dataset")
    parent_dataset_id: Optional[constr(strict=True)] = Field(None, alias="parentDatasetId", description="MongoDB ObjectId")
    original_dataset_id: Optional[constr(strict=True)] = Field(None, alias="originalDatasetId", description="MongoDB ObjectId")
    __properties = ["id", "name", "userId", "accessType", "type", "imgType", "nSamples", "sizeInBytes", "metaDataConfigurationId", "datasources", "createdAt", "lastModifiedAt", "datasourceProcessedUntilTimestamp", "accessRole", "parentDatasetId", "originalDatasetId"]

    @validator('id')
    def id_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^[a-f0-9]{24}$", value):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return value

    @validator('name')
    def name_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9 _-]+$", value):
            raise ValueError(r"must validate the regular expression /^[a-zA-Z0-9][a-zA-Z0-9 _-]+$/")
        return value

    @validator('meta_data_configuration_id')
    def meta_data_configuration_id_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if value is None:
            return value

        if not re.match(r"^[a-f0-9]{24}$", value):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return value

    @validator('access_role')
    def access_role_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if value is None:
            return value

        if not re.match(r"^0b[01]{6}$", value):
            raise ValueError(r"must validate the regular expression /^0b[01]{6}$/")
        return value

    @validator('parent_dataset_id')
    def parent_dataset_id_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if value is None:
            return value

        if not re.match(r"^[a-f0-9]{24}$", value):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return value

    @validator('original_dataset_id')
    def original_dataset_id_validate_regular_expression(cls, value):
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
    def from_json(cls, json_str: str) -> DatasetData:
        """Create an instance of DatasetData from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> DatasetData:
        """Create an instance of DatasetData from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return DatasetData.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in DatasetData) in the input: " + str(obj))

        _obj = DatasetData.parse_obj({
            "id": obj.get("id"),
            "name": obj.get("name"),
            "user_id": obj.get("userId"),
            "access_type": obj.get("accessType"),
            "type": obj.get("type"),
            "img_type": obj.get("imgType"),
            "n_samples": obj.get("nSamples"),
            "size_in_bytes": obj.get("sizeInBytes"),
            "meta_data_configuration_id": obj.get("metaDataConfigurationId"),
            "datasources": obj.get("datasources"),
            "created_at": obj.get("createdAt"),
            "last_modified_at": obj.get("lastModifiedAt"),
            "datasource_processed_until_timestamp": obj.get("datasourceProcessedUntilTimestamp"),
            "access_role": obj.get("accessRole"),
            "parent_dataset_id": obj.get("parentDatasetId"),
            "original_dataset_id": obj.get("originalDatasetId")
        })
        return _obj

