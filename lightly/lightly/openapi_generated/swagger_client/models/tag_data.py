class TagData(BaseModel):
    """
    TagData
    """
    id: constr(strict=True) = Field(..., description="MongoDB ObjectId")
    dataset_id: constr(strict=True) = Field(..., alias="datasetId", description="MongoDB ObjectId")
    prev_tag_id: Optional[constr(strict=True)] = Field(..., alias="prevTagId", description="MongoObjectID or null.  Generally: The prevTagId is this tag's parent, i.e. it is a superset of this tag. Sampler: The prevTagId is the initial-tag if there was no preselectedTagId, otherwise, it's the preselectedTagId. ")
    query_tag_id: Optional[constr(strict=True)] = Field(None, alias="queryTagId", description="MongoDB ObjectId")
    preselected_tag_id: Optional[constr(strict=True)] = Field(None, alias="preselectedTagId", description="MongoDB ObjectId")
    name: constr(strict=True, min_length=3) = Field(..., description="The name of the tag")
    bit_mask_data: constr(strict=True) = Field(..., alias="bitMaskData", description="BitMask as a base16 (hex) string")
    tot_size: StrictInt = Field(..., alias="totSize")
    created_at: conint(strict=True, ge=0) = Field(..., alias="createdAt", description="unix timestamp in milliseconds")
    last_modified_at: Optional[conint(strict=True, ge=0)] = Field(None, alias="lastModifiedAt", description="unix timestamp in milliseconds")
    changes: Optional[conlist(TagChangeEntry)] = None
    run_id: Optional[constr(strict=True)] = Field(None, alias="runId", description="MongoDB ObjectId")
    __properties = ["id", "datasetId", "prevTagId", "queryTagId", "preselectedTagId", "name", "bitMaskData", "totSize", "createdAt", "lastModifiedAt", "changes", "runId"]

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

    @validator('prev_tag_id')
    def prev_tag_id_validate_regular_expression(cls, value):
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

    @validator('preselected_tag_id')
    def preselected_tag_id_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if value is None:
            return value

        if not re.match(r"^[a-f0-9]{24}$", value):
            raise ValueError(r"must validate the regular expression /^[a-f0-9]{24}$/")
        return value

    @validator('name')
    def name_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9 .:;=@_-]+$", value):
            raise ValueError(r"must validate the regular expression /^[a-zA-Z0-9][a-zA-Z0-9 .:;=@_-]+$/")
        return value

    @validator('bit_mask_data')
    def bit_mask_data_validate_regular_expression(cls, value):
        """Validates the regular expression"""
        if not re.match(r"^0x[a-f0-9]+$", value):
            raise ValueError(r"must validate the regular expression /^0x[a-f0-9]+$/")
        return value

    @validator('run_id')
    def run_id_validate_regular_expression(cls, value):
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
    def from_json(cls, json_str: str) -> TagData:
        """Create an instance of TagData from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of each item in changes (list)
        _items = []
        if self.changes:
            for _item in self.changes:
                if _item:
                    _items.append(_item.to_dict(by_alias=by_alias))
            _dict['changes' if by_alias else 'changes'] = _items
        # set to None if prev_tag_id (nullable) is None
        # and __fields_set__ contains the field
        if self.prev_tag_id is None and "prev_tag_id" in self.__fields_set__:
            _dict['prevTagId' if by_alias else 'prev_tag_id'] = None

        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> TagData:
        """Create an instance of TagData from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return TagData.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in TagData) in the input: " + str(obj))

        _obj = TagData.parse_obj({
            "id": obj.get("id"),
            "dataset_id": obj.get("datasetId"),
            "prev_tag_id": obj.get("prevTagId"),
            "query_tag_id": obj.get("queryTagId"),
            "preselected_tag_id": obj.get("preselectedTagId"),
            "name": obj.get("name"),
            "bit_mask_data": obj.get("bitMaskData"),
            "tot_size": obj.get("totSize"),
            "created_at": obj.get("createdAt"),
            "last_modified_at": obj.get("lastModifiedAt"),
            "changes": [TagChangeEntry.from_dict(_item) for _item in obj.get("changes")] if obj.get("changes") is not None else None,
            "run_id": obj.get("runId")
        })
        return _obj

