class FilenameAndReadUrl(BaseModel):
    """
    Filename and corresponding read url for a sample in a tag
    """
    file_name: StrictStr = Field(..., alias="fileName")
    read_url: StrictStr = Field(..., alias="readUrl", description="A URL which allows anyone in possession of said URL to access the resource")
    __properties = ["fileName", "readUrl"]

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
    def from_json(cls, json_str: str) -> FilenameAndReadUrl:
        """Create an instance of FilenameAndReadUrl from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> FilenameAndReadUrl:
        """Create an instance of FilenameAndReadUrl from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return FilenameAndReadUrl.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in FilenameAndReadUrl) in the input: " + str(obj))

        _obj = FilenameAndReadUrl.parse_obj({
            "file_name": obj.get("fileName"),
            "read_url": obj.get("readUrl")
        })
        return _obj

