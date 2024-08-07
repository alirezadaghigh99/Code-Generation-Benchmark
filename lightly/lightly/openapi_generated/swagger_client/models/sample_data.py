class SampleData(BaseModel):
    """
    SampleData
    """
    id: constr(strict=True) = Field(..., description="MongoDB ObjectId")
    type: SampleType = Field(...)
    dataset_id: Optional[constr(strict=True)] = Field(None, alias="datasetId", description="MongoDB ObjectId")
    file_name: StrictStr = Field(..., alias="fileName")
    thumb_name: Optional[StrictStr] = Field(None, alias="thumbName")
    exif: Optional[Dict[str, Any]] = None
    index: Optional[StrictInt] = None
    created_at: Optional[conint(strict=True, ge=0)] = Field(None, alias="createdAt", description="unix timestamp in milliseconds")
    last_modified_at: Optional[conint(strict=True, ge=0)] = Field(None, alias="lastModifiedAt", description="unix timestamp in milliseconds")
    meta_data: Optional[SampleMetaData] = Field(None, alias="metaData")
    custom_meta_data: Optional[Dict[str, Any]] = Field(None, alias="customMetaData")
    video_frame_data: Optional[VideoFrameData] = Field(None, alias="videoFrameData")
    crop_data: Optional[CropData] = Field(None, alias="cropData")
    __properties = ["id", "type", "datasetId", "fileName", "thumbName", "exif", "index", "createdAt", "lastModifiedAt", "metaData", "customMetaData", "videoFrameData", "cropData"]

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
    def from_json(cls, json_str: str) -> SampleData:
        """Create an instance of SampleData from a JSON string"""
        return cls.from_dict(json.loads(json_str))

    def to_dict(self, by_alias: bool = False):
        """Returns the dictionary representation of the model"""
        _dict = self.dict(by_alias=by_alias,
                          exclude={
                          },
                          exclude_none=True)
        # override the default output from pydantic by calling `to_dict()` of meta_data
        if self.meta_data:
            _dict['metaData' if by_alias else 'meta_data'] = self.meta_data.to_dict(by_alias=by_alias)
        # override the default output from pydantic by calling `to_dict()` of video_frame_data
        if self.video_frame_data:
            _dict['videoFrameData' if by_alias else 'video_frame_data'] = self.video_frame_data.to_dict(by_alias=by_alias)
        # override the default output from pydantic by calling `to_dict()` of crop_data
        if self.crop_data:
            _dict['cropData' if by_alias else 'crop_data'] = self.crop_data.to_dict(by_alias=by_alias)
        # set to None if thumb_name (nullable) is None
        # and __fields_set__ contains the field
        if self.thumb_name is None and "thumb_name" in self.__fields_set__:
            _dict['thumbName' if by_alias else 'thumb_name'] = None

        # set to None if exif (nullable) is None
        # and __fields_set__ contains the field
        if self.exif is None and "exif" in self.__fields_set__:
            _dict['exif' if by_alias else 'exif'] = None

        # set to None if custom_meta_data (nullable) is None
        # and __fields_set__ contains the field
        if self.custom_meta_data is None and "custom_meta_data" in self.__fields_set__:
            _dict['customMetaData' if by_alias else 'custom_meta_data'] = None

        # set to None if video_frame_data (nullable) is None
        # and __fields_set__ contains the field
        if self.video_frame_data is None and "video_frame_data" in self.__fields_set__:
            _dict['videoFrameData' if by_alias else 'video_frame_data'] = None

        # set to None if crop_data (nullable) is None
        # and __fields_set__ contains the field
        if self.crop_data is None and "crop_data" in self.__fields_set__:
            _dict['cropData' if by_alias else 'crop_data'] = None

        return _dict

    @classmethod
    def from_dict(cls, obj: dict) -> SampleData:
        """Create an instance of SampleData from a dict"""
        if obj is None:
            return None

        if not isinstance(obj, dict):
            return SampleData.parse_obj(obj)

        # raise errors for additional fields in the input
        for _key in obj.keys():
            if _key not in cls.__properties:
                raise ValueError("Error due to additional fields (not defined in SampleData) in the input: " + str(obj))

        _obj = SampleData.parse_obj({
            "id": obj.get("id"),
            "type": obj.get("type"),
            "dataset_id": obj.get("datasetId"),
            "file_name": obj.get("fileName"),
            "thumb_name": obj.get("thumbName"),
            "exif": obj.get("exif"),
            "index": obj.get("index"),
            "created_at": obj.get("createdAt"),
            "last_modified_at": obj.get("lastModifiedAt"),
            "meta_data": SampleMetaData.from_dict(obj.get("metaData")) if obj.get("metaData") is not None else None,
            "custom_meta_data": obj.get("customMetaData"),
            "video_frame_data": VideoFrameData.from_dict(obj.get("videoFrameData")) if obj.get("videoFrameData") is not None else None,
            "crop_data": CropData.from_dict(obj.get("cropData")) if obj.get("cropData") is not None else None
        })
        return _obj

