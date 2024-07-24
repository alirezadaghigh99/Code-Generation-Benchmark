class InferenceRequestImage(BaseModel):
    """Image data for inference request.

    Attributes:
        type (str): The type of image data provided, one of 'url', 'base64', or 'numpy'.
        value (Optional[Any]): Image data corresponding to the image type.
    """

    type: str = Field(
        examples=["url"],
        description="The type of image data provided, one of 'url', 'base64', or 'numpy'",
    )
    value: Optional[Any] = Field(
        None,
        examples=["http://www.example-image-url.com"],
        description="Image data corresponding to the image type, if type = 'url' then value is a string containing the url of an image, else if type = 'base64' then value is a string containing base64 encoded image data, else if type = 'numpy' then value is binary numpy data serialized using pickle.dumps(); array should 3 dimensions, channels last, with values in the range [0,255].",
    )

