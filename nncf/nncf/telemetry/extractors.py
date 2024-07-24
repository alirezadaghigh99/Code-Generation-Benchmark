class CollectedEvent:
    """
    name: The name for the event, will be sent as `event_action`
    data: Optional - the data associated with the event. Must be a string - serialize to string if it is not.
    int_data: Optional - the integer data associated with the event. Must be a positive integer.
    """

    name: str
    data: SerializableData = None  # GA limitations
    int_data: int = None

