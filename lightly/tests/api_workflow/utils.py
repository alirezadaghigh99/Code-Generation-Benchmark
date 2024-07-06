def generate_id(length: int = 24) -> str:
    return "".join([random.choice(_CHARACTER_SET) for i in range(length)])

