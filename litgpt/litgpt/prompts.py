    def from_name(cls, name: str) -> "PromptStyle":
        return prompt_styles[name]()