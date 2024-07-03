    def get_context(self, code: types.CodeType):
        ctx = self.code_context.get(code)
        if ctx is None:
            ctx = {}
            self.code_context[code] = ctx
        return ctx