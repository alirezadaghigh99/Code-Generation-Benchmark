def format(self, object, context, maxlevels, level):
        return _safe_repr(
            object, context, maxlevels, level, changed_only=self._changed_only
        )

