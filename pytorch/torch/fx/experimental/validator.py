        def sqrt(self, number: z3.ArithRef) -> z3.ArithRef:
            # Square-root:
            # 1. Only work with reals
            number = _Z3Ops.to_real(number)
            # 2. The number should be positive or zero.
            #    Otherwise, Z3 returns 'unknown'.
            self.validator.add_assertion(number >= 0)
            return number ** 0.5