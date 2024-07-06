def unsqueeze(g: jit_utils.GraphContext, self, dim):
    if symbolic_helper._is_constant(dim):
        dim = symbolic_helper._get_const(dim, "i", "dim")

    return symbolic_helper._unsqueeze_helper(g, self, [dim])

def remainder(g: jit_utils.GraphContext, input, other):
    if symbolic_helper._is_fp(input) or symbolic_helper._is_fp(other):
        return opset9.remainder(g, input, other)
    return g.op("Mod", input, other, fmod_i=0)

