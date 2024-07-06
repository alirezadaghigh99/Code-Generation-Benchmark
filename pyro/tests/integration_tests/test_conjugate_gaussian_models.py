def add_edge(s):
            deps = []
            if s == "1":
                deps.extend(["1L", "1R"])
            else:
                if s[-1] == "R":
                    deps.append(s[0:-1] + "L")
                if len(s) < self.N:
                    deps.extend([s + "L", s + "R"])
                for k in range(len(s) - 2):
                    base = s[1 : -1 - k]
                    if base[-1] == "R":
                        deps.append("1" + base[:-1] + "L")
            for dep in deps:
                g.add_edge("loc_latent_" + dep, "loc_latent_" + s)

