def _expand_hostlist(nodelist: str) -> List[str]:
        """Expand a compressed hostlist string and returns all hosts listed.

        Source : https://github.com/LLNL/py-hostlist/blob/master/hostlist/hostlist.py

        Args:
            nodelist: Compressed hostlist string

        .. note::
            The host names can be composed by any character except the special ones `[`, `]`, `,`. Only one
            sequence `[...]` is supported per hostname.

        .. versionadded:: 0.4.6
        """
        result_hostlist = []

        nodelist_match = r"([^,\[\]]+\[[^\[\]]*\][^,\[\]]*|[^,\[\]]*),?"

        nodelist = nodelist.replace(" ", "")

        for node in re.findall(nodelist_match, nodelist):
            node_match = r"(.+)\[((,?[0-9]+-?,?-?){0,})\](.*)?"

            match = re.search(node_match, node)

            if match is None:
                if node:
                    result_hostlist.append(node)
            else:
                # holds the ranges of nodes as a string
                # now we can manipulate the string and cast it to a list of numbers
                num = str(match.group(2)).replace("[", "").replace("]", "")

                if len(num) == 0:
                    raise ValueError(f"hostlist invalid : {nodelist}")

                num_list = num.split(",")

                # find range of node numbers
                ranges = [elem.split("-") if "-" in elem else [elem, elem] for elem in num_list]

                # if the node numbers contain leading zeros, store them to be
                lead_zeros = max([len(s) - len(s.lstrip("0")) for s, _ in ranges])

                # list of expanded ranges of node numbers
                nodes_list = [list(range(int(s), int(e) + 1)) for s, e in ranges]

                # flat the list
                final_list = [item for sublist in nodes_list for item in sublist]

                # put final list in ascending order and append cluster name to each node number
                final_list = list(sorted(set(final_list)))

                # prepend leading zeros to numbers required
                hostlist_tmp = [str(elem).zfill(lead_zeros + 1) for elem in final_list]

                # append hostname to the node numbers
                hostlist_no_suffix = [match.group(1) + elem for elem in hostlist_tmp]

                # append suffix to hostlist if there is one
                final_hostlist = [elem + match.group(4) for elem in hostlist_no_suffix]

                result_hostlist += final_hostlist

        return result_hostlist

def _setup_ddp_vars_from_slurm_env(environ: Dict[str, str]) -> Dict[str, Union[str, int]]:
        """Method to setup DDP env vars required by PyTorch from SLURM env"""
        # 1) Tools like enroot can have hooks to translate slurm env vars to RANK, LOCAL_RANK, WORLD_SIZE etc
        # See https://github.com/NVIDIA/enroot/blob/v3.1.0/conf/hooks/extra/50-slurm-pytorch.sh
        # 2) User can use torch.distributed.launch tool to schedule on N local GPUs using 1 node, 1 task by SLURM
        # To cover case 1), let's ensure that defined RANK == SLURM_PROCID, LOCAL_RANK == SLURM_LOCALID,
        #   WORLD_SIZE == SLURM_NTASKS. We will use defined MASTER_ADDR and MASTER_PORT instead of defining
        #   them by our means
        # To cover case 2), let's check that defined RANK >= SLURM_PROCID, LOCAL_RANK >= SLURM_LOCALID,
        #   WORLD_SIZE >= SLURM_NTASKS, SLURM_JOB_NUM_NODES == 1

        ddp_vars: Dict[str, Union[str, int, None]] = {
            "RANK": int(environ["SLURM_PROCID"]),
            "LOCAL_RANK": int(environ["SLURM_LOCALID"]),
            "WORLD_SIZE": int(environ["SLURM_NTASKS"]),
            "MASTER_ADDR": None,
            "MASTER_PORT": None,
        }

        pth_ddp_env_vars = {key: environ.get(key, None) for key in ddp_vars}
        defined_pth_ddp_env_vars = [v is not None for v in pth_ddp_env_vars.values()]
        if all(defined_pth_ddp_env_vars):
            nnodes = int(environ["SLURM_JOB_NUM_NODES"])
            if nnodes > 1:
                # ensure that all pth_ddp_env_vars are consistent with slurm vars
                for key in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]:
                    slurm_var = cast(int, ddp_vars[key])
                    pth_var = int(cast(str, pth_ddp_env_vars[key]))
                    if slurm_var != pth_var:
                        raise RuntimeError(
                            "Environment variable defined for PyTorch Distributed context is inconsistent with "
                            f"equivalent SLURM env variable. {key}: {pth_var} vs {slurm_var}\n"
                            f"SLURM vars: {ddp_vars}\n"
                            f"PTH vars: {pth_ddp_env_vars}\n"
                        )
            else:
                # ensure that PTH RANK >= SLURM_PROCID, PTH LOCAL_RANK >= SLURM_LOCALID,
                # PTH WORLD_SIZE >= SLURM_NTASKS
                for key in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]:
                    slurm_var = cast(int, ddp_vars[key])
                    pth_var = int(cast(str, pth_ddp_env_vars[key]))
                    if pth_var < slurm_var:
                        raise RuntimeError(
                            "Environment variable defined for PyTorch Distributed context is "
                            "inconsistent with equivalent SLURM env variable. "
                            f"We expect that {key}: {pth_var} >= {slurm_var}\n"
                            f"SLURM vars: {ddp_vars}\n"
                            f"PTH vars: {pth_ddp_env_vars}\n"
                        )
                    ddp_vars[key] = pth_var
            # set up MASTER_ADDR and MASTER_PORT from PTH
            ddp_vars["MASTER_ADDR"] = cast(str, pth_ddp_env_vars["MASTER_ADDR"])
            ddp_vars["MASTER_PORT"] = int(cast(str, pth_ddp_env_vars["MASTER_PORT"]))
        elif any(defined_pth_ddp_env_vars):
            # Let's warn user about PTH env variables that we could not taken into account
            warnings.warn(
                "We detected the following env variables: "
                f"{[(k, v) for k, v in pth_ddp_env_vars.items() if v is not None]},\n"
                "but will not take them into account as the following env vars are missing:"
                f"{[k for k, v in pth_ddp_env_vars.items() if v is None]},\n"
            )

        if ddp_vars["MASTER_ADDR"] is None:
            nodelist = environ["SLURM_JOB_NODELIST"]
            try:
                # use scontrol to expand hostname list
                hostnames = subprocess.check_output(["scontrol", "show", "hostnames", nodelist])
                method = "scontrol"
            except FileNotFoundError:
                # expand hostname list as scontrol
                hostnames = " ".join(_expand_hostlist(nodelist)).encode("utf-8")
                method = "ignite"
            # at least one hostname should be defined
            hostname_list = hostnames.split()
            if len(hostname_list) < 1:
                raise RuntimeError(f"No hostname detected in SLURM_JOB_NODELIST by {method} (nodelist={nodelist})")
            # master address is the first hostname of nodes list
            ddp_vars["MASTER_ADDR"] = str(hostname_list[0].decode("utf-8"))

        if ddp_vars["MASTER_PORT"] is None:
            # port should be the same over all process
            slurm_port = environ["SLURM_JOB_ID"]
            slurm_port = slurm_port[-4:]
            ddp_vars["MASTER_PORT"] = int(slurm_port) + 15000

        return cast(Dict[str, Union[str, int]], ddp_vars)

