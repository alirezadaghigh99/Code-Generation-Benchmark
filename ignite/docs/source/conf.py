def run(self):
        for auto in ("autolist", "autolist-classes", "autolist-functions"):
            if auto in self.options:
                # Get current module name
                module_name = self.env.ref_context.get("py:module")
                # Import module
                module = import_module(module_name)

                # Get public names (if possible)
                try:
                    names = getattr(module, "__all__")
                except AttributeError:
                    # Get classes defined in the module
                    cls_names = [
                        name[0]
                        for name in getmembers(module, isclass)
                        if name[-1].__module__ == module_name and not (name[0].startswith("_"))
                    ]
                    # Get functions defined in the module
                    fn_names = [
                        name[0]
                        for name in getmembers(module, isfunction)
                        if (name[-1].__module__ == module_name) and not (name[0].startswith("_"))
                    ]
                    names = cls_names + fn_names
                    # It may happen that module doesn't have any defined class or func
                    if not names:
                        names = [name[0] for name in getmembers(module)]

                # Filter out members w/o doc strings
                filtered_names = []
                for name in names:
                    try:
                        if not name.startswith("_") and getattr(module, name).__doc__ is not None:
                            filtered_names.append(name)
                    except AttributeError:
                        continue

                names = filtered_names

                if auto == "autolist":
                    # Get list of all classes and functions inside module
                    names = [
                        name for name in names if (isclass(getattr(module, name)) or isfunction(getattr(module, name)))
                    ]
                else:
                    if auto == "autolist-classes":
                        # Get only classes
                        check = isclass
                    elif auto == "autolist-functions":
                        # Get only functions
                        check = isfunction
                    else:
                        raise NotImplementedError

                    names = [name for name in names if check(getattr(module, name))]

                # Update content
                self.content = StringList(names)
        return super().run()

