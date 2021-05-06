class ObjDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    @classmethod
    def read_from_file_python3(cls,input_path,attr_name="config"):
        import importlib
        spec = importlib.util.spec_from_file_location("cfg",input_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod,attr_name)

    @classmethod
    def read_all_from_file_python3(cls,input_path,):
        import importlib
        spec = importlib.util.spec_from_file_location("cfg",input_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
