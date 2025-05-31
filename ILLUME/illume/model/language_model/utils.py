class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def convert_llm_output(output):
    return output
    # return DotDict(output.__dict__)  # legacy
