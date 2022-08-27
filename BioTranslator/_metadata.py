import traceback

def within_flit():
    for frame in traceback.extract_stack():
        if frame.name == 'get_docstring_and_version_via_import':
            return True
    return False
