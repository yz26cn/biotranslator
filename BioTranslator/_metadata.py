import traceback

def within_flit():
    """\
    Checks if we are being imported by flit.
    This is necessary so flit can import __version__ without all depedencies installed.
    There are a few options to make this hack unnecessary, see:
    https://github.com/takluyver/flit/issues/253#issuecomment-737870438
    """
    for frame in traceback.extract_stack():
        if frame.name == 'get_docstring_and_version_via_import':
            return True
    return False