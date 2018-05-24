class UnsupportedExtensionError(Exception):
    def __init__(self, ext):
        message = '{} is not a known file extension'.format(ext)
        super(UnsupportedExtensionError, self).__init__(message)


class UnsupportedFormatError(Exception):
    def __init__(self, fmt):
        message = '{} is not a known annotation format'.format(fmt)
        super(UnsupportedFormatError, self).__init__(message)
