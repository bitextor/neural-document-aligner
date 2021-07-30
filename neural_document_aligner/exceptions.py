
class FileFoundError(Exception):

    def __init__(self, msg, prefix_msg="File or directory does exist"):
        super().__init__(f"{prefix_msg}: {msg}")

class DirNotFoundError(Exception):

    def __init__(self, msg, prefix_msg="Directory does not exist"):
        super().__init__(f"{prefix_msg}: {msg}")
