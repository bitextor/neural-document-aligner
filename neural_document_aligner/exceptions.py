
class FileFoundError(Exception):

    def __init__(self, msg, prefix_msg="File or directory does exist"):
        super().__init__(f"{prefix_msg}: {msg}")
