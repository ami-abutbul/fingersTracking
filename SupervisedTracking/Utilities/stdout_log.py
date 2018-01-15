
class StdoutLog(object):
    def __init__(self, log_path, stdout, print_to_log=True, print_to_stdout=True):
        if print_to_log:
            self.log = open(log_path, 'w')
        self.str = ""
        self.stdout = stdout
        self.print_to_log = print_to_log
        self.print_to_stdout = print_to_stdout

    def write(self, incoming_str):
        if incoming_str.endswith('\n'):
            output = self.str + incoming_str

            if self.print_to_log:
                self.log.write(output)
                self.log.flush()

            if self.print_to_stdout:
                self.stdout.write(output)
                self.stdout.flush()

            self.str = ""
        else:
            self.str = self.str + incoming_str

    def flush(self):
        self.stdout.flush()
