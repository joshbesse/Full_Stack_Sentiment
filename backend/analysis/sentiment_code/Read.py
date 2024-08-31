class Read:
    def file_to_list(self, file):
        words = []
        try:
            with open(file, 'r') as current_file:
                for line in current_file:
                    words.append(line.strip())
        except FileNotFoundError:
            print(f"File not found: {file}")
        return words 