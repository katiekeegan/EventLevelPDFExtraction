class tqdm:
    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable
        
    def __iter__(self):
        return iter(self.iterable)
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        pass
        
    def set_description(self, desc):
        pass
        
    def update(self, n=1):
        pass