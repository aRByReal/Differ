class vertex:
    def __init__(self, id):
        self.id = id
        self.gain = 0
        self.locked = False
        self.list = None
        self.side = None

    def set_gain(self, gain):
        self.gain = gain

    def set_locked(self, locked):
        self.locked = locked

    def set_list_node(self, list_node):
        self.list = list_node

    def set_side(self, side):
        self.side = side

    def __repr__(self):
        return f"Vertex(id={self.id}, gain={self.gain}, locked={self.locked})"