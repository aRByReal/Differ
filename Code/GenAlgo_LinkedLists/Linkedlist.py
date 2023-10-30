class DoublyLinkedList:
    class Node:
        def __init__(self, data):
            self.data = data
            self.next = None
            self.prev = None
            self.list_node = None  # pointer to the vertex in the cell array

    def __init__(self, cell_array):
        self.head = None
        self.tail = None
        self.length = 0
        self.cell_array = cell_array

    def add_to_tail(self, data):
        new_node = self.Node(data)
        self.cell_array[data - 1].list_node = new_node  # set pointer to vertex at position data - 1 in the cell_array
        if self.length == 0:
            self.head = new_node
            self.tail = new_node
        else:
            new_node.prev = self.tail
            if self.tail is not None:
                self.tail.next = new_node
            self.tail = new_node
        self.length += 1

    def remove(self, node):
        if self.length == 0:
            raise ValueError("Linked list is empty")
        elif self.length == 1:
            self.head = None
            self.tail = None
        elif node == self.head:
            self.head = self.head.next
            self.head.prev = None
        elif node == self.tail:
            self.tail = self.tail.prev
            self.tail.next = None
        elif node.prev is None and node.next is None:
            self.tail = None
            self.head = None
        else:
            node.prev.next = node.next
            node.next.prev = node.prev
        self.length -= 1

    def remove_by_list_node(self, list_node):
        if list_node is None:
            return
        vertex = list_node.data
        self.remove(list_node)
        self.cell_array[vertex - 1].list_node = None

    def pop_front(self):
        if self.length == 0:
            print("Linked list is empty")
        else:
            data = self.head.data
            self.cell_array[data - 1].list_node = None
            self.remove(self.head)
            return data

    def get_list(self):
        vertices = []
        node = self.head
        while node is not None:
            vertices.append(node.data)
            node = node.next
        return vertices


    def count(self):
        return self.length

    def print_list(self):
        curr_node = self.head
        while curr_node is not None:
            print("The element is:", curr_node.data)
            curr_node = curr_node.next