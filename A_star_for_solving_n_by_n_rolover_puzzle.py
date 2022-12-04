import heapq


class Node:
    def __init__(self, parent=None, direction=None, mat=None):
        self.parent = parent
        self.direction = direction
        self.mat = mat
        self.g = 0
        self.h = 0
        self.f = 0
        self.solved_x = 0
        self.solved_y = 0

    def __eq__(self, other):
        return self.mat == other.mat

    def __lt__(self, other):
        return self.f < other.f


def heuristic_seq(node):
    mat = node.mat
    num = len(mat)
    h = 0
    for i in range(num):
        for j in range(num):
            x = final_mat_positions[mat[i][j]][0]
            y = final_mat_positions[mat[i][j]][1]
            if x == phase // num and y == phase % num:
                delta = min(abs(x - i), num - abs(x - i)) + min(abs(4 - j), num - abs(4 - j))
                h = h + delta
    return h


def heuristic(node):
    mat = node.mat
    node.solved_x = solved_x
    node.solved_y = solved_y
    num = len(mat)
    if node.g != 0 and (node.parent.solved_x == solved_x and node.parent.solved_y == solved_y):
        direction = node.direction.split()
        if direction[0] == "right" or direction[0] == "left":
            h = node.parent.h
            for j in range(num):
                row = int(direction[1]) - 1
                x = final_mat_positions[mat[row][j]][0]
                y = final_mat_positions[mat[row][j]][1]
                delta = min(abs(x - row), num - abs(x - row)) + min(abs(y - j), num - abs(y - j))
                if x < solved_x and y < solved_y:
                    h = h + delta
                x = final_mat_positions[node.parent.mat[row][j]][0]
                y = final_mat_positions[node.parent.mat[row][j]][1]
                delta = min(abs(x - row), num - abs(x - row)) + min(abs(y - j), num - abs(y - j))
                if x < solved_x and y < solved_y:
                    h = h - delta
        else:
            h = node.parent.h
            for i in range(num):
                row = int(direction[1]) - 1
                x = final_mat_positions[mat[i][row]][0]
                y = final_mat_positions[mat[i][row]][1]
                delta = min(abs(x - i), num - abs(x - i)) + min(abs(y - row), num - abs(y - row))
                if x < solved_x and y < solved_y:
                    h = h + delta
                x = final_mat_positions[node.parent.mat[i][row]][0]
                y = final_mat_positions[node.parent.mat[i][row]][1]
                delta = min(abs(x - i), num - abs(x - i)) + min(abs(y - row), num - abs(y - row))
                if x < solved_x and y < solved_y:
                    h = h - delta
    else:
        h = 0
        for i in range(num):
            for j in range(num):
                x = final_mat_positions[mat[i][j]][0]
                y = final_mat_positions[mat[i][j]][1]
                if x < solved_x and y < solved_y:
                    delta = min(abs(x - i), num - abs(x - i)) + min(abs(y - j), num - abs(y - j))
                    h = h + delta
    return h


def shift_right(mat, row):
    new_mat = []
    k = len(mat)
    for i in range(k):
        if i != row:
            new_row = []
            for j in range(k):
                new_row.append(mat[i][j])
            new_mat.append(new_row)
        else:
            new_row = [mat[row][k - 1]]
            for j in range(k - 1):
                new_row.append(mat[i][j])
            new_mat.append(new_row)
    return new_mat


def shift_left(mat, row):
    new_mat = []
    k = len(mat)
    for i in range(k):
        if i != row:
            new_row = []
            for j in range(k):
                new_row.append(mat[i][j])
            new_mat.append(new_row)
        else:
            new_row = []
            for j in range(1, k):
                new_row.append(mat[i][j])
            new_row.append(mat[row][0])
            new_mat.append(new_row)
    return new_mat


def shift_down(mat, col):
    new_mat = []
    k = len(mat)
    for i in range(k):
        new_row = []
        for j in range(k):
            new_row.append(mat[i][j])
        new_mat.append(new_row)
    arr = []
    for i in range(k):
        arr.append(mat[i][col])
    for i in range(k - 1):
        new_mat[i + 1][col] = arr[i]
    new_mat[0][col] = arr[k - 1]
    return new_mat


def shift_up(mat, col):
    new_mat = []
    k = len(mat)
    for i in range(k):
        new_row = []
        for j in range(k):
            new_row.append(mat[i][j])
        new_mat.append(new_row)
    arr = []
    for i in range(k):
        arr.append(mat[i][col])
    for i in range(k - 1):
        new_mat[i][col] = arr[i + 1]
    new_mat[k - 1][col] = arr[0]
    return new_mat


visited_nodes = set()
solved_x = 1
solved_y = 1
phase = 0


def add_to_visited(visited_node):
    global visited_nodes
    state = visited_node.mat
    visited_nodes.add(tuple(i for row in state for i in row))


def is_visited(new_node):
    global visited_nodes
    state = new_node.mat
    return tuple(i for row in state for i in row) in visited_nodes


def clear(node, row):
    new_node = Node(node, "right " + str(row + 1), shift_right(node.mat, row))
    for i in range(4):
        num = 0
        while final_mat_positions[new_node.mat[row][4]][0] == row and num < 10:
            num = num + 1
            new_node = Node(new_node, "down " + str(5), shift_down(new_node.mat, 4))
        new_node = Node(new_node, "right " + str(row + 1), shift_right(new_node.mat, row))
    return new_node


def astar_seq(initial_mat, final_mat):
    global phase
    start_node = Node(None, None, initial_mat)
    start_node.g = 0
    start_node.f = start_node.h = heuristic_seq(start_node)
    end_node = Node(None, None, final_mat)
    end_node.g = end_node.h = end_node.f = 0

    open_list = []

    first_root = clear(start_node, 0)
    heapq.heappush(open_list, first_root)
    while True > 0:
        current_node = heapq.heappop(open_list)
        children = []
        for i in range(phase // 5 + 1, len(initial_mat)):
            new_node = Node(current_node, "right " + str(i + 1), shift_right(current_node.mat, i))
            if current_node.direction != "left " + str(i + 1):
                children.append(new_node)

            new_node = Node(current_node, "left " + str(i + 1), shift_left(current_node.mat, i))
            if current_node.direction != "right " + str(i + 1):
                children.append(new_node)

        for i in range(4, len(initial_mat)):
            new_node = Node(current_node, "down " + str(i + 1), shift_down(current_node.mat, i))
            if current_node.direction != "up " + str(i + 1):
                children.append(new_node)

            new_node = Node(current_node, "up " + str(i + 1), shift_up(current_node.mat, i))
            if current_node.direction != "down " + str(i + 1):
                children.append(new_node)

        for child in children:
            if False is False:
                child.g = current_node.g + 1
                child.h = heuristic_seq(child)
                child.f = child.g + child.h
                if child.h == 0:
                    if phase == 19:
                        direction = []
                        current = child
                        mat = current.mat
                        while current is not None:
                            direction.append(current.direction)
                            current = current.parent
                        return direction[::-1], mat
                    else:
                        open_list = []
                        new_root = Node(child, "left " + str(phase // 5 + 1), shift_left(child.mat, phase // 5))
                        phase = phase + 1
                        if phase == 4:
                            phase = phase + 1
                            new_root = clear(new_root, 1)
                        if phase == 9:
                            phase = phase + 1
                            new_root = clear(new_root, 2)
                        if phase == 14:
                            phase = phase + 1
                            new_root = clear(new_root, 3)
                        heapq.heappush(open_list, new_root)
                        break
                heapq.heappush(open_list, child)


def astar(initial_mat, final_mat):
    global solved_x, solved_y
    start_node = Node(None, None, initial_mat)
    start_node.g = 0
    start_node.f = start_node.h = heuristic(start_node)
    end_node = Node(None, None, final_mat)
    end_node.g = end_node.h = end_node.f = 0

    open_list = []

    if start_node == end_node:
        path = []
        direction = []
        current = start_node
        while current is not None:
            path.append(current.mat)
            direction.append(current.direction)
            current = current.parent
        return path[::-1], direction[::-1]

    heapq.heappush(open_list, start_node)
    add_to_visited(start_node)
    while True > 0:
        current_node = heapq.heappop(open_list)
        children = []
        bound = solved_y - 2
        if len(end_node.mat) == 4:
            if solved_x == 2:
                bound = solved_x - 3
        if len(end_node.mat) == 5:
            bound = solved_x - 1
        for i in range(max(0, bound), len(initial_mat)):
            new_node = Node(current_node, "right " + str(i + 1), shift_right(current_node.mat, i))
            if current_node.direction != "left " + str(i + 1):
                children.append(new_node)

            new_node = Node(current_node, "down " + str(i + 1), shift_down(current_node.mat, i))
            if current_node.direction != "up " + str(i + 1):
                children.append(new_node)

            new_node = Node(current_node, "left " + str(i + 1), shift_left(current_node.mat, i))
            if current_node.direction != "right " + str(i + 1):
                children.append(new_node)

            new_node = Node(current_node, "up " + str(i + 1), shift_up(current_node.mat, i))
            if current_node.direction != "down " + str(i + 1):
                children.append(new_node)

        for child in children:
            if is_visited(child) is False:
                child.g = current_node.g + 1
                child.h = heuristic(child)
                child.f = child.g + child.h
                if child.h == 0:
                    end = len(child.mat)
                    if end == 5:
                        end = 5
                    if solved_x == end and solved_y == end or child.mat == end_node.mat or (
                            solved_x == 1 and len(child.mat) == 3):
                        direction = []
                        current = child
                        while current is not None:
                            direction.append(current.direction)
                            current = current.parent
                        return direction[::-1]
                    else:
                        solved_x = solved_x + 1
                        solved_y = solved_y + 1
                        open_list = []
                heapq.heappush(open_list, child)
                add_to_visited(child)


final_mat_positions = [[0, 0] for i in range(40)]
times = []


def main():
    k = int(input())
    global solved_x
    if k == 3:
        solved_x = 2
    final_mat = []
    initial_mat = []

    for i in range(k):
        single_row = list(map(int, input().split()))
        final_mat.append(single_row)
    for i in range(k):
        single_row = list(map(int, input().split()))
        initial_mat.append(single_row)
    for i in range(k):
        for j in range(k):
            number = final_mat[i][j]
            final_mat_positions[number][0] = i
            final_mat_positions[number][1] = j
    if k < 5:
        directions = astar(initial_mat, final_mat)
        print(len(directions) - 1)
        for i in range(len(directions) - 1):
            print(directions[i + 1])
    else:
        directions, mid_mat = astar_seq(initial_mat, final_mat)
        direction = astar(mid_mat, final_mat)
        print(len(directions) + len(direction) - 2)
        for i in range(len(directions) - 1):
            print(directions[i + 1])
        for i in range(len(direction) - 1):
            print(direction[i + 1])


if __name__ == '__main__':
    main()
