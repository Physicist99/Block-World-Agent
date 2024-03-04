import copy

class BlockState(object):
    __slots__ = ("initial_arrangement", "goal_arrangement", "elements_count", "moves")
    def __init__(self, initial_arrangement, goal_arrangement, elements_count, moves=None):
        if moves is None:
            moves = []
        self.initial_arrangement = initial_arrangement
        self.goal_arrangement = goal_arrangement
        self.elements_count = elements_count
        self.moves = moves

    def __eq__(self, other):
        return (self.initial_arrangement == other.initial_arrangement and self.goal_arrangement == other.goal_arrangement
                and self.elements_count == other.elements_count and self.moves == other.moves)

    def chose_move(self):
        for idx, lst in enumerate(self.initial_arrangement):
            for idx2, lst2 in enumerate(self.initial_arrangement):
                if idx != idx2:  # don't move to itself lst
                    curr_table, move = self.move_block(self.initial_arrangement, idx, idx2)
                    new_state = BlockState(curr_table, self.goal_arrangement, self.elements_count, copy.copy(self.moves))
                    new_state.moves.append(move)
                    if count_difference(new_state) < count_difference(self):
                        return new_state

        for idx, lst in enumerate(self.initial_arrangement):
            if len(lst) > 1:
                curr_table, move = self.move_block(self.initial_arrangement, idx, -1)
                new_state = BlockState(curr_table, self.goal_arrangement, self.elements_count, copy.copy(self.moves))
                new_state.moves.append(move)
                if count_difference(new_state) <= count_difference(self):
                    return new_state

    def move_block(self, table, start_index, end_index):
        temp_table = copy.deepcopy(table)
        left = temp_table[start_index]
        top_block = left.pop()
        right = []

        if end_index < 0:
            temp_table.append(right)
            move = (top_block, 'Table')
        else:  # move to lst
            right = temp_table[end_index]
            move = (top_block, right[-1])
        right.append(top_block)

        if len(left) == 0:
            temp_table.remove(left)
        return temp_table, move
class BlockWorldAgent:
    __slots__ = ()
    def __init__(self):
        pass

    def solve(self, initial_arrangement, goal_arrangement):
        elements_count = 0
        for ls in initial_arrangement:
            for e in ls:
                elements_count += 1
        state = BlockState(initial_arrangement, goal_arrangement, elements_count)
        solution = arrange_blocks(state)
        return solution
def arrange_blocks(blockState):
    while count_difference(blockState) != 0:
        blockState = blockState.chose_move()
    return blockState.moves
def count_difference(blockState):
    same_num = 0
    # compare each lst on two stacks
    for left in blockState.initial_arrangement:
        for right in blockState.goal_arrangement:
            idx = 0
            while idx < len(left) and idx < len(right):
                if left[idx] == right[idx]:
                    same_num += 1
                    idx += 1
                else:
                    break
    diff = blockState.elements_count - same_num
    return diff
    