import numpy as np
import matplotlib.pyplot as plt

rows, cols = 10, 10
# 0 = North, 1 = East, 2 = South, 3 = West
discount_rate = 0.9
def create_grid():
    # Create a 10x10 grid
    grid = np.ones((rows, cols))

    # Set the outer part to -10
    grid[0, :] = +10
    grid[-1, :] = +10
    grid[:, 0] = +10
    grid[:, -1] = +10

    # Obstacle
    grid[7, 3:7] = +10
    grid[2:6, 4] = +10
    grid[2, 5] = +10
    grid[4:6, 7] = +10

    # Reward
    grid[8,8] = -21

    return grid
grid = create_grid()
# GRID INPUT 10 -10 and 1

def create_transition_matrix():
    # Create transition matrix
    transition_matrix = np.zeros((rows**2, cols**2,4))
    # 0 = North, 1 = East, 2 = South, 3 = West

    for i in range(rows):
        for j in range(cols):
            # Obstacle
            if grid[i, j] == 10:
                transition_matrix[(i*rows)+i,(j*cols)+j, :] = 1

            # Goal
            if grid[i,j] < -5:
                transition_matrix[(i * rows) + i, (j * cols) + j,:] = 1

            # Anywhere else
            if grid[i,j] ==1:
                current_row = (i*rows)+i
                current_col = (j*cols)+j

                #South movement
                transition_matrix[current_row+1, current_col,2] += 0.7
                transition_matrix[current_row - 1, current_col,2] += 0.1
                transition_matrix[current_row , current_col-1,2] += 0.1
                transition_matrix[current_row , current_col+1,2] += 0.1

                # North movement
                transition_matrix[current_row + 1, current_col,0] += 0.1
                transition_matrix[current_row - 1, current_col,0] += 0.7
                transition_matrix[current_row, current_col - 1,0] += 0.1
                transition_matrix[current_row, current_col + 1,0] += 0.1

                # East movement
                transition_matrix[current_row + 1, current_col,1] += 0.1
                transition_matrix[current_row - 1, current_col,1] += 0.1
                transition_matrix[current_row, current_col - 1,1] += 0.1
                transition_matrix[current_row, current_col + 1,1] += 0.7

                # West movement
                transition_matrix[current_row + 1, current_col,3] += 0.1
                transition_matrix[current_row - 1, current_col,3] += 0.1
                transition_matrix[current_row, current_col - 1,3] += 0.7
                transition_matrix[current_row, current_col + 1,3] += 0.1
    return transition_matrix

def value_sweep(grid, policy, threshold_min):
    score_cumm = grid
    threshold = 10
    grid2 = np.zeros((rows, cols))
    while threshold > threshold_min:
        score_cumm = discount_rate * score_cumm
        for i in range(rows):
            for j in range(cols):
                current_row_trans = (i * rows) + i
                current_col_trans = (j * cols) + j

                current_row_grid = i
                current_col_grid = j
                current_policy = int(policy[i, j])
                # Cannot be -1 or +100

                # Initialize probabilities of translation to None for edges
                prob_north = 0 if current_row_trans == 0 else transition_matrix[
                    current_row_trans - 1, current_col_trans, current_policy]
                prob_south = 0 if current_row_trans == rows ** 2 - 1 else transition_matrix[
                    current_row_trans + 1, current_col_trans, current_policy]
                prob_east = 0 if current_col_trans == cols ** 2 - 1 else transition_matrix[
                    current_row_trans, current_col_trans + 1, current_policy]
                prob_west = 0 if current_col_trans == 0 else transition_matrix[
                    current_row_trans, current_col_trans - 1, current_policy]
                prob_curr = transition_matrix[current_row_trans, current_col_trans, current_policy]

                # Initialize score for grid
                score_north = 0 if current_row_grid == 0 else score_cumm[current_row_grid - 1, current_col_grid]
                score_south = 0 if current_row_grid == rows - 1 else score_cumm[current_row_grid + 1, current_col_grid]
                score_east = 0 if current_col_grid == cols - 1 else score_cumm[current_row_grid, current_col_grid + 1]
                score_west = 0 if current_col_grid == 0 else score_cumm[current_row_grid, current_col_grid - 1]
                score_curr = score_cumm[current_row_grid, current_col_grid]

                sum_score = prob_north * score_north + prob_south * score_south + prob_east * score_east + prob_west * score_west + prob_curr * score_curr
                score_cumm[current_row_grid, current_col_grid] = grid[current_row_grid, current_col_grid] + sum_score

            # Update grid after the loop
        threshold = np.max(np.abs(score_cumm - grid2))
        grid2 = score_cumm
    return score_cumm

def choose_best_policy(grid):
    policy = 0
    """Improve Policy"""
    comparison_grid = np.zeros((rows, cols, 4))
    while policy < 4:
        score_cumm = grid
        score_cumm = discount_rate * score_cumm
        grid2 = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                current_row_trans = (i * rows) + i
                current_col_trans = (j * cols) + j

                current_row_grid = i
                current_col_grid = j
                current_policy = policy
                # Cannot be -1 or +100

                # Initialize probabilities of translation to None for edges
                prob_north = 0 if current_row_trans == 0 else transition_matrix[
                    current_row_trans - 1, current_col_trans, current_policy]
                prob_south = 0 if current_row_trans == rows ** 2 - 1 else transition_matrix[
                    current_row_trans + 1, current_col_trans, current_policy]
                prob_east = 0 if current_col_trans == cols ** 2 - 1 else transition_matrix[
                    current_row_trans, current_col_trans + 1, current_policy]
                prob_west = 0 if current_col_trans == 0 else transition_matrix[
                    current_row_trans, current_col_trans - 1, current_policy]
                prob_curr = transition_matrix[current_row_trans, current_col_trans, current_policy]

                # Initialize score for grid
                score_north = 0 if current_row_grid == 0 else score_cumm[current_row_grid - 1, current_col_grid]
                score_south = 0 if current_row_grid == rows - 1 else score_cumm[current_row_grid + 1, current_col_grid]
                score_east = 0 if current_col_grid == cols - 1 else score_cumm[current_row_grid, current_col_grid + 1]
                score_west = 0 if current_col_grid == 0 else score_cumm[current_row_grid, current_col_grid - 1]
                score_curr = score_cumm[current_row_grid, current_col_grid]

                sum_score = prob_north * score_north + prob_south * score_south + prob_east * score_east + prob_west * score_west + prob_curr * score_curr
                grid2[current_row_grid, current_col_grid] = grid[current_row_grid, current_col_grid] + sum_score
        comparison_grid[:, :, policy] = grid2
        policy += 1

    A, B, C, D = comparison_grid[:, :, 0], comparison_grid[:, :, 1], comparison_grid[:, :, 2], comparison_grid[:, :, 3]
    stacked_matrices = np.stack((A, B, C, D))
    max_indices = np.argmin(stacked_matrices, axis=0)
    return max_indices

def print_first_4():
    """First Val Sweep"""
    init_policy = np.ones((rows, cols))
    init_policy[:,:] = 1
    grid1 = value_sweep(grid, init_policy, 0.2)
    reversed_policy = init_policy[::-1]
    plt.figure()
    plt.imshow(grid1, cmap='winter', interpolation='nearest',extent=[0, 10, 0, 10])
    plt.title('1')
    plt.colorbar()
    arrow_length = 0.4
    obstacle = [(2, 5), (2, 4), (2, 3), (2, 6), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
                (0, 9), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (1, 0), (2, 0),
                (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9),
                (7, 9), (8, 9), (9, 9), (1, 8), (4, 4), (5, 4), (6, 4), (7, 4), (7, 5), (4, 7), (5, 7)]
    for i in range(init_policy.shape[0] - 1, -1, -1):  # Reverse the direction of i
        for j in range(init_policy.shape[1]):
            direction = reversed_policy[i, j]  # Corrected variable name
            if (i, j) in obstacle:
                continue
            if direction == 0:  # North
                dx, dy = 0, arrow_length
            elif direction == 1:  # East
                dx, dy = arrow_length, 0
            elif direction == 2:  # South
                dx, dy = 0, -arrow_length
            else:  # West
                dx, dy = -arrow_length, 0
            # Calculate starting and ending points of the arrow
            start_x = j + 0.5  # Adjusted for the shift
            start_y = i + 0.5  # Adjusted for the shift
            end_x = max(0.5, min(9.5, start_x + dx))  # Ensure end_x stays within 0.5 to 9.5
            end_y = max(0.5, min(9.5, start_y + dy))  # Ensure end_y stays within 0.5 to 9.5
            plt.arrow(start_x, start_y, dx, dy, head_width=0.3, head_length=0.3, fc='k', ec='k')
    # 0 = North, 1 = East, 2 = South, 3 = West
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    new_policy = choose_best_policy(grid1)
    reversed_policy = new_policy[::-1]
    grid2 = value_sweep(grid, new_policy, 0.2)
    plt.figure()
    plt.imshow(grid2, cmap='winter', interpolation='nearest', extent=[0, 10, 0, 10])
    plt.title('2')
    plt.colorbar()
    arrow_length = 0.4
    obstacle = [(2, 5), (2, 4), (2, 3), (2, 6), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
                (0, 9), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (1, 0), (2, 0),
                (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9),
                (7, 9), (8, 9), (9, 9), (1, 8), (4, 4), (5, 4), (6, 4), (7, 4), (7, 5), (4, 7), (5, 7)]
    for i in range(init_policy.shape[0] - 1, -1, -1):  # Reverse the direction of i
        for j in range(init_policy.shape[1]):
            direction = reversed_policy[i, j]  # Corrected variable name
            if (i, j) in obstacle:
                continue
            if direction == 0:  # North
                dx, dy = 0, arrow_length
            elif direction == 1:  # East
                dx, dy = arrow_length, 0
            elif direction == 2:  # South
                dx, dy = 0, -arrow_length
            else:  # West
                dx, dy = -arrow_length, 0
            # Calculate starting and ending points of the arrow
            start_x = j + 0.5  # Adjusted for the shift
            start_y = i + 0.5  # Adjusted for the shift
            end_x = max(0.5, min(9.5, start_x + dx))  # Ensure end_x stays within 0.5 to 9.5
            end_y = max(0.5, min(9.5, start_y + dy))  # Ensure end_y stays within 0.5 to 9.5
            plt.arrow(start_x, start_y, dx, dy, head_width=0.3, head_length=0.3, fc='k', ec='k')
    # 0 = North, 1 = East, 2 = South, 3 = West
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    new_policy = choose_best_policy(grid2)
    reversed_policy = new_policy[::-1]
    grid3 = value_sweep(grid, new_policy, 0.2)
    plt.figure()
    plt.imshow(grid3, cmap='winter', interpolation='nearest', extent=[0, 10, 0, 10])
    plt.title('3')
    plt.colorbar()
    arrow_length = 0.4
    obstacle = [(2, 5), (2, 4), (2, 3), (2, 6), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
                (0, 9), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (1, 0), (2, 0),
                (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9),
                (7, 9), (8, 9), (9, 9), (1, 8), (4, 4), (5, 4), (6, 4), (7, 4), (7, 5), (4, 7), (5, 7)]
    for i in range(init_policy.shape[0] - 1, -1, -1):  # Reverse the direction of i
        for j in range(init_policy.shape[1]):
            direction = reversed_policy[i, j]  # Corrected variable name
            if (i, j) in obstacle:
                continue
            if direction == 0:  # North
                dx, dy = 0, arrow_length
            elif direction == 1:  # East
                dx, dy = arrow_length, 0
            elif direction == 2:  # South
                dx, dy = 0, -arrow_length
            else:  # West
                dx, dy = -arrow_length, 0
            # Calculate starting and ending points of the arrow
            start_x = j + 0.5  # Adjusted for the shift
            start_y = i + 0.5  # Adjusted for the shift
            end_x = max(0.5, min(9.5, start_x + dx))  # Ensure end_x stays within 0.5 to 9.5
            end_y = max(0.5, min(9.5, start_y + dy))  # Ensure end_y stays within 0.5 to 9.5
            plt.arrow(start_x, start_y, dx, dy, head_width=0.3, head_length=0.3, fc='k', ec='k')
    # 0 = North, 1 = East, 2 = South, 3 = West
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    new_policy = choose_best_policy(grid3)
    reversed_policy = new_policy[::-1]
    grid4 = value_sweep(grid, new_policy, 0.2)
    plt.figure()
    plt.imshow(grid4, cmap='winter', interpolation='nearest',extent=[0, 10, 0, 10])
    plt.title('4')
    plt.colorbar()
    arrow_length = 0.4
    obstacle = [(2, 5), (2, 4), (2, 3), (2, 6), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
                (0, 9), (9, 0), (9, 1), (9, 2), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (1, 0), (2, 0),
                (3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9),
                (7, 9), (8, 9), (9, 9), (1, 8), (4, 4), (5, 4), (6, 4), (7, 4), (7, 5), (4, 7), (5, 7)]
    for i in range(init_policy.shape[0] - 1, -1, -1):  # Reverse the direction of i
        for j in range(init_policy.shape[1]):
            direction = reversed_policy[i, j]  # Corrected variable name
            if (i, j) in obstacle:
                continue
            if direction == 0:  # North
                dx, dy = 0, arrow_length
            elif direction == 1:  # East
                dx, dy = arrow_length, 0
            elif direction == 2:  # South
                dx, dy = 0, -arrow_length
            else:  # West
                dx, dy = -arrow_length, 0
            # Calculate starting and ending points of the arrow
            start_x = j + 0.5  # Adjusted for the shift
            start_y = i + 0.5  # Adjusted for the shift
            end_x = max(0.5, min(9.5, start_x + dx))  # Ensure end_x stays within 0.5 to 9.5
            end_y = max(0.5, min(9.5, start_y + dy))  # Ensure end_y stays within 0.5 to 9.5
            plt.arrow(start_x, start_y, dx, dy, head_width=0.3, head_length=0.3, fc='k', ec='k')
    # 0 = North, 1 = East, 2 = South, 3 = West
    plt.xlim(0, 10)
    plt.ylim(0, 10)

    # 0 = North, 1 = East, 2 = South, 3 = West
    plt.show()

transition_matrix = create_transition_matrix()
print_first_4()

init_policy = np.ones((rows, cols))
init_policy[:,:] = 1
n_iter = 0
check = True
while check:
    n_iter +=1
    grid1 = value_sweep(grid, init_policy, 0.001)
    new_policy = choose_best_policy(grid1)
    if not np.array_equal(init_policy, new_policy):
        check = True
    else:
        check = False
    init_policy = new_policy
print(n_iter)
reversed_policy = init_policy[::-1]
plt.figure()
plt.imshow(grid1, cmap='winter', interpolation='nearest', extent=[0, 10, 0, 10])
plt.title('Final Grid')
plt.colorbar()
# Define arrow lengths
arrow_length = 0.4

obstacle = [(2, 5), (2, 4), (2, 3),(2,6),(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,9),(9,0),(9,1),(9,2),(9,3),(9,4),(9,5),(9,6),(9,7),(9,8),(9,9),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0),(8,0),(9,0),(1,9),(2,9),(3,9),(4,9),(5,9),(6,9),(7,9),(8,9),(9,9),(1,8),(4,4),(5,4),(6,4),(7,4),(7,5),(4,7),(5,7)]
for i in range(init_policy.shape[0]-1, -1, -1):  # Reverse the direction of i
    for j in range(init_policy.shape[1]):
        direction = reversed_policy[i, j]  # Corrected variable name
        if (i, j) in obstacle:
            continue
        if direction == 0:  # North
            dx, dy = 0, arrow_length
        elif direction == 1:  # East
            dx, dy = arrow_length, 0
        elif direction == 2:  # South
            dx, dy = 0, -arrow_length
        else:  # West
            dx, dy = -arrow_length, 0
        # Calculate starting and ending points of the arrow
        start_x = j + 0.5  # Adjusted for the shift
        start_y = i + 0.5  # Adjusted for the shift
        end_x = max(0.5, min(9.5, start_x + dx))  # Ensure end_x stays within 0.5 to 9.5
        end_y = max(0.5, min(9.5, start_y + dy))  # Ensure end_y stays within 0.5 to 9.5
        plt.arrow(start_x, start_y, dx, dy, head_width=0.3, head_length=0.3, fc='k', ec='k')
# 0 = North, 1 = East, 2 = South, 3 = West
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.show()
# AFTER 5 iteration control does not change, and loop in the left part of the map, we basically are able to change the barrier or obstacle cost to solve this

