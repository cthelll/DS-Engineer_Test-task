def count_islands(grid):
    def explore_island(row, col):
        if 0 <= row < len(grid) and 0 <= col < len(grid[0]) and grid[row][col] == 1:
            grid[row][col] = 0
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for d_row, d_col in directions:
                explore_island(row + d_row, col + d_col)

    island_count = 0
    for row in range(len(grid)):
        for col in range(len(grid[0])):
            if grid[row][col] == 1:
                explore_island(row, col)
                island_count += 1
    return island_count

def run_tests():
    test_cases = [
        ([[0, 1, 0], [0, 0, 0], [0, 1, 1]], 2),
        ([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], 3),
        ([[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 1]], 2),
    ]

    for matrix, expected in test_cases:
        print("Matrix:")
        for row in matrix:
            print(row)
        matrix_copy = [row[:] for row in matrix]
        result = count_islands(matrix_copy)
        print(f"Islands found: {result}, Expected: {expected}\n")
        assert result == expected, f"Test failed for matrix: {matrix}"

    print("All tests passed successfully!")

def main():
    user_input = input("Enter rows, columns, and matrix values: ").split()
    rows, cols = int(user_input[0]), int(user_input[1])
    grid = [list(map(int, user_input[i * cols + 2:(i + 1) * cols + 2])) for i in range(rows)]
    print("Number of islands:", count_islands(grid))

if __name__ == "__main__":
    run_tests()
    main()
