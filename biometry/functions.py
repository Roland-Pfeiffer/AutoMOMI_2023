"""
MIT License

Copyright (c) 2021 SINTEF Ocean

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np

def longest_spine(skeleton, x, y):
    # https://www.geeksforgeeks.org/diameter-of-a-binary-tree/
    finish = False
    next_step = [0, 0]
    spine = np.array([y, x])

    while not finish:
        x = x + next_step[0]
        y = y + next_step[1]

        skeleton[y, x] = False

        branch_roots = np.empty((0, 2), dtype=np.int)
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if skeleton[y + j, x + i]:
                    next_step = [i, j]
                    branch_roots = np.vstack((branch_roots, next_step))

        if branch_roots.shape[0] == 0:
            finish = True

        spine = np.vstack((spine, [y + next_step[1], x + next_step[0]]))
        if branch_roots.shape[0] > 1:
            longest_branch = None
            branch_length = 0
            for branch_root in branch_roots:
                branch_spine = longest_spine(skeleton, x + branch_root[0], y + branch_root[1])
                if branch_spine.shape[0] > branch_length:
                    longest_branch = branch_spine
                    branch_length = branch_spine.shape[0]
            return np.vstack((spine, longest_branch))

    return spine

