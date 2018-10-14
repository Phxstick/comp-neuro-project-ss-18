import os
import cv2
import numpy as np
import matplotlib.image as imglib
from task import *

COLOR_RED = (1, 0, 0)  # RGB red
ANTI_ALIASED = 16  # Anti-aliased line type


class Visualizer:

    @staticmethod
    def draw_grid(grid, width, height, value_function=None, state_labels=None,
                  policy=None, output_path="output.png"):
        """ Draw the given grid and output it to a file.
            
        If a value function is provided, color states in a grayscale according
        to their value (brighter means higher value) and draw arrows indicating
        the policy which tries to maximize the value function.

        Certain states can be labeled by providing a mapping state -> label.
        """
        cell_size = 25
        output_image = np.zeros((height * cell_size, width * cell_size, 3))
        min_gray = 0.0
        max_gray = 1.0
        if value_function is not None:
            min_value = min(value_function.values())
            max_value = max(value_function.values())
        for state in grid:
            x, y = state
            # Draw background of labeled states in white
            if state_labels is not None and state in state_labels:
                gray_value = 1.0
            # Draw the accessible fields in gray (scaled by values if given)
            else:
                gray_value = 0.5
            if value_function is not None and state in value_function:
                value = value_function[state]
                value_frac = (value - min_value) / (max_value - min_value)
                gray_value = min_gray + value_frac * (max_gray - min_gray)
            output_image[y * cell_size : (y + 1) * cell_size,
                         x * cell_size : (x + 1) * cell_size, :] = gray_value
            # Draw the policy as arrows
            if policy is not None and state in policy and not grid[state]:
                cell_center = (int((x + 0.5) * cell_size),
                               int((y + 0.5) * cell_size))
                Visualizer._draw_arrow(output_image, cell_center,
                                       int(cell_size/2), policy[state])
        # Draw state labels (if given)
        if state_labels is not None:
            for state in state_labels:
                x, y = state
                text = state_labels[state]
                cell_center = (int((x + 0.5) * cell_size),
                               int((y + 0.5) * cell_size))
                font_face = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 1
                (text_width, text_height), baseline = \
                        cv2.getTextSize(text, font_face, font_scale, 1)
                text_pos = (int(cell_center[0] - text_width / 2),
                            int(cell_center[1] + text_height / 2))
                color = (0, 0, 0) if value_function is None else COLOR_RED
                cv2.putText(output_image, text, text_pos, font_face, font_scale,
                            color=color, thickness=thickness)
        # Save image to file at given output path
        imglib.imsave(output_path, output_image, vmin=min_gray, vmax=max_gray,
                format=os.path.splitext(output_path)[1][1:])
        
    @staticmethod
    def _draw_arrow(image, start_point, length, action, thickness=1,
                    color=COLOR_RED):
        """ Draw an arrow into the given image. The starting point and length
            is given directly, the direction is determined by the given action.
        """
        x1, y1 = start_point
        arrow_ratio = 0.2
        arrow_tip_offset = 2
        if action == GO_TOP:
            x2, y2 = x1, y1 - length
            y3 = y4 = y1 - (1 - arrow_ratio) * length
            x3, x4 = x1 - arrow_tip_offset, x1 + arrow_tip_offset
        elif action == GO_RIGHT:
            x2, y2 = x1 + length, y1
            x3 = x4 = x1 + (1 - arrow_ratio) * length
            y3, y4 = y1 - arrow_tip_offset, y1 + arrow_tip_offset
        elif action == GO_BOTTOM:
            x2, y2 = x1, y1 + length
            y3 = y4 = y1 + (1 - arrow_ratio) * length
            x3, x4 = x1 - arrow_tip_offset, x1 + arrow_tip_offset
        elif action == GO_LEFT:
            x2, y2 = x1 - length, y1
            x3 = x4 = x1 - (1 - arrow_ratio) * length
            y3, y4 = y1 - arrow_tip_offset, y1 + arrow_tip_offset
        else:
            return
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)
        arrow_tip_points = np.array([(x2, y2), (x3, y3), (x4, y4)], dtype=int)
        arrow_tip_points.reshape((-1, 1, 2))
        cv2.fillConvexPoly(image, arrow_tip_points, color)

    @staticmethod
    def print_value_function(value_function, grid, width, height):
        """ Draws the given value function on the terminal. Used for debugging.
        """
        print()
        print("Value function:")
        print("==============================")
        for y in range(height):
            print()
            for x in range(width):
                if (x, y) in value_function:
                    print("%4.1f" % value_function[x, y], end=" ")
                elif (x, y) not in grid:
                    print("████", end=" ")
                else:
                    print("    ", end=" ")
        print()
        print()

    @staticmethod
    def print_policy(policy, grid, width, height):
        """ Draws the given policy on the terminal. Used for debugging.
        """
        print("Policy:")
        print("==============================")
        for y in range(height):
            print()
            for x in range(width):
                if (x, y) in policy:
                    if policy[x, y] == GO_TOP:
                        print("↑ ", end=" ")
                    elif policy[x, y] == GO_RIGHT:
                        print("→ ", end=" ")
                    elif policy[x, y] == GO_BOTTOM:
                        print("↓ ", end=" ")
                    elif policy[x, y] == GO_LEFT:
                        print("← ", end=" ")
                elif (x, y) not in grid:
                    print("██", end=" ")
                else:
                    print("  ", end=" ")
        print()
        print()
