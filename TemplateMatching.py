#Stephen Esposito
#Homework 1: Finding Waldo
#February 5,2015
# import necessary libraries
import numpy
import matplotlib.pyplot as pyplot
import cv2

def main():
    # Load puzzle image
    #puzzle = cv2.imread("puzzle_1.jpg")
    puzzle = cv2.imread("puzzle_2.png")

    # Load query image
    #search = cv2.imread("query_1.jpg")
    search = cv2.imread("query_2.png")

    # Get the dimensions of Waldo's image
    width, height = search.shape[:2]

    # Perform template matching: Slide template image over scene image and
    # get scores for matches at each position
    tempmeth = cv2.TM_SQDIFF_NORMED # change this to use a different template matching method
    result = cv2.matchTemplate(puzzle, search, tempmeth)

    # Display the results using Matplotlib and save figure in file
    pyplot.imshow(result);
    pyplot.title("Matching Result"), pyplot.xticks([]), pyplot.yticks([])
    pyplot.savefig("Matplot.png");
    pyplot.show()

    # Find best scores
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    topLeft = max_loc
    if tempmeth in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        topLeft = min_loc

    # Make a copy of puzzle image
    copy = puzzle.copy()

    # Draw rectangle on image where the best score is found
    bottomRight = (topLeft[0] + height, topLeft[1] + width)
    # Draw a blue rectangle with line thickness of 2 pixles
    cv2.rectangle(copy, topLeft, bottomRight, 255, 2)

    # Display image using OpenCV
    cv2.imshow("Results", copy)

    # Save in file using OpenCV
    cv2.imwrite("Results.png", copy)

    # Wait for the user to hit enter then close everything
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass

if __name__ == "__main__":
    main()