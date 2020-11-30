import cv2


def pix(input, ratio):
    # Get input size
    width, height, _ = input.shape
    
    # Desired "pixelated" size (1/2 ~ 1/16)
    w, h = (width//int(ratio), height//int(ratio))
    
    # Resize input to "pixelated" size
    temp = cv2.resize(input, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Initialize output image
    output = cv2.resize(temp, (height, width), interpolation=cv2.INTER_NEAREST)

    return output
