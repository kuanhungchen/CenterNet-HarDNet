import cv2


def blurry(input, kernel_size):
    # kernel_size: interage, filter w h    
    # Kernel (5~20)
    kernel = (int(kernel_size), int(kernel_size))
    
    # Average Blur
    output = cv2.blur(input, kernel)
    
    return output
