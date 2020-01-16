import cv2

def edgeDetect(img_path):
    img_A = cv2.imread(image_path)
    _, img_B = image_contour(img_A)

def image_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    # edged = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
    return gray, edged