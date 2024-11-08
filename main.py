import cv2
import numpy as np
import matplotlib.pyplot as plt

# Đọc ảnh đầu vào
image_path = 'image1.jpg'  # Thay thế đường dẫn ảnh của bạn
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Đọc ảnh ở dạng grayscale

# Áp dụng bộ lọc Gaussian để làm mờ ảnh, giảm nhiễu
blurred_image = cv2.GaussianBlur(image, (5, 5), 1)

# Tính toán Gradient với toán tử Sobel
sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

# Tính toán Gradient với toán tử Prewitt
kernel_prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
kernel_prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
prewitt_x = cv2.filter2D(blurred_image, -1, kernel_prewitt_x)
prewitt_y = cv2.filter2D(blurred_image, -1, kernel_prewitt_y)
prewitt_magnitude = cv2.magnitude(prewitt_x.astype(float), prewitt_y.astype(float))

# Tính toán Gradient với toán tử Roberts
kernel_roberts_x = np.array([[1, 0], [0, -1]])
kernel_roberts_y = np.array([[0, 1], [-1, 0]])
roberts_x = cv2.filter2D(blurred_image, -1, kernel_roberts_x)
roberts_y = cv2.filter2D(blurred_image, -1, kernel_roberts_y)
roberts_magnitude = cv2.magnitude(roberts_x.astype(float), roberts_y.astype(float))

# Áp dụng toán tử Canny
canny_edges = cv2.Canny(blurred_image, 100, 200)

# Hiển thị kết quả
plt.figure(figsize=(12, 8))
plt.subplot(231), plt.imshow(image, cmap='gray'), plt.title("Ảnh gốc")
plt.subplot(232), plt.imshow(sobel_magnitude, cmap='gray'), plt.title("Toán tử Sobel")
plt.subplot(233), plt.imshow(prewitt_magnitude, cmap='gray'), plt.title("Toán tử Prewitt")
plt.subplot(234), plt.imshow(roberts_magnitude, cmap='gray'), plt.title("Toán tử Roberts")
plt.subplot(235), plt.imshow(canny_edges, cmap='gray'), plt.title("Toán tử Canny")
plt.subplot(236), plt.imshow(blurred_image, cmap='gray'), plt.title("Gaussian Blur")
plt.show()
