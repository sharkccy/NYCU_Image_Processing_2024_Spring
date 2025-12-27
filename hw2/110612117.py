import cv2
import numpy as np
from matplotlib import pyplot as plt
def show_image(img_name, image):
    cv2.imshow(f'{img_name}', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_histogram(img):
    histogtam, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = histogtam.cumsum()
    cdf = cdf * histogtam.max() / cdf.max()
    plt.plot(cdf, color='b')
    plt.hist(img.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()

def histogram_equalization(img):
    histogtam, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = histogtam.cumsum()
    masked_cdf = np.ma.masked_equal(cdf, 0)
    masked_cdf = (masked_cdf - masked_cdf.min()) * 255 / (masked_cdf.max() - masked_cdf.min())
    masked_cdf = np.ma.filled(masked_cdf, 0).astype('uint8')
    histogtam_equalized_img = masked_cdf[img]
    show_histogram(img)
    show_histogram(histogtam_equalized_img)
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title('original_img')
    plt.subplot(122), plt.imshow(histogtam_equalized_img, cmap='gray'), plt.title('histogtam_equalized_img')
    plt.show()
    cv2.imwrite('Q1_histogtam_equalized_img.jpg', histogtam_equalized_img)

def histogram_specification(src_img, ref_img):
    hist_src, bins_src = np.histogram(src_img.flatten(), 256, [0, 256])
    cdf_src = hist_src.cumsum()
    # cdf_src_normalized = cdf_src * hist_src.max() / cdf_src.max()
    cdf_src_normalized = cdf_src / cdf_src.max()

    hist_ref, bins_ref = np.histogram(ref_img.flatten(), 256, [0, 256])
    cdf_ref = hist_ref.cumsum()
    # cdf_ref_normalized = cdf_ref * hist_ref.max() / cdf_ref.max()
    cdf_ref_normalized = cdf_ref  / cdf_ref.max()

    Mapping = np.zeros(256)
    for i in  range(256):
        Mapping[i] = np.argmin(np.abs(cdf_src_normalized[i] - cdf_ref_normalized))

    shape = src_img.shape
    output_img = np.zeros(shape)
    for(i, j), value in np.ndenumerate(src_img):
        output_img[i, j] = Mapping[value]
    cv2.imwrite('Q2_histogram_specification.jpg', output_img)
    show_histogram(output_img)
    plt.subplot(131), plt.imshow(src_img, cmap='gray'), plt.title('Source Image')
    plt.subplot(132), plt.imshow(ref_img, cmap='gray'), plt.title('Reference Image')
    plt.subplot(133), plt.imshow(output_img, cmap='gray'), plt.title('Output Image')
    plt.show()

img = cv2.imread('Q1.jpg', 0)
src = cv2.imread('Q2_source.jpg', 0)
ref = cv2.imread('Q2_reference.jpg', 0)
histogram_equalization(img)
show_histogram(src)
show_histogram(ref)
histogram_specification(src, ref)
